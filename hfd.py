#!/usr/bin/env python3
import os
import sys
import asyncio
import threading
from pathlib import Path

import httpx
from readchar import key, readkey
from rich.live import Live
from rich.table import Table
from rich.console import Console
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    DownloadColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from huggingface_hub import HfApi, hf_hub_url

# Configuration
CONCURRENT_FILES = 3  # Parallel files if multiple selected
THREADS_PER_FILE = 8  # Multithreaded single file download
LOCAL_DIR = Path(os.getenv("HF_LOCAL_DIR", Path.home() / "models/"))
_HF_TOKEN = os.getenv("HF_TOKEN")

console = Console()
semaphore = asyncio.Semaphore(CONCURRENT_FILES)


def get_key():
    """Read a keypress from stdin."""
    k = readkey()
    if k == key.CTRL_C:
        raise KeyboardInterrupt
    if k == key.UP:
        return "UP"
    if k == key.DOWN:
        return "DOWN"
    if k == key.ENTER:
        return "ENTER"
    if k == key.SPACE:
        return "SPACE"
    if k == key.ESC:
        return "ESC"
    return k


class HFDWrapper:
    """Pull repo info from hf API."""

    def __init__(self):
        self.api = HfApi(token=_HF_TOKEN)
        self.active_files = set()  # Track files currently being written
        LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    def parse_url(self, url: str):
        """Parses an HF blob URL into repo_id and filename."""
        url = url.replace("huggingface.co/", "hf.co/")
        if "hf.co/" in url:
            parts = url.split("hf.co/")[1].split("/")
            repo_id = f"{parts[0]}/{parts[1]}"
            filename = parts[-1] if "blob/main" in url else None
            return repo_id, filename
        return url, None

    def get_vram_est(self, size_bytes: int) -> str:
        """Rough estimate: File size + 2GB buffer for context/KV cache."""
        gb = size_bytes / (1024**3)
        return f"{gb + 2.0:.2f} GB"

    def get_ggufs(self, repo_id: str) -> list:
        """Parse GGUF info from hf API."""
        files = [f for f in self.api.list_repo_files(repo_id) if f.endswith(".gguf")]
        if not files:
            console.print("[red]No GGUF files found in repository.[/red]")
            return []

        # Get file metadata for sizes
        repo_info = self.api.model_info(repo_id, files_metadata=True)
        siblings = repo_info.siblings or []
        ggufs = []
        for sibling in siblings:
            if sibling.rfilename.endswith(".gguf"):
                size = sibling.size or 0
                ggufs.append({"name": sibling.rfilename, "size": size})

        return ggufs

    async def download_chunk(  # noqa: PLR0913
        self,
        client,
        url,
        start,
        end,
        progress,
        sub_task_id,
        main_task_id,
        file_obj,
        lock,
        retries=3,
    ):
        """Download chunks of a single file."""
        headers = {"Range": f"bytes={start}-{end}"}
        if _HF_TOKEN:
            headers["Authorization"] = f"Bearer {_HF_TOKEN}"

        for attempt in range(retries):
            try:
                async with client.stream(
                    "GET", url, headers=headers, follow_redirects=True
                ) as response:
                    response.raise_for_status()

                    progress.update(sub_task_id, completed=0)

                    current_pos = start
                    async for chunk in response.aiter_bytes():
                        chunk_len = len(chunk)
                        with lock:
                            file_obj.seek(current_pos)
                            file_obj.write(chunk)

                        current_pos += chunk_len
                        progress.update(sub_task_id, advance=chunk_len)
                        progress.update(main_task_id, advance=chunk_len)
                    progress.remove_task(sub_task_id)
                    return  # Success!
            except (
                OSError,
                httpx.HTTPError,
                httpx.ConnectError,
                KeyboardInterrupt,
            ) as _e:
                if attempt < retries - 1:
                    msg = f"[yellow]  ! Chunk retry {attempt + 1}/{retries} due to: {_e}[/yellow]"
                    progress.console.print(msg)
                    await asyncio.sleep(1)  # Backoff
                else:
                    msg = f"[red]  ✘ Chunk failed after {retries} attempts.[/red]"
                    progress.console.print(msg)
                    raise

    async def fast_download(self, url: str, filename: str, total_size: int, overall_progress):
        """Dowmload files fast."""
        async with semaphore:
            dest = LOCAL_DIR / filename
            self.active_files.add(dest)

            main_task = overall_progress.add_task(f"[bold cyan]{filename}", total=total_size)

            try:
                with dest.open(mode="wb") as f:
                    f.truncate(total_size)
                chunk_size = total_size // THREADS_PER_FILE
                lock = threading.Lock()

                async with httpx.AsyncClient(
                    timeout=30, limits=httpx.Limits(max_connections=THREADS_PER_FILE)
                ) as client:
                    tasks = []
                    with dest.open(mode="r+b") as f:
                        for i in range(THREADS_PER_FILE):
                            start = i * chunk_size
                            end = (
                                total_size - 1
                                if i == THREADS_PER_FILE - 1
                                else (start + chunk_size - 1)
                            )
                            sub_task = overall_progress.add_task(
                                f"  [dim]↳ thread-{i + 1}[/dim]",
                                total=(end - start + 1),
                            )

                            tasks.append(
                                self.download_chunk(
                                    client,
                                    url,
                                    start,
                                    end,
                                    overall_progress,
                                    sub_task,
                                    main_task,
                                    f,
                                    lock,
                                )
                            )

                        await asyncio.gather(*tasks)
                self.active_files.remove(dest)
            except (Exception,) as _e:
                msg = f"[red]Error downloading {filename}: {_e}[/red]"
                overall_progress.console.print(msg)
                raise

    def generate_tui(self, ggufs: list, selected: set, _idx: int, repo_id: str) -> Table:
        """Make a new table."""
        grid = Table.grid(expand=True)
        table = Table(title=f"{repo_id}", title_style="bold cyan")
        table.add_column("ID", justify="center")
        table.add_column("Filename")
        table.add_column("Size", justify="right", style="bright_cyan")
        table.add_column("Est. VRAM", justify="right", style="cyan2")

        grid.add_row(table)
        grid.add_row(
            "\n[yellow]Use 🔼/🔽 arrows, Space to toggle select,"
            " Enter to download, ESC to exit.[/yellow]"
        )
        for i, g in enumerate(ggufs):
            mark = "➤" if i == _idx else " "
            check = "[ x ]" if i in selected else "[   ]"
            style = "reverse" if i == _idx else "bold green" if i in selected else ""
            table.add_row(
                f"{mark} {i:2d} {check}",
                g["name"],
                f"{g['size'] / (1024**2):.1f} MB",
                self.get_vram_est(g["size"]),
                style=style,
            )

        return grid

    def interactive_menu(self, repo_id: str):
        """TUI for selecting GGUF files."""
        ggufs = self.get_ggufs(repo_id)
        selected = set()
        # Auto-select logic
        for _i, _g in enumerate(ggufs):
            if "Q4_K_M" in _g["name"] or "mmproj" in _g["name"]:
                selected.add(_i)

        _idx = 0

        with Live(
            self.generate_tui(ggufs, selected, _idx, repo_id), console=console, screen=True
        ) as tui:
            while True:
                ch = get_key()
                if ch == "SPACE":
                    if _idx in selected:
                        selected.remove(_idx)
                    else:
                        selected.add(_idx)
                elif ch == "UP":
                    _idx = max(0, _idx - 1)
                elif ch == "DOWN":
                    _idx = min(len(ggufs) - 1, _idx + 1)
                elif ch == "ENTER":
                    break
                elif ch == "ESC":
                    return []
                tui.update(self.generate_tui(ggufs, selected, _idx, repo_id))
        return [ggufs[i] for i in selected]

    async def run(self, arg: str):
        """Main run loop."""
        repo_id, filename = self.parse_url(arg)

        to_download = []
        if filename:
            # Direct link provided
            url = hf_hub_url(repo_id, filename)
            info = self.api.model_info(repo_id, files_metadata=True)
            siblings = info.siblings or []
            size = next(f.size for f in siblings if f.rfilename == filename)
            to_download.append({"name": filename, "size": size, "url": url})
        else:
            # Repo ID provided, show TUI
            selected = self.interactive_menu(repo_id)
            for s in selected:
                s["url"] = hf_hub_url(repo_id, s["name"])
                to_download.append(s)

        if not to_download:
            return

        progress = Progress(
            TextColumn("[bold blue]{task.description}", justify="right"),
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.1f}%",
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
        )

        with Live(progress, console=console, refresh_per_second=10):
            dl_tasks = [
                self.fast_download(f["url"], f["name"], f["size"], progress) for f in to_download
            ]
            await asyncio.gather(*dl_tasks)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        console.print("[yellow]Usage:[/yellow] hfd.py <url_or_repo>")
        sys.exit(1)

    wrapper = HFDWrapper()

    try:
        asyncio.run(wrapper.run(sys.argv[1]))
    except (Exception, KeyboardInterrupt) as e:
        if type(e) is KeyboardInterrupt:
            console.print("\n[bold red]Stopping...[/bold red]")
        else:
            console.print(f"[red]Stopping due to Error: {e}[/red]")

        for file_path in list(wrapper.active_files):
            console.print(f"[purple]Cleaning up partial download:[/purple] {file_path.name}")
            if file_path.exists():
                try:
                    file_path.unlink()
                    console.print(f"  [dim]Deleted partial: {file_path.name}[/dim]")
                except (Exception,) as e:
                    console.print(f"  [red]Failed to delete {file_path.name}: {e}[/red]")
        console.print("[bold yellow]hfd exit.[/bold yellow]")
        sys.exit(0)
    else:
        console.print("\n[bold green]🗸[/bold green] [white]hfd exit.[/white]")
