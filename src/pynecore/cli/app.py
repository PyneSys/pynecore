import os
import sys
import tomllib
from pathlib import Path
from dataclasses import dataclass

try:
    import typer
except ImportError:
    print("You need the Pyne CLI dependencies to run Pyne CLI. "
          "Please run `pip install pynesys-pynecore[cli]`.", file=sys.stderr)
    raise SystemExit(1)

__all__ = ["app", "app_state"]

app = typer.Typer(
    invoke_without_command=True,
    context_settings={"help_option_names": ["-h", "--help"]},
    rich_markup_mode="rich",
    # Never render frame locals in the crash traceback. A provider/broker
    # error boundary can carry credential-bearing configuration objects
    # (session tokens, API secrets) in its local variables; Typer's default
    # ``pretty_exceptions_show_locals=True`` would dump those verbatim into
    # the durable CLI transcript. Keeping the traceback (still useful for
    # diagnosing where the failure surfaced) but stripping the locals is the
    # CLI-level fix — no exception should ever leak secrets to the console.
    pretty_exceptions_show_locals=False,
)


@dataclass(slots=True)
class AppState:
    """
    Application state variables
    """
    _workdir: Path | None = None

    def __post_init__(self):
        # If no workdir was explicitly set, find it automatically
        if self._workdir is None:
            self._workdir = self._find_workdir()

    # noinspection DuplicatedCode
    @staticmethod
    def _is_workdir(path: Path) -> bool:
        """
        Check if the directory itself is a workdir, marked by a `.pyne` file with a
        supported workdir structure version.
        """
        marker = path / ".pyne"
        if not marker.is_file():
            return False
        try:
            with marker.open("rb") as f:
                data = tomllib.load(f)
            major = int(str(data.get("workdir_version", "")).split(".")[0])
        except (OSError, ValueError, tomllib.TOMLDecodeError):
            return False
        return major >= 1

    # noinspection DuplicatedCode
    @staticmethod
    def _find_workdir() -> Path:
        """
        Recursively searches for the workdir folder in the current and parent directories.
        A directory containing a valid `.pyne` marker file is itself a workdir.
        If not found, uses the current directory.
        """
        current_dir = Path().resolve()
        # Maximum search depth in parent directories
        max_depth = 10

        # Start with the current directory and move upwards
        for _ in range(max_depth):
            if AppState._is_workdir(current_dir):
                return current_dir

            workdir_candidate = current_dir / "workdir"
            if workdir_candidate.exists() and workdir_candidate.is_dir():
                return workdir_candidate

            # Check if we've reached the filesystem root
            parent = current_dir.parent
            if parent == current_dir:  # Reached the root
                break

            current_dir = parent

        # If no workdir folder was found, use the current directory
        return (Path() / "workdir").resolve()

    @property
    def workdir(self) -> Path:
        """
        The path to the working directory
        """
        return self._workdir or self._find_workdir()

    @workdir.setter
    def workdir(self, path: Path):
        """
        Sets the working directory
        """
        self._workdir = path

    @property
    def scripts_dir(self):
        return self.workdir / "scripts"

    @property
    def output_dir(self):
        return self.workdir / "output"

    @property
    def data_dir(self):
        """Directory for OHLCV data files.

        Honours the ``PYNE_DATA_DIR`` environment variable so a caller can direct
        downloads to an explicit location (e.g. a per-run cache) without moving the
        whole workdir; credentials and config keep resolving from ``config_dir``.
        Defaults to ``workdir/data``.
        """
        env = os.environ.get("PYNE_DATA_DIR")
        return Path(env) if env else self.workdir / "data"

    @property
    def config_dir(self):
        return self.workdir / "config"

    @property
    def cache_dir(self):
        return self.workdir / "cache"


app_state = AppState()


def _get_pine_tree_art():
    """Returns the colorized pine tree ASCII art as a list of strings."""
    return [
        "           [dark_green] █ [/]              ",
        "           [blue on dark_green]   [/]            ",
        "         [blue on dark_green] ▄▄▄▄  [/]          ",
        "       [blue on dark_green]  █▄████[/][yellow on dark_green]▄▄ [/]        ",
        "     [blue on dark_green]  ▄▇▇▇▇▇██[/][dark_green on yellow]   [/][yellow on dark_green]  [/]      ",
        "   [blue on dark_green]    ███[/][dark_green on yellow]  ▁▁▁▁▁▄[/][yellow on dark_green]    [/]    ",
        " [blue on dark_green]       ▀▀[/][dark_green on yellow]    ▄ [/][yellow on dark_green]        [/]  ",
        "         [yellow on dark_green]  ▀▀▀▀ [/]             "
    ]


def _get_base_text_art():
    """Returns the base 'Pyne' text ASCII art as a list of strings."""
    return [
        "  ____",
        " |  _ \\ _   _ _ __   ___",
        " | |_) | | | | '_ \\ / _ \\",
        " |  __/| |_| | | | |  __/",
        " |_|    \\__, |_| |_|\\___|",
        "        |___/       Core™"
    ]


def _get_pyne_text_art():
    """Returns the colorized 'Pyne' text ASCII art with diagonal-like gradient effect."""
    base_art = _get_base_text_art()

    colors = ["bright_red", "magenta", "bright_magenta", "bright_blue"]
    cols = [7, 13, 20, 1000]
    column_lens = [len(line) for line in base_art]
    max_len = max(column_lens)

    # Apply horizontal gradient - left to right color transition
    gradient_art = []
    for line in base_art:
        color_num = 0
        changed = True
        color = colors[color_num]
        new_line = ""
        for c in range(max_len):
            if c >= len(line):
                line += " "
            if c == cols[color_num]:
                color_num += 1
                color = colors[color_num]
                changed = True
            if changed:
                new_line += f"[{color}]{line[c]}"
                changed = False
            else:
                new_line += line[c]

        new_line = new_line.replace("\\[", "\\\\[")
        gradient_art.append(new_line)

    return gradient_art


def _get_plain_text_art():
    """Returns the plain text ASCII art as a list of strings."""
    return _get_base_text_art()


def print_logo():
    try:
        from rich import print as rprint
        from rich.console import Console

        console = Console()

        if console.color_system and not os.getenv("NO_COLOR"):
            tree_art = _get_pine_tree_art()
            text_art = _get_pyne_text_art()

            rprint("")

            # Print tree and text side by side
            max_lines = max(len(tree_art), len(text_art))
            for i in range(max_lines):
                tree_line = tree_art[i] if i < len(tree_art) else ""
                text_line = text_art[i - 1] if i >= 1 and (i - 1) < len(text_art) else ""

                # Combine tree and text with proper spacing
                combined_line = f"{tree_line:<40} {text_line}"
                rprint(combined_line)
        else:
            raise ImportError  # Force the except block

    except (UnicodeEncodeError, ImportError):
        plain_text = _get_plain_text_art()
        for line in plain_text:
            print(line)


if (not os.getenv("PYNE_NO_LOGO") and not os.getenv("PYNE_QUIET")
        # Don't print logo when completion is requested
        and not os.getenv("_TYPER_COMPLETE_ARGS")):
    print_logo()
