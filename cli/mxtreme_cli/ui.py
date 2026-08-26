"""Terminal output helpers: headings, status lines and simple tables.

Everything user-facing in the CLI prints through here so the look stays consistent and colour is
decided in exactly one place. Colour is emitted only when stdout is a TTY, so piping the CLI's output
to a file or a log gives clean text.
"""

from __future__ import annotations

import shutil
import sys

_COLOR = sys.stdout.isatty()

_BOLD = "\033[1m" if _COLOR else ""
_DIM = "\033[2m" if _COLOR else ""
_RED = "\033[31m" if _COLOR else ""
_GREEN = "\033[32m" if _COLOR else ""
_YELLOW = "\033[33m" if _COLOR else ""
_CYAN = "\033[36m" if _COLOR else ""
_RESET = "\033[0m" if _COLOR else ""

#: Width used for rules and wrapping, clamped so it stays readable in a very wide terminal.
WIDTH = min(shutil.get_terminal_size((80, 24)).columns, 88)


def rule(char: str = "=") -> None:
    """Print a horizontal rule the width of the display."""
    print(_DIM + char * WIDTH + _RESET)


def banner(title: str, subtitle: str = "") -> None:
    """Print a screen heading -- the title of whatever menu or step is now in front of the user."""
    print()
    rule("=")
    print(f"{_BOLD}{title}{_RESET}")
    if subtitle:
        print(f"{_DIM}{subtitle}{_RESET}")
    rule("=")


def section(title: str) -> None:
    """Print a heading for a step within a screen."""
    print(f"\n{_BOLD}{title}{_RESET}")
    rule("-")


def info(message: str) -> None:
    """Print an ordinary status line."""
    print(message)


def hint(message: str) -> None:
    """Print secondary guidance -- how to answer a prompt, what a default means."""
    print(f"{_DIM}{message}{_RESET}")


def success(message: str) -> None:
    """Print a line reporting that something finished."""
    print(f"{_GREEN}{message}{_RESET}")


def warn(message: str) -> None:
    """Print a line reporting something the user should notice but that is not fatal."""
    print(f"{_YELLOW}! {message}{_RESET}")


def error(message: str) -> None:
    """Print a line reporting a failure."""
    print(f"{_RED}x {message}{_RESET}")


def bullet(message: str) -> None:
    """Print an indented list item."""
    print(f"  - {message}")


def numbered(index: int, label: str, detail: str = "", label_width: int = 0) -> None:
    """Print one selectable option of a menu or parameter list.

    :param index: The number the user types to choose this option.
    :param label: The option itself.
    :param detail: Secondary text shown to the right of the label.
    :param label_width: Pad the label to this width, so a list's details line up in a column.
    """
    padded = f"{label:<{label_width}}" if label_width else label
    tail = f"   {_DIM}{detail}{_RESET}" if detail else ""
    print(f"  {_CYAN}{index:>2}{_RESET}) {padded}{tail}")


def truncate(text: str, width: int) -> str:
    """Shorten ``text`` to ``width`` characters, marking what was cut with an ellipsis."""
    if width <= 1 or len(text) <= width:
        return text
    return text[: width - 1] + "…"


def key_values(rows: list[tuple[str, str]], indent: str = "  ") -> None:
    """Print aligned ``label: value`` rows, e.g. a summary of what is about to run.

    :param rows: ``(label, value)`` pairs, printed in order.
    :param indent: Prefix for every row.
    """
    if not rows:
        return
    width = max(len(label) for label, _ in rows)
    for label, value in rows:
        print(f"{indent}{label:<{width}}  {value}")


def block(text: str, indent: str = "  ") -> None:
    """Print a multi-line string with every line indented."""
    for line in text.splitlines():
        print(f"{indent}{line}")
