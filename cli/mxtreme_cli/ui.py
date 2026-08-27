"""Terminal output helpers: headings, status lines and simple tables.

Everything user-facing in the CLI prints through here so the look stays consistent and colour is
decided in exactly one place. Colour is emitted only when stdout is a TTY, so piping the CLI's output
to a file or a log gives clean text.
"""

from __future__ import annotations

import shutil
import sys
from typing import Callable

_COLOR = sys.stdout.isatty()

_BOLD = "\033[1m" if _COLOR else ""
_DIM = "\033[2m" if _COLOR else ""
_RED = "\033[31m" if _COLOR else ""
_GREEN = "\033[32m" if _COLOR else ""
_YELLOW = "\033[33m" if _COLOR else ""
_CYAN = "\033[36m" if _COLOR else ""
_REVERSE = "\033[7m" if _COLOR else ""
_RESET = "\033[0m" if _COLOR else ""

#: Blanks the line the cursor is on, so a redrawn row cannot leave a tail of the longer row it
#: replaced. Harmless when the cursor is already at the start of an empty line.
CLEAR_LINE = "\033[2K" if _COLOR else ""

#: Marks the highlighted row of an arrow-driven menu.
MARKER = "\u25b8"

#: Name shown above every screen heading, so the user always knows what they are in.
PRODUCT = "MXtreme"

#: Width used for rules and wrapping, clamped so it stays readable in a very wide terminal.
WIDTH = min(shutil.get_terminal_size((80, 24)).columns, 88)


def cursor_up(lines: int) -> None:
    """Move the cursor up ``lines`` rows, so a menu can be redrawn over itself.

    Repainting in place is what keeps the scrollback intact: an arrow-key menu that cleared the
    screen would take the scan's progress log with it. A no-op where there is nothing to repaint.
    """
    if _COLOR and lines > 0:
        sys.stdout.write(f"\033[{lines}A")
        sys.stdout.flush()


def rule(char: str = "=") -> None:
    """Print a horizontal rule the width of the display."""
    print(_DIM + char * WIDTH + _RESET)


#: Supplies the lines shown under every banner, or ``None`` for none. The CLI points this at the
#: background scan registry at startup; keeping it a hook rather than an import is what lets
#: :mod:`mxtreme_cli.ui` stay the bottom of the import graph.
status_provider: Callable[[], list[str]] | None = None


def status() -> list[str]:
    """Whatever the status provider currently reports, or nothing if it is unset or broken."""
    if status_provider is None:
        return []
    try:
        return status_provider()
    except Exception:  # noqa: BLE001 -- a broken status line must not take a screen down with it
        return []


def banner(title: str, subtitle: str = "", product: str = "") -> None:
    """Print a screen heading -- the title of whatever menu or step is now in front of the user.

    :param title: The screen's own name.
    :param subtitle: One line saying what the screen is for.
    :param product: Shown above the title, so every screen carries the same identity. Screens pass
        nothing and get the CLI's own name.
    """
    print()
    rule("=")
    print(f"{_DIM}{product or PRODUCT}{_RESET}")
    print(f"{_BOLD}{title}{_RESET}")
    if subtitle:
        print(f"{_DIM}{subtitle}{_RESET}")
    for line in status():
        print(f"{_CYAN}\u2022 {line}{_RESET}")
    rule("=")


def section(title: str) -> None:
    """Print a heading for a step within a screen."""
    print(f"\n{_BOLD}{title}{_RESET}")
    rule("-")


def info(message: str) -> None:
    """Print an ordinary status line."""
    print(message)


def hint(message: str) -> None:
    """Print secondary guidance -- how to answer a prompt, what a default means.

    Cleared before it is written, because the key hint under an arrow-driven menu is redrawn in
    place along with the rows above it.
    """
    print(f"{CLEAR_LINE}{_DIM}{message}{_RESET}")


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


def numbered(
    index: int | None,
    label: str,
    detail: str = "",
    label_width: int = 0,
    selected: bool = False,
) -> None:
    """Print one selectable option of a menu or parameter list.

    :param index: The number the user can type to jump to this option, or ``None`` for a row that
        is only reachable with the arrow keys -- an action rather than a value.
    :param label: The option itself.
    :param detail: Secondary text shown to the right of the label.
    :param label_width: Pad the label to this width, so a list's details line up in a column.
    :param selected: Draw this row as the cursor position of an arrow-driven menu.
    """
    number = "  " if index is None else f"{index:>2}"
    gap = "  " if index is None else ") "
    padded = f"{label:<{label_width}}" if label_width else label

    # No row may exceed the display width. A row that wraps costs two physical lines, which throws
    # out the line count cursor_up() moves by and leaves an arrow-driven list redrawing over itself.
    body = max(WIDTH - 2, 20)
    head = truncate(f"{number}{gap}{padded}", body)
    room = body - len(head) - 3
    detail = truncate(detail, room) if detail and room > 0 else ""

    if selected:
        # The row's own colours are dropped here: dim or cyan text inside reverse video reads as a
        # smudge rather than as emphasis. Padding to the full width makes one solid bar.
        row = f"{head}   {detail}" if detail else head
        print(f"{CLEAR_LINE}{_CYAN}{MARKER}{_RESET} {_REVERSE}{row:<{body}}{_RESET}")
        return

    tail = f"   {_DIM}{detail}{_RESET}" if detail else ""
    print(f"{CLEAR_LINE}  {_CYAN}{number}{_RESET}{gap}{padded}{tail}")


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
