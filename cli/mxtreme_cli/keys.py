"""Single-keypress input, for menus the user drives with the arrow keys.

A terminal normally hands a program whole lines: nothing arrives until Enter is pressed. Reading one
key at a time means putting the terminal into *cbreak* mode for as long as a menu is on screen, which
is what :func:`raw_mode` does -- and, just as importantly, putting it back afterwards.

Not every terminal can do this. Windows has no ``termios``, and neither piped input nor a log file is
a terminal at all. :func:`available` answers that question, and every menu falls back to typed
numbers when the answer is no, so the CLI stays scriptable and testable.
"""

from __future__ import annotations

import contextlib
import os
import select
import sys
from typing import Iterator

try:  # POSIX only -- Windows falls back to the numbered menus
    import termios
    import tty
except ImportError:  # pragma: no cover -- neither the rig nor a Mac takes this path
    termios = None  # type: ignore[assignment]
    tty = None  # type: ignore[assignment]

#: Tokens :func:`read_key` returns for keys that are not a single printable character.
UP = "\x00up"
DOWN = "\x00down"
ENTER = "\x00enter"
ESCAPE = "\x00escape"
HOME = "\x00home"
END = "\x00end"

_ESC = b"\x1b"
_INTERRUPT = b"\x03"  # Ctrl-C, if the terminal is ever set up to deliver it as a byte
_EOF = b"\x04"  # Ctrl-D

#: What follows the escape byte for each key we care about. Terminals disagree: ``[`` is the normal
#: form and ``O`` the one sent in application cursor mode, which some ssh sessions turn on.
_SEQUENCES = {
    "[A": UP,
    "OA": UP,
    "[B": DOWN,
    "OB": DOWN,
    "[H": HOME,
    "OH": HOME,
    "[1~": HOME,
    "[F": END,
    "OF": END,
    "[4~": END,
}


def available() -> bool:
    """Whether this terminal can deliver keypresses one at a time.

    Both streams have to be a terminal: stdin to read keys without waiting for Enter, stdout to
    repaint the list in place.
    """
    if termios is None:
        return False
    try:
        return sys.stdin.isatty() and sys.stdout.isatty() and sys.stdin.fileno() >= 0
    except (AttributeError, OSError, ValueError):
        # A closed or substituted stdin (pytest's capture, some launchers) has no usable fileno.
        return False


@contextlib.contextmanager
def raw_mode(hide_cursor: bool = True) -> Iterator[None]:
    """Put the terminal into cbreak mode for the body of the ``with`` block.

    The previous settings are restored on every exit path, including exceptions: a terminal left
    without echo is unusable afterwards, so this must never leak.

    :param hide_cursor: Hide the blinking cursor while a menu is being redrawn, where it would
        otherwise sit at the end of the last line and flicker on every keypress.
    """
    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)

        # Keep ISIG on so Ctrl-C still raises KeyboardInterrupt. Every prompt in this CLI treats
        # that as "back out of this step", and Python has not been consistent across versions about
        # whether setcbreak() leaves the flag alone.
        mode = termios.tcgetattr(fd)
        mode[tty.LFLAG] |= termios.ISIG
        termios.tcsetattr(fd, termios.TCSANOW, mode)

        if hide_cursor:
            sys.stdout.write("\033[?25l")
            sys.stdout.flush()
        yield
    finally:
        if hide_cursor:
            sys.stdout.write("\033[?25h")
            sys.stdout.flush()
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)


def read_key(timeout: float | None = None) -> str | None:
    """Block until one key is pressed and return it. Only meaningful inside :func:`raw_mode`.

    :param timeout: Give up after this many seconds and return ``None``. Used by menus that have a
        live status line to repaint while the user is not pressing anything.
    :returns: One of this module's key tokens, the character itself for an ordinary key, or ``None``
        if the timeout ran out first.
    :raises KeyboardInterrupt: On Ctrl-C.
    :raises EOFError: On Ctrl-D, or if the input stream ends.
    """
    if timeout is not None and not _pending(timeout):
        return None

    fd = sys.stdin.fileno()
    data = os.read(fd, 6)

    if not data:
        raise EOFError
    if data == _INTERRUPT:
        raise KeyboardInterrupt
    if data == _EOF:
        raise EOFError
    if data in (b"\r", b"\n"):
        return ENTER

    if data.startswith(_ESC):
        # An arrow key is three bytes that almost always arrive together, but a slow link can split
        # them. A lone Esc is followed by nothing, so a brief wait tells the two apart without
        # making a real Esc press feel sluggish.
        if data == _ESC and _pending():
            data += os.read(fd, 5)
        if data == _ESC:
            return ESCAPE
        return _SEQUENCES.get(data[1:].decode("ascii", "replace"), ESCAPE)

    # Anything else is an ordinary key. Decoding is per-keypress, so a multi-byte character read in
    # pieces degrades to a replacement character rather than raising -- menus only act on ASCII.
    return data.decode("utf-8", "replace")[:1]


def _pending(timeout: float = 0.02) -> bool:
    """Whether more input is already waiting, used to complete a split escape sequence."""
    return bool(select.select([sys.stdin], [], [], timeout)[0])
