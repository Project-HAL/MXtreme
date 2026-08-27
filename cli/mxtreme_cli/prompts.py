"""Terminal input helpers: menus, typed prompts and the parameter editor.

Three things every screen in the CLI needs:

- :func:`menu` -- pick one of a list of options, or go back.
- :func:`ask` and the ``parse_*`` functions -- read one value, re-prompting until it parses.
- :func:`edit_params` -- show a dataclass of parameters with their current values and let the user
  change any of them, or press Enter to run with the defaults.

Lists are driven with the arrow keys where the terminal allows it (see :mod:`mxtreme_cli.keys`) and
fall back to typed numbers where it does not -- piped input, a log file, Windows. Both paths are
kept: the numbered one is what makes the CLI scriptable, and it is the only path a non-interactive
run can take.

Every prompt treats Ctrl-C and end-of-input as "back out of this step" by raising :class:`Cancelled`,
which the calling menu catches; the top-level menu treats it as "quit".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from mxtreme_cli import keys, ui

class Cancelled(Exception):
    """Raised when the user backs out of a prompt with Ctrl-C or end-of-input."""


def _read(prompt: str) -> str:
    """Read one line, turning Ctrl-C and EOF into :class:`Cancelled`."""
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        raise Cancelled from None


# --------------------------------------------------------------------------------------------------
# Parsers -- each takes the raw string and returns the typed value, raising ValueError on bad input.
# --------------------------------------------------------------------------------------------------


def parse_text(raw: str) -> str:
    """Accept any non-empty string."""
    if not raw:
        raise ValueError("Enter a value.")
    return raw


def parse_int(minimum: int | None = None, maximum: int | None = None) -> Callable[[str], int]:
    """Build a parser for an integer, optionally bounded.

    :param minimum: Smallest accepted value, or ``None`` for unbounded.
    :param maximum: Largest accepted value, or ``None`` for unbounded.
    """

    def parse(raw: str) -> int:
        try:
            value = int(raw)
        except ValueError:
            raise ValueError(f"'{raw}' is not a whole number.") from None
        if minimum is not None and value < minimum:
            raise ValueError(f"Must be at least {minimum}.")
        if maximum is not None and value > maximum:
            raise ValueError(f"Must be at most {maximum}.")
        return value

    return parse


def parse_float(minimum: float | None = None, maximum: float | None = None) -> Callable[[str], float]:
    """Build a parser for a real number, optionally bounded."""

    def parse(raw: str) -> float:
        try:
            value = float(raw)
        except ValueError:
            raise ValueError(f"'{raw}' is not a number.") from None
        if minimum is not None and value < minimum:
            raise ValueError(f"Must be at least {minimum}.")
        if maximum is not None and value > maximum:
            raise ValueError(f"Must be at most {maximum}.")
        return value

    return parse


def parse_int_list(minimum: int | None = None, maximum: int | None = None) -> Callable[[str], list[int]]:
    """Build a parser for a comma- or space-separated list of integers, e.g. wells ``0, 1, 2``."""
    parse_one = parse_int(minimum, maximum)

    def parse(raw: str) -> list[int]:
        tokens = [t for t in raw.replace(",", " ").split() if t]
        if not tokens:
            raise ValueError("Enter at least one number.")
        return [parse_one(t) for t in tokens]

    return parse


def parse_str_list(raw: str) -> list[str]:
    """Parse a comma-separated list of labels; an empty string means an empty list."""
    return [t.strip() for t in raw.split(",") if t.strip()]


def parse_dir(raw: str) -> str:
    """Parse a directory path, expanding ``~``. The directory need not exist yet."""
    if not raw:
        raise ValueError("Enter a path.")
    return str(Path(raw).expanduser())


def parse_existing_file(suffix: str | None = None) -> Callable[[str], Path]:
    """Build a parser for the path of a file that must already exist.

    :param suffix: When given, the file must end with it (e.g. ``".h5"``).
    """

    def parse(raw: str) -> Path:
        if not raw:
            raise ValueError("Enter a path.")
        # Drag-and-drop into a terminal often leaves surrounding quotes and an escaped space.
        cleaned = raw.strip().strip("'\"").replace("\\ ", " ")
        path = Path(cleaned).expanduser()
        if not path.exists():
            raise ValueError(f"No such file: {path}")
        if not path.is_file():
            raise ValueError(f"Not a file: {path}")
        if suffix and not path.name.endswith(suffix):
            raise ValueError(f"Expected a {suffix} file, got: {path.name}")
        return path

    return parse


def parse_optional_int(raw: str) -> int | None:
    """Parse an integer, mapping the empty string to ``None``."""
    if not raw:
        return None
    return parse_int()(raw)


# --------------------------------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------------------------------


def ask(
    prompt: str,
    default: Any = None,
    parse: Callable[[str], Any] = parse_text,
    allow_blank: bool = False,
) -> Any:
    """Ask for one value, re-prompting until it parses.

    :param prompt: Question to show, without the trailing colon.
    :param default: Returned when the user presses Enter. Shown in the prompt.
    :param parse: Parser applied to the raw input; raise ``ValueError`` to reject it.
    :param allow_blank: When ``True`` and there is no default, an empty answer is passed to ``parse``
        rather than re-prompting -- for genuinely optional values.
    :raises Cancelled: If the user backs out.
    :returns: The parsed value, or ``default``.
    """
    shown = f"{prompt} [{default}]: " if default is not None else f"{prompt}: "
    while True:
        raw = _read(shown)
        if not raw and default is not None:
            return default
        if not raw and not allow_blank:
            ui.error("Enter a value.")
            continue
        try:
            return parse(raw)
        except ValueError as exc:
            ui.error(str(exc))


def confirm(prompt: str, default: bool = True) -> bool:
    """Ask a yes/no question.

    :param prompt: Question to show.
    :param default: Answer used when the user presses Enter.
    :raises Cancelled: If the user backs out.
    """
    shown = f"{prompt} [{'Y/n' if default else 'y/N'}]: "
    while True:
        raw = _read(shown).lower()
        if not raw:
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        ui.error("Answer y or n.")


def pause(message: str = "Press Enter to continue") -> None:
    """Wait for the user to acknowledge, so output is not scrolled away by the next menu."""
    try:
        input(f"\n{message}... ")
    except (EOFError, KeyboardInterrupt):
        print()


#: A row's number is unambiguous only while every row has a single digit. Past that a keypress
#: cannot tell "1" from the first half of "12", so digits move the cursor instead of choosing.
_DIRECT_LIMIT = 10

#: Shown under an arrow-driven list. Kept short: it is redrawn on every keypress.
_KEY_HINT = "\u2191/\u2193 move \u00b7 Enter select \u00b7 number jumps \u00b7 q back"
_EDIT_HINT = "\u2191/\u2193 move \u00b7 Enter select \u00b7 number jumps \u00b7 q cancel"

#: How often a menu repaints while something is running in the background, in seconds.
_STATUS_REFRESH = 1.0


def menu(
    options: Sequence[tuple[Any, str] | tuple[Any, str, str]],
    prompt: str = "Select an option",
    back_label: str = "Back",
    refresh: float | None = None,
) -> Any | None:
    """Show a menu and return the chosen option's value.

    Driven with the arrow keys on a terminal that supports it, and with typed numbers everywhere
    else; the two behave identically from the caller's side.

    :param options: ``(value, label)`` or ``(value, label, detail)`` tuples, listed in order from 1.
    :param prompt: Question shown with the list.
    :param back_label: Label for option ``0``, which returns ``None``. Empty for a menu with no way
        back, where only Ctrl-C leaves.
    :param refresh: Repaint every this many seconds even when no key is pressed, so a live status
        line stays current. Left unset, a menu shown while something is running picks its own.
    :returns: The chosen value, or ``None`` if the user chose to go back.
    :raises Cancelled: If the user backs out with Ctrl-C.
    """
    if refresh is None and ui.status():
        # Any menu shown while a scan is running gets the live block, not just the main one.
        refresh = _STATUS_REFRESH
    if keys.available():
        return _menu_keys(options, prompt, back_label, refresh)
    return _menu_numbers(options, prompt, back_label)


def _menu_numbers(
    options: Sequence[tuple[Any, str] | tuple[Any, str, str]],
    prompt: str,
    back_label: str,
) -> Any | None:
    """Menu for terminals that cannot read single keypresses: type a number, press Enter."""
    label_width = max(len(option[1]) for option in options)
    while True:
        print()
        for line in ui.status():  # no live repaint here, but the menu still says what is running
            ui.info(f"  {line}")
        for index, option in enumerate(options, start=1):
            _value, label, *rest = option
            ui.numbered(index, label, rest[0] if rest else "", label_width)
        ui.numbered(0, back_label)

        raw = _read(f"\n{prompt}: ")
        if raw == "0" or (not raw and back_label):
            return None
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1][0]
        ui.error(f"Enter a number from 0 to {len(options)}.")


def _menu_keys(
    options: Sequence[tuple[Any, str] | tuple[Any, str, str]],
    prompt: str,
    back_label: str,
    refresh: float | None = None,
) -> Any | None:
    """Menu driven with the arrow keys, redrawn in place as the cursor moves.

    A live status -- a scan running in the background -- is drawn inside this repaint region rather
    than left to the banner above, which is only a snapshot from when the screen was entered. That
    is what keeps the progress bar moving while the menu just sits there.
    """
    label_width = max(len(option[1]) for option in options)

    # The back entry is one more row rather than a hidden key, so everything on offer is on screen.
    rows: list[tuple[int, Any, str, str]] = [
        (index, option[0], option[1], option[2] if len(option) > 2 else "")
        for index, option in enumerate(options, start=1)
    ]
    if back_label:
        rows.append((0, None, back_label, ""))

    position = 0
    painted = 0  # lines the last paint used, which is how far back up the next one starts

    def draw() -> None:
        nonlocal painted
        if painted:
            ui.cursor_up(painted)

        lines = 0
        if refresh is not None:
            for line in ui.status():
                ui.info(f"{ui.CLEAR_LINE}  {line}")
                lines += 1
            if lines:
                print(ui.CLEAR_LINE)
                lines += 1

        for row_index, (number, _value, label, detail) in enumerate(rows):
            ui.numbered(number, label, detail, label_width, selected=row_index == position)
            lines += 1
        ui.hint(f"{prompt}: {_KEY_HINT}")
        lines += 1

        # A shorter paint than the last one -- a scan ended and dropped off the status -- would
        # leave the tail of the old paint on screen, so blank it and step back over it.
        for _ in range(painted - lines):
            print(ui.CLEAR_LINE)
        if painted > lines:
            ui.cursor_up(painted - lines)
        painted = lines

    print()
    with keys.raw_mode():
        while True:
            draw()
            try:
                key = keys.read_key(refresh)
            except (EOFError, KeyboardInterrupt):
                print()
                raise Cancelled from None

            if key is None:
                continue  # refresh tick: nothing pressed, come round and repaint the status
            if key == keys.UP:
                position = (position - 1) % len(rows)
            elif key == keys.DOWN:
                position = (position + 1) % len(rows)
            elif key == keys.HOME:
                position = 0
            elif key == keys.END:
                position = len(rows) - 1
            elif key == keys.ENTER:
                return rows[position][1]
            elif key in ("q", "Q", keys.ESCAPE):
                return None
            elif key.isdigit():
                target = _row_with_number(rows, int(key))
                if target is None:
                    continue
                position = target
                if len(rows) <= _DIRECT_LIMIT:
                    # Typing the number is the whole choice here, which is how the numbered menu
                    # behaved -- no Enter needed for a list this short.
                    draw()
                    return rows[position][1]


def _row_with_number(rows: Sequence[tuple[int, Any, str, str]], number: int) -> int | None:
    """Find the row a typed digit refers to, or ``None`` if no row carries that number."""
    for index, row in enumerate(rows):
        if row[0] == number:
            return index
    return None


# --------------------------------------------------------------------------------------------------
# Parameter editing
# --------------------------------------------------------------------------------------------------


@dataclass
class Param:
    """One editable field of a parameter object.

    :param attr: Attribute name on the parameter object.
    :param label: Human-readable name shown in the list.
    :param parse: Parser for a new value.
    :param format: Renders the current value for display; defaults to :class:`str`.
    :param help: One-line explanation shown beside the value.
    :param allow_blank: Whether an empty answer is meaningful (e.g. "no conditions").
    """

    attr: str
    label: str
    parse: Callable[[str], Any] = parse_text
    format: Callable[[Any], str] = str
    help: str = ""
    allow_blank: bool = False


def _param_rows(target: Any, params: Sequence[Param]) -> list[tuple[str, str]]:
    """Render each parameter as an aligned ``(row, help)`` pair.

    Values and help text are truncated to keep each row on one line: a long path or description
    wrapping across three lines makes the list unreadable, and the full value is always shown again
    when the field is edited. Truncating here rather than at print time also keeps the highlighted
    row of an arrow-driven list the same width as the rest.
    """
    label_width = max(len(p.label) for p in params)
    values = [p.format(getattr(target, p.attr)) for p in params]

    # Split the terminal between the value and help columns. The 13 covers the fixed furniture:
    # the indent, the number, the two gaps, and a column of slack so a full row never reaches the
    # right edge -- ui.numbered() truncates anything that still would.
    budget = max(ui.WIDTH - label_width - 13, 18)
    value_width = min(max(len(v) for v in values), max(budget // 2, 16))
    help_width = max(budget - value_width, 10)

    return [
        (
            f"{param.label:<{label_width}}  {ui.truncate(value, value_width):<{value_width}}",
            ui.truncate(param.help, help_width),
        )
        for param, value in zip(params, values)
    ]


def show_params(target: Any, params: Sequence[Param]) -> None:
    """Print the current value of every parameter, numbered for editing."""
    for index, (row, help_text) in enumerate(_param_rows(target, params), start=1):
        ui.numbered(index, row, help_text)


#: Outcomes of choosing a row in the parameter editor.
_CONTINUE = "continue"
_CANCEL = "cancel"
_EDIT = "edit"

#: The parameter editor's first row: the fast path is to open the list and press Enter.
_CONTINUE_LABEL = "Continue with these values"


def edit_params(
    target: Any,
    params: Sequence[Param],
    title: str = "Parameters",
    validate: Callable[[Any], None] | None = None,
) -> bool:
    """Let the user review and change a parameter object in place.

    The defaults are always usable: the cursor starts on "continue", so Enter alone accepts
    everything as shown. Choosing a parameter edits that one and returns to the list, so several can
    be changed in a row.

    :param target: The parameter object, mutated in place.
    :param params: The fields to expose, in display order.
    :param title: Heading for the parameter list.
    :param validate: Optional check run when the user accepts; raise ``ValueError`` to send them back
        to the list with the message shown.
    :returns: ``True`` if the user accepted the parameters, ``False`` if they cancelled.
    """
    position = 0  # kept across edits, so changing several fields in a row does not walk back up
    while True:
        ui.section(title)
        try:
            if keys.available():
                action, index = _choose_param_keys(target, params, position)
            else:
                action, index = _choose_param_numbers(target, params)
        except Cancelled:
            return False

        if action == _CANCEL:
            return False

        if action == _CONTINUE:
            if validate is None:
                return True
            try:
                validate(target)
                return True
            except ValueError as exc:
                ui.error(str(exc))
                continue

        _edit_one(target, params[index])
        position = index + 1


def _choose_param_numbers(target: Any, params: Sequence[Param]) -> tuple[str, int]:
    """Pick a row by typing its number, for terminals that cannot read single keypresses."""
    show_params(target, params)
    ui.hint("\nEnter a number to change that value, Enter to continue, or 'q' to cancel.")

    while True:
        raw = _read("> ")
        if raw.lower() in ("q", "quit", "cancel"):
            return _CANCEL, -1
        if not raw:
            return _CONTINUE, -1
        if raw.isdigit() and 1 <= int(raw) <= len(params):
            return _EDIT, int(raw) - 1
        ui.error(f"Enter a number from 1 to {len(params)}, or Enter to continue.")


def _choose_param_keys(target: Any, params: Sequence[Param], start: int = 0) -> tuple[str, int]:
    """Pick a row with the arrow keys, redrawn in place as the cursor moves.

    Row 0 is "continue" rather than a hidden key, and the cursor starts there on the first pass:
    opening the list and pressing Enter runs with the defaults, which is the common case.

    :param start: Row to put the cursor on, so returning from an edit lands where it left off.
    """
    rows = _param_rows(target, params)
    position = min(max(start, 0), len(rows))
    drawn = False

    def draw() -> None:
        nonlocal drawn
        if drawn:
            ui.cursor_up(len(rows) + 2)
        ui.numbered(None, _CONTINUE_LABEL, selected=position == 0)
        for index, (row, help_text) in enumerate(rows, start=1):
            ui.numbered(index, row, help_text, selected=index == position)
        ui.hint(_EDIT_HINT)
        drawn = True

    with keys.raw_mode():
        while True:
            draw()
            try:
                key = keys.read_key()
            except (EOFError, KeyboardInterrupt):
                print()
                raise Cancelled from None

            if key == keys.UP:
                position = (position - 1) % (len(rows) + 1)
            elif key == keys.DOWN:
                position = (position + 1) % (len(rows) + 1)
            elif key == keys.HOME:
                position = 0
            elif key == keys.END:
                position = len(rows)
            elif key == keys.ENTER:
                return (_CONTINUE, -1) if position == 0 else (_EDIT, position - 1)
            elif key in ("q", "Q", keys.ESCAPE):
                return _CANCEL, -1
            elif key.isdigit() and 1 <= int(key) <= len(rows):
                # Only the cursor moves: a list this long has two-digit rows, and a keypress cannot
                # tell "1" from the first half of "12".
                position = int(key)


def _edit_one(target: Any, param: Param) -> None:
    """Ask for one parameter's new value and write it back, leaving it alone if nothing changed."""
    formatted = param.format(getattr(target, param.attr))
    if param.help:
        ui.hint(param.help)

    try:
        if param.allow_blank:
            # No default: an empty answer is itself a meaningful value here (e.g. "no conditions"),
            # so it cannot double as "keep what is there".
            ui.hint(f"Currently: {formatted or '(none)'}. Enter clears it.")
            value = ask(param.label, parse=param.parse, allow_blank=True)
        else:
            value = ask(param.label, default=formatted, parse=param.parse)
    except Cancelled:
        return  # backing out of one field returns to the list, not out of the screen

    # ask() hands back the formatted default unchanged when the user just presses Enter; that is a
    # string, not a parsed value, so it must not be written onto the parameter object.
    if value == formatted and not param.allow_blank:
        return
    setattr(target, param.attr, value)
