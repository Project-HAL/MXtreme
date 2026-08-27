"""Terminal input helpers: menus, typed prompts and the parameter editor.

Three things every screen in the CLI needs:

- :func:`menu` -- pick one of a numbered list of options, or go back.
- :func:`ask` and the ``parse_*`` functions -- read one value, re-prompting until it parses.
- :func:`edit_params` -- show a dataclass of parameters with their current values and let the user
  change any of them, or press Enter to run with the defaults.

Every prompt treats Ctrl-C and end-of-input as "back out of this step" by raising :class:`Cancelled`,
which the calling menu catches; the top-level menu treats it as "quit".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from mxtreme_cli import ui

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


def menu(
    options: Sequence[tuple[Any, str] | tuple[Any, str, str]],
    prompt: str = "Select an option",
    back_label: str = "Back",
) -> Any | None:
    """Show a numbered menu and return the chosen option's value.

    :param options: ``(value, label)`` or ``(value, label, detail)`` tuples, listed in order from 1.
    :param prompt: Question shown under the list.
    :param back_label: Label for option ``0``, which returns ``None``.
    :returns: The chosen value, or ``None`` if the user chose to go back.
    :raises Cancelled: If the user backs out with Ctrl-C.
    """
    label_width = max(len(option[1]) for option in options)
    while True:
        print()
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


def show_params(target: Any, params: Sequence[Param]) -> None:
    """Print the current value of every parameter, numbered for editing.

    Values and help text are truncated to keep each row on one line: a long path or description
    wrapping across three lines makes the list unreadable, and the full value is always shown again
    when the field is edited.
    """
    label_width = max(len(p.label) for p in params)
    values = [p.format(getattr(target, p.attr)) for p in params]

    # Split the terminal between the value and help columns, leaving room for the number and gaps.
    budget = max(ui.WIDTH - label_width - 12, 30)
    value_width = min(max(len(v) for v in values), max(budget // 2, 20))
    help_width = max(budget - value_width, 12)

    for index, (param, value) in enumerate(zip(params, values), start=1):
        row = f"{param.label:<{label_width}}  {ui.truncate(value, value_width):<{value_width}}"
        ui.numbered(index, row, ui.truncate(param.help, help_width))


def edit_params(
    target: Any,
    params: Sequence[Param],
    title: str = "Parameters",
    validate: Callable[[Any], None] | None = None,
) -> bool:
    """Let the user review and change a parameter object in place.

    The defaults are always usable: Enter accepts everything as shown and moves on. Entering a
    number edits that one parameter and returns to the list, so several can be changed in a row.

    :param target: The parameter object, mutated in place.
    :param params: The fields to expose, in display order.
    :param title: Heading for the parameter list.
    :param validate: Optional check run when the user accepts; raise ``ValueError`` to send them back
        to the list with the message shown.
    :returns: ``True`` if the user accepted the parameters, ``False`` if they cancelled.
    """
    while True:
        ui.section(title)
        show_params(target, params)
        ui.hint("\nEnter a number to change that value, Enter to continue, or 'q' to cancel.")

        try:
            raw = _read("> ")
        except Cancelled:
            return False

        if raw.lower() in ("q", "quit", "cancel"):
            return False

        if not raw:
            if validate is None:
                return True
            try:
                validate(target)
                return True
            except ValueError as exc:
                ui.error(str(exc))
                continue

        if not (raw.isdigit() and 1 <= int(raw) <= len(params)):
            ui.error(f"Enter a number from 1 to {len(params)}, or Enter to continue.")
            continue

        param = params[int(raw) - 1]
        current = getattr(target, param.attr)
        formatted = param.format(current)
        if param.help:
            ui.hint(param.help)
        try:
            if param.allow_blank:
                # No default: an empty answer is itself a meaningful value here (e.g. "no
                # conditions"), so it cannot double as "keep what is there".
                ui.hint(f"Currently: {formatted or '(none)'}. Enter clears it.")
                value = ask(param.label, parse=param.parse, allow_blank=True)
            else:
                value = ask(param.label, default=formatted, parse=param.parse)
        except Cancelled:
            continue  # backing out of one field returns to the list, not out of the screen

        # ask() hands back the formatted default unchanged when the user just presses Enter; that is
        # a string, not a parsed value, so it must not be written onto the parameter object.
        if value == formatted and not param.allow_blank:
            continue
        setattr(target, param.attr, value)
