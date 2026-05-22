"""
General-purpose dataclass-based configuration with self-healing TOML files.

Generates a TOML configuration file from a Python dataclass definition.
On each run the file is regenerated from the dataclass: user-modified values
are preserved while the structure always reflects the current field set.

Convention::

    #key = value   — default / unmodified (commented out)
    key = value    — user-modified (uncommented)
    #key =         — None value
"""

import ast
import dataclasses
import inspect
import textwrap
import tomllib
from pathlib import Path
from typing import Any, cast


class mlstr(str):
    """TOML multi-line string marker.

    Annotate a config field with this type to force the writer to render
    the value as a multi-line literal block (``'''…'''``), even when the
    default is empty or a single line. Useful for fields where users will
    paste multi-line content (PEM keys, certificates, SSH keys, etc.) —
    the commented default already shows the paste markers, so users do
    not have to know TOML's multi-line literal syntax to fill it in.
    """


def format_value(value: str | int | float | bool) -> str:
    """
    Format a Python value as a TOML value string.

    Handles the four TOML-native types: ``str``, ``int``, ``float``, ``bool``.
    ``mlstr`` values, and any plain string containing ``\\n``, are emitted
    as TOML multi-line literal blocks (``'''…'''``) so PEM keys,
    certificates and similar payloads stay readable in the generated
    config.  Strings that contain ``'''`` fall back to a single-line
    escaped form.
    This function is intentionally public so that other modules (e.g.
    ``core.script``) can reuse it for consistent TOML formatting.

    :param value: The value to format.
    :return: TOML-formatted string representation.
    """
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return str(value)
    if isinstance(value, mlstr) and "'''" not in value:
        return f"'''\n{value}'''"
    if isinstance(value, str):
        if '\n' in value and "'''" not in value:
            return f"'''\n{value}'''"
        escaped = (
            value
            .replace('\\', '\\\\')
            .replace('"', '\\"')
            .replace('\n', '\\n')
            .replace('\r', '\\r')
        )
        return f'"{escaped}"'
    return str(value)


def _format_for_toml(value: Any) -> str:
    """Format any supported value as a TOML literal.

    Wraps :func:`format_value` for the four scalar types and adds TOML
    inline-table / array support so ``dict`` and ``list`` fields (e.g.
    ``LiveProviderConfig.symbol_map``) round-trip through the
    self-healing config writer.
    """
    if isinstance(value, dict):
        if not value:
            return "{}"
        parts = [
            f"{format_value(str(k))} = {_format_for_toml(v)}"
            for k, v in value.items()
        ]
        return "{ " + ", ".join(parts) + " }"
    if isinstance(value, list):
        if not value:
            return "[]"
        return "[" + ", ".join(_format_for_toml(v) for v in value) + "]"
    return format_value(value)


def _is_mlstr_field(f: dataclasses.Field) -> bool:
    """Check if a dataclass field is annotated as ``mlstr``.

    Relies on ``f.type`` being a real type object (not a forward-ref
    string) — guaranteed in this project because ``from __future__ import
    annotations`` is intentionally not used.
    """
    t = f.type
    return isinstance(t, type) and issubclass(t, mlstr)


def _emit_assignment(
    lines: list[str],
    name: str,
    formatted: str,
    *,
    commented: bool,
) -> None:
    """Append a ``name = value`` assignment to ``lines``.

    For multi-line formatted values (e.g. ``'''…'''``), every continuation
    line is prefixed with ``#`` when ``commented`` is true so the whole
    block is one TOML comment, not a half-commented stray.
    """
    parts = formatted.split('\n')
    prefix = '#' if commented else ''
    lines.append(f"{prefix}{name} = {parts[0]}")
    for cont in parts[1:]:
        lines.append(f"{prefix}{cont}")


def extract_field_docs(config_cls: type) -> dict[str, str]:
    """
    Extract attribute docstrings from a dataclass source via AST parsing.

    Looks for ``Expr(Constant(str))`` nodes immediately following
    ``AnnAssign`` nodes in the class body (PEP 257 attribute docstrings).
    Walks the MRO in reverse order so docstrings on inherited fields are
    picked up too — a subclass-level docstring overrides the inherited
    one for the same field name.

    :param config_cls: The dataclass type to inspect.
    :return: Mapping of field name to its docstring.
    """
    docs: dict[str, str] = {}
    for cls in reversed(config_cls.__mro__):
        if cls is object:
            continue
        try:
            source = textwrap.dedent(inspect.getsource(cls))
        except (OSError, TypeError):
            continue

        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue

        class_def = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == cls.__name__:
                class_def = node
                break

        if class_def is None:
            continue

        body = class_def.body
        for i, node in enumerate(body):
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                field_name = node.target.id
                if i + 1 < len(body):
                    next_node = body[i + 1]
                    if (
                        isinstance(next_node, ast.Expr)
                        and isinstance(next_node.value, ast.Constant)
                        and isinstance(next_node.value.value, str)
                    ):
                        docs[field_name] = next_node.value.value

    return docs


def generate_toml(
    config_cls: type,
    user_values: dict | None = None,
) -> str:
    """
    Generate a TOML string from a dataclass definition.

    Fields with user-modified values are written uncommented.  Fields at their
    default value are written as comments (``#key = value``).

    :param config_cls: The config dataclass type.
    :param user_values: User-modified values to write uncommented.
    :return: Generated TOML content string.
    """
    field_docs = extract_field_docs(config_cls)
    lines: list[str] = []

    class_doc = config_cls.__doc__
    if class_doc:
        for doc_line in class_doc.strip().splitlines():
            stripped = doc_line.strip()
            lines.append(f"# {stripped}" if stripped else "#")

    for f in dataclasses.fields(cast(Any, config_cls)):
        name = f.name
        default = f.default
        is_ml = _is_mlstr_field(f)

        # Resolve ``field(default_factory=...)`` to an actual default value
        # so the commented placeholder shows real TOML syntax (e.g.
        # ``#symbol_map = {}``) instead of a bare ``#symbol_map =``.
        if default is dataclasses.MISSING and f.default_factory is not dataclasses.MISSING:
            try:
                default = f.default_factory()
            except Exception:  # noqa: BLE001
                default = dataclasses.MISSING

        lines.append("")

        if name in field_docs:
            for doc_line in field_docs[name].strip().splitlines():
                lines.append(f"# {doc_line.strip()}")

        if user_values and name in user_values:
            value = user_values[name]
            if is_ml and not isinstance(value, mlstr):
                value = mlstr(value)
            _emit_assignment(lines, name, _format_for_toml(value), commented=False)
        elif default is dataclasses.MISSING or default is None:
            if is_ml:
                _emit_assignment(lines, name, "'''\n'''", commented=True)
            else:
                lines.append(f"#{name} =")
        else:
            value = default
            if is_ml and not isinstance(value, mlstr):
                value = mlstr(value)
            _emit_assignment(lines, name, _format_for_toml(value), commented=True)

    return '\n'.join(lines) + '\n'


def parse_toml_with_comments(toml_content: str) -> dict:
    """
    Parse TOML content, returning only uncommented (user-modified) values.

    Commented lines (``#key = value``) are standard TOML comments and are
    excluded by the parser.  Only actively set values are returned.

    :param toml_content: Raw TOML file content.
    :return: Dict of parsed key-value pairs.
    """
    return tomllib.loads(toml_content)


def ensure_config(config_cls: type, config_path: Path) -> object:
    """
    Main entry point.  Call on every application run.

    1. If the file does not exist, generate it with all defaults (commented).
    2. If it exists, read user values, regenerate from the dataclass, write back.
    3. Return a populated dataclass instance with user values over defaults.

    The result is cached on ``config_cls._ensured``, so repeated calls
    return the same instance without file I/O.

    TOML table sections (e.g. ``[binance]``) not managed by the dataclass
    are preserved verbatim at the end of the file.

    :param config_cls: The config dataclass type (not an instance).
    :param config_path: Path to the TOML file.
    :return: A populated config dataclass instance.
    """
    if hasattr(config_cls, '_ensured'):
        return config_cls._ensured

    user_values = None
    extra_content = ""

    if config_path.exists():
        user_values, extra_content = _parse_existing(config_path, config_cls)

    toml_content = generate_toml(config_cls, user_values)

    if extra_content:
        toml_content += '\n' + extra_content
        if not extra_content.endswith('\n'):
            toml_content += '\n'

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(toml_content, encoding='utf-8')

    instance = _create_instance(config_cls, user_values)
    config_cls._ensured = instance
    return instance


def _parse_existing(config_path: Path, config_cls: type) -> tuple[dict, str]:
    """
    Parse an existing config file to extract user values and extra sections.

    :param config_path: Path to the TOML file.
    :param config_cls: The config dataclass type.
    :return: ``(user_values, extra_sections_raw_text)``.
    """
    content = config_path.read_text(encoding='utf-8')

    parsed = tomllib.loads(content)

    field_names = {f.name for f in dataclasses.fields(cast(Any, config_cls))}

    # Field-typed values are kept; non-field keys (typically TOML section
    # headers like ``[binance]``) are left out and preserved verbatim via
    # :func:`_extract_extra_sections`. Dict-typed user values (e.g.
    # ``symbol_map = {...}``) are accepted only when the key matches a
    # known dataclass field.
    user_values: dict = {}
    for key, value in parsed.items():
        if key in field_names:
            user_values[key] = value

    extra_content = _extract_extra_sections(content, field_names)

    return user_values, extra_content


def _extract_extra_sections(content: str, field_names: set[str] | None = None) -> str:
    """
    Extract raw text of TOML table sections from file content.

    Walks the file from top to bottom and keeps every ``[section]`` block
    whose header does not match a known dataclass field. Field-matching
    table sections (e.g. ``[symbol_map]``) are filtered out so the
    self-healing writer does not re-emit them alongside the inline-table
    form already produced by :func:`generate_toml`.

    :param content: Raw file content.
    :param field_names: Known dataclass field names whose sections should
                        be dropped from the extras. ``None`` keeps every
                        section (legacy behaviour).
    :return: Raw text of extra sections, or empty string.
    """
    lines = content.splitlines()
    out: list[str] = []
    skipping = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('[') and not stripped.startswith('#'):
            # New section header — decide whether to keep or drop.
            header = stripped.strip('[]').split('.', 1)[0]
            skipping = field_names is not None and header in field_names
            if not skipping:
                out.append(line)
            continue
        if not skipping:
            # Only start capturing once we've seen the first kept header.
            if out:
                out.append(line)
    return '\n'.join(out)


def _create_instance(config_cls: type, user_values: dict | None):
    """
    Create a dataclass instance with user values merged over defaults.

    Handles ``int`` to ``float`` coercion when the field default is a
    float, and ``str`` to ``mlstr`` coercion when the field is annotated
    as ``mlstr`` (tomllib returns plain ``str`` regardless).

    :param config_cls: The config dataclass type.
    :param user_values: User-modified values, or ``None``.
    :return: A populated config dataclass instance.
    """
    if not user_values:
        return config_cls()

    kwargs: dict = {}
    for f in dataclasses.fields(cast(Any, config_cls)):
        if f.name not in user_values:
            continue
        value = user_values[f.name]
        if (
            f.default is not dataclasses.MISSING
            and isinstance(f.default, float)
            and isinstance(value, int)
            and not isinstance(value, bool)
        ):
            value = float(value)
        if _is_mlstr_field(f) and isinstance(value, str) and not isinstance(value, mlstr):
            value = mlstr(value)
        kwargs[f.name] = value

    return config_cls(**kwargs)
