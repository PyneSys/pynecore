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


def format_value(value: str | int | float | bool) -> str:
    """
    Format a Python value as a TOML value string.

    Handles the four TOML-native types: ``str``, ``int``, ``float``, ``bool``.
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
    if isinstance(value, str):
        escaped = (
            value
            .replace('\\', '\\\\')
            .replace('"', '\\"')
            .replace('\n', '\\n')
            .replace('\r', '\\r')
        )
        return f'"{escaped}"'
    return str(value)


def extract_field_docs(config_cls: type) -> dict[str, str]:
    """
    Extract attribute docstrings from a dataclass source via AST parsing.

    Looks for ``Expr(Constant(str))`` nodes immediately following
    ``AnnAssign`` nodes in the class body (PEP 257 attribute docstrings).

    :param config_cls: The dataclass type to inspect.
    :return: Mapping of field name to its docstring.
    """
    try:
        source = textwrap.dedent(inspect.getsource(config_cls))
    except (OSError, TypeError):
        return {}

    tree = ast.parse(source)

    class_def = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == config_cls.__name__:
            class_def = node
            break

    if class_def is None:
        return {}

    docs: dict[str, str] = {}
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

    for f in dataclasses.fields(config_cls):
        name = f.name
        default = f.default

        lines.append("")

        if name in field_docs:
            for doc_line in field_docs[name].strip().splitlines():
                lines.append(f"# {doc_line.strip()}")

        if user_values and name in user_values:
            lines.append(f"{name} = {format_value(user_values[name])}")
        elif default is dataclasses.MISSING or default is None:
            lines.append(f"#{name} =")
        else:
            lines.append(f"#{name} = {format_value(default)}")

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

    TOML table sections (e.g. ``[binance]``) not managed by the dataclass
    are preserved verbatim at the end of the file.

    :param config_cls: The config dataclass type (not an instance).
    :param config_path: Path to the TOML file.
    :return: A populated config dataclass instance.
    """
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

    return _create_instance(config_cls, user_values)


def _parse_existing(config_path: Path, config_cls: type) -> tuple[dict, str]:
    """
    Parse an existing config file to extract user values and extra sections.

    :param config_path: Path to the TOML file.
    :param config_cls: The config dataclass type.
    :return: ``(user_values, extra_sections_raw_text)``.
    """
    content = config_path.read_text(encoding='utf-8')

    parsed = tomllib.loads(content)

    field_names = {f.name for f in dataclasses.fields(config_cls)}

    user_values: dict = {}
    for key, value in parsed.items():
        if key in field_names and not isinstance(value, dict):
            user_values[key] = value

    extra_content = _extract_extra_sections(content)

    return user_values, extra_content


def _extract_extra_sections(content: str) -> str:
    """
    Extract raw text of TOML table sections from file content.

    Everything from the first ``[section]`` header to end of file is returned.

    :param content: Raw file content.
    :return: Raw text of extra sections, or empty string.
    """
    lines = content.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('[') and not stripped.startswith('#'):
            return '\n'.join(lines[i:])
    return ""


def _create_instance(config_cls: type, user_values: dict | None):
    """
    Create a dataclass instance with user values merged over defaults.

    Handles ``int`` to ``float`` coercion when the field default is a float.

    :param config_cls: The config dataclass type.
    :param user_values: User-modified values, or ``None``.
    :return: A populated config dataclass instance.
    """
    if not user_values:
        return config_cls()

    kwargs: dict = {}
    for f in dataclasses.fields(config_cls):
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
        kwargs[f.name] = value

    return config_cls(**kwargs)
