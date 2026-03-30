"""Tests for the general-purpose dataclass config system."""

from dataclasses import dataclass
from pathlib import Path

import pytest

from pynecore.core.config import (
    ensure_config,
    extract_field_docs,
    format_value,
    generate_toml,
    parse_toml_with_comments,
)


@dataclass
class SampleConfig:
    """Sample configuration"""

    api_key: str = ""
    """API key for the service"""

    timeout: int = 30
    """Request timeout in seconds"""

    rate: float = 1.5
    """Rate multiplier"""

    enabled: bool = False
    """Enable the feature"""


@dataclass
class MinimalConfig:
    """Minimal"""

    name: str = "default"


def __test_format_value_str__():
    assert format_value("hello") == '"hello"'
    assert format_value("") == '""'
    assert format_value('has "quotes"') == '"has \\"quotes\\""'
    assert format_value("line\nbreak") == '"line\\nbreak"'
    assert format_value("back\\slash") == '"back\\\\slash"'


def __test_format_value_int__():
    assert format_value(0) == "0"
    assert format_value(42) == "42"
    assert format_value(-7) == "-7"


def __test_format_value_float__():
    assert format_value(3.14) == "3.14"
    assert format_value(0.0) == "0.0"


def __test_format_value_bool__():
    assert format_value(True) == "true"
    assert format_value(False) == "false"


def __test_extract_field_docs__():
    docs = extract_field_docs(SampleConfig)
    assert docs["api_key"] == "API key for the service"
    assert docs["timeout"] == "Request timeout in seconds"
    assert docs["rate"] == "Rate multiplier"
    assert docs["enabled"] == "Enable the feature"


def __test_extract_field_docs_missing__():
    @dataclass
    class NoDocs:
        x: int = 0
        y: str = ""

    docs = extract_field_docs(NoDocs)
    assert docs == {}


def __test_first_run_all_defaults__(tmp_path: Path):
    """First run: generates TOML with all values commented out."""
    cfg_path = tmp_path / "config.toml"
    result = ensure_config(SampleConfig, cfg_path)

    content = cfg_path.read_text()
    assert "# Sample configuration" in content
    assert '#api_key = ""' in content
    assert "#timeout = 30" in content
    assert "#rate = 1.5" in content
    assert "#enabled = false" in content

    for line in content.splitlines():
        stripped = line.strip()
        if "=" in stripped and not stripped.startswith("#"):
            pytest.fail(f"Unexpected uncommented line: {stripped}")

    assert result.api_key == ""
    assert result.timeout == 30
    assert result.rate == 1.5
    assert result.enabled is False


def __test_user_values_preserved__(tmp_path: Path):
    """User-modified values survive regeneration."""
    cfg_path = tmp_path / "config.toml"

    ensure_config(SampleConfig, cfg_path)

    content = cfg_path.read_text()
    content = content.replace('#api_key = ""', 'api_key = "my_key"')
    content = content.replace("#enabled = false", "enabled = true")
    cfg_path.write_text(content)

    result = ensure_config(SampleConfig, cfg_path)

    content = cfg_path.read_text()
    assert 'api_key = "my_key"' in content
    assert "enabled = true" in content
    assert "#timeout = 30" in content
    assert "#rate = 1.5" in content

    assert result.api_key == "my_key"
    assert result.enabled is True
    assert result.timeout == 30
    assert result.rate == 1.5


def __test_new_field_appears__(tmp_path: Path):
    """A new field in the dataclass appears commented in existing TOML."""
    cfg_path = tmp_path / "config.toml"

    ensure_config(MinimalConfig, cfg_path)

    content = cfg_path.read_text()
    assert '#name = "default"' in content

    cfg_path.write_text('name = "custom"\n')

    result = ensure_config(SampleConfig, cfg_path)

    content = cfg_path.read_text()
    assert "#timeout = 30" in content
    assert "#rate = 1.5" in content
    assert "#enabled = false" in content


def __test_removed_field_disappears__(tmp_path: Path):
    """A field removed from the dataclass disappears from TOML."""
    cfg_path = tmp_path / "config.toml"

    ensure_config(SampleConfig, cfg_path)

    content = cfg_path.read_text()
    content = content.replace("#timeout = 30", "timeout = 60")
    cfg_path.write_text(content)

    result = ensure_config(MinimalConfig, cfg_path)

    content = cfg_path.read_text()
    assert "timeout" not in content
    assert "api_key" not in content
    assert "rate" not in content
    assert "enabled" not in content
    assert '#name = "default"' in content


def __test_user_resets_to_default__(tmp_path: Path):
    """When user comments a field back, it stays commented."""
    cfg_path = tmp_path / "config.toml"

    ensure_config(SampleConfig, cfg_path)

    result = ensure_config(SampleConfig, cfg_path)

    content = cfg_path.read_text()
    assert "#timeout = 30" in content
    assert result.timeout == 30


def __test_type_validation__(tmp_path: Path):
    """All four types are correctly formatted and parsed."""
    cfg_path = tmp_path / "config.toml"

    cfg_path.write_text(
        'api_key = "test"\n'
        'timeout = 99\n'
        'rate = 2.718\n'
        'enabled = true\n'
    )

    result = ensure_config(SampleConfig, cfg_path)

    assert result.api_key == "test"
    assert result.timeout == 99
    assert result.rate == 2.718
    assert result.enabled is True

    content = cfg_path.read_text()
    assert 'api_key = "test"' in content
    assert "timeout = 99" in content
    assert "rate = 2.718" in content
    assert "enabled = true" in content


def __test_class_docstring_as_header__(tmp_path: Path):
    """Class docstring becomes the header comment in TOML."""
    cfg_path = tmp_path / "config.toml"
    ensure_config(SampleConfig, cfg_path)

    content = cfg_path.read_text()
    lines = content.splitlines()
    assert lines[0] == "# Sample configuration"


def __test_extra_sections_preserved__(tmp_path: Path):
    """TOML table sections not in the dataclass are preserved."""
    cfg_path = tmp_path / "config.toml"

    ensure_config(SampleConfig, cfg_path)

    content = cfg_path.read_text()
    content += '\n[binance]\napiKey = "binance_key"\nsecret = "binance_secret"\n'
    cfg_path.write_text(content)

    result = ensure_config(SampleConfig, cfg_path)

    content = cfg_path.read_text()
    assert "[binance]" in content
    assert 'apiKey = "binance_key"' in content
    assert 'secret = "binance_secret"' in content


def __test_ensure_config_returns_correct_instance__(tmp_path: Path):
    """ensure_config returns a properly typed dataclass instance."""
    cfg_path = tmp_path / "config.toml"

    cfg_path.write_text('api_key = "real_key"\ntimeout = 60\n')

    result = ensure_config(SampleConfig, cfg_path)

    assert isinstance(result, SampleConfig)
    assert result.api_key == "real_key"
    assert result.timeout == 60
    assert result.rate == 1.5
    assert result.enabled is False


def __test_generate_toml_standalone__():
    """generate_toml produces correct output without file I/O."""
    toml = generate_toml(SampleConfig)
    assert "# Sample configuration" in toml
    assert '#api_key = ""' in toml
    assert "#timeout = 30" in toml

    toml_with_user = generate_toml(SampleConfig, {"api_key": "key1", "timeout": 99})
    assert 'api_key = "key1"' in toml_with_user
    assert "timeout = 99" in toml_with_user
    assert "#rate = 1.5" in toml_with_user


def __test_parse_toml_with_comments__():
    """parse_toml_with_comments returns only uncommented values."""
    content = '# header\n#commented = 1\nactive = 2\n'
    result = parse_toml_with_comments(content)
    assert result == {"active": 2}
    assert "commented" not in result


def __test_int_to_float_coercion__(tmp_path: Path):
    """Integer TOML values are coerced to float when the field default is float."""
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text("rate = 3\n")

    result = ensure_config(SampleConfig, cfg_path)
    assert result.rate == 3.0
    assert isinstance(result.rate, float)


def __test_creates_parent_directories__(tmp_path: Path):
    """ensure_config creates parent directories if they don't exist."""
    cfg_path = tmp_path / "sub" / "dir" / "config.toml"
    result = ensure_config(SampleConfig, cfg_path)
    assert cfg_path.exists()
    assert isinstance(result, SampleConfig)


def __test_none_default_field__(tmp_path: Path):
    """Fields with None default are written as #key =."""

    @dataclass
    class WithNone:
        """Config with optional field"""

        name: str = "test"
        """The name"""

        tag: str | None = None
        """Optional tag"""

    cfg_path = tmp_path / "config.toml"
    ensure_config(WithNone, cfg_path)

    content = cfg_path.read_text()
    assert '#name = "test"' in content
    assert "#tag =" in content
    assert "#tag = None" not in content


def __test_idempotent_regeneration__(tmp_path: Path):
    """Running ensure_config twice with no changes produces identical files."""
    cfg_path = tmp_path / "config.toml"

    ensure_config(SampleConfig, cfg_path)
    content1 = cfg_path.read_text()

    ensure_config(SampleConfig, cfg_path)
    content2 = cfg_path.read_text()

    assert content1 == content2
