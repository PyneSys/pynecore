from pynecore.cli.commands import data
from pynecore.cli.utils.keyreader import Key
from pynecore.cli.utils.provider_picker import ProviderChoice, ProviderPicker
from pynecore.core.plugin import ProviderPlugin


class _Provider(ProviderPlugin):
    plugin_name = "Readable Provider"


class _EntryPoint:
    def __init__(self, plugin_class):
        self.plugin_class = plugin_class

    def load(self):
        return self.plugin_class


def __test_provider_picker_selects_highlighted_provider__():
    picker = ProviderPicker([
        ProviderChoice("bybit", "Bybit"),
        ProviderChoice("ccxt", "CCXT"),
    ])

    assert picker._handle_key(Key.DOWN)
    assert not picker._handle_key(Key.ENTER)
    assert picker.selected == "ccxt"


def __test_provider_picker_filters_entry_point_and_display_name__():
    picker = ProviderPicker([
        ProviderChoice("capitalcom", "Capital.com"),
        ProviderChoice("ccxt", "CCXT"),
    ])

    picker.filter_text = "capital."
    picker._apply_filter()
    assert [provider.id for provider in picker.filtered] == ["capitalcom"]

    picker.filter_text = "ccxt"
    picker._apply_filter()
    assert [provider.id for provider in picker.filtered] == ["ccxt"]


def __test_choose_provider_lists_only_provider_plugins_sorted__(monkeypatch):
    seen = []

    monkeypatch.setattr(data, "discover_plugins", lambda: {
        "z-provider": _EntryPoint(_Provider),
        "not-a-provider": _EntryPoint(object),
        "a-provider": _EntryPoint(_Provider),
    })

    def select_first(picker):
        seen.extend(picker.brokers)
        return picker.brokers[0].id

    monkeypatch.setattr(ProviderPicker, "run", select_first)

    assert data._choose_provider() == "a-provider"
    assert [(provider.id, provider.name) for provider in seen] == [
        ("a-provider", "Readable Provider"),
        ("z-provider", "Readable Provider"),
    ]
