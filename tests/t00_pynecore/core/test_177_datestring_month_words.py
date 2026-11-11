"""
Month words of ``core.datetime.parse_datestring`` (``timestamp(dateString)``).

Verified on TradingView (CAPITALCOM:EURUSD 60): a word is a month when its first
three characters spell an English month abbreviation in any letter case, the rest
of the word is ignored, and of several month words the last one wins. A word runs
over every character from "A" up, so "[", "_", "~" and non-ASCII letters glue to
it, while a "." or "," after it is skipped.

``timestamp("1 Janv 2017 00:00:00")`` as an ``input.time`` default failed the
module load with "Invalid date format" (GitHub issue #111).
"""
import pytest

from pynecore.core.datetime import parse_datestring


def _millis(datestring: str) -> int:
    return round(parse_datestring(datestring).timestamp() * 1000)


def __test_month_word_prefix__():
    """Any word starting with a month abbreviation is that month"""
    expected = {
        "1 Janv 2017 00:00:00": 1483228800000,
        "1 Janvier 2017 00:00:00": 1483228800000,
        "1 Janu 2017 00:00:00": 1483228800000,
        "1 Janxyz 2017 00:00:00": 1483228800000,
        "1 JANV 2017 00:00:00": 1483228800000,
        "1 jAnV 2017 00:00:00": 1483228800000,
        "1 Jan_x 2017 00:00:00": 1483228800000,
        "1 Jan[x 2017": 1483228800000,
        "1 Jan~ 2017": 1483228800000,
        "1 Jan. 2017 00:00:00": 1483228800000,
        "1 Jan.. 2017": 1483228800000,
        "1 Janvier. 2017": 1483228800000,
        "Janv 1 2017 00:00:00": 1483228800000,
        "1 Janv 2017": 1483228800000,
        "1 Janv 2017 00:00": 1483228800000,
        "Janv 2017": 1483228800000,
        "1 Februar 2017 00:00:00": 1485907200000,
        "1 Mars 2017 00:00:00": 1488326400000,
        "1 Marzo 2017 00:00:00": 1488326400000,
        "1 Mayo 2017 00:00:00": 1493596800000,
        "1 Juni 2017 00:00:00": 1496275200000,
        "1 Junk 2017 00:00:00": 1496275200000,
        "1 Augu 2017 00:00:00": 1501545600000,
        "1 Sept 2017 00:00:00": 1504224000000,
        "1 Sept, 2017 00:00:00": 1504224000000,
        "1 Septem 2017 00:00:00": 1504224000000,
        "1 Septembre 2017 00:00:00": 1504224000000,
        "1 Septiembre 2017 00:00:00": 1504224000000,
        "Septembre 1 2017 00:00:00": 1504224000000,
        "Septembre 2017": 1504224000000,
        "1 Octobre 2017 00:00:00": 1506816000000,
        "1 Decimal 2017 00:00:00": 1512086400000,
        "1 Decé 2017 00:00:00": 1512086400000,
    }
    for datestring, millis in expected.items():
        assert _millis(datestring) == millis, datestring


def __test_month_word_with_zone__():
    """A month word keeps the zone handling of the rest of the string"""
    assert _millis("1 Janv 2017 00:00:00 GMT+0200") == 1483221600000
    assert _millis("1 Janv 2017 00:00:00 +0200") == 1483221600000
    assert _millis("01 Janvier 2017 00:00 EST") == 1483246800000


def __test_last_month_word_wins__():
    """Of several month words the last one is the month"""
    assert _millis("1 Jan Feb 2017") == 1485907200000
    assert _millis("Feb 1 Jan 2017") == 1483228800000
    assert _millis("Foo Mars 1 2017") == 1488326400000
    assert _millis("Junk 01 Jan 2022 00:00") == 1640995200000
    assert _millis("Decor 01 Feb 2022 00:00") == 1643673600000


def __test_non_english_month_words_are_rejected__():
    """A word whose first three characters are no English month is not a month"""
    for word in ("Févr", "Fevr", "Avr", "Mai", "Juin", "Juil", "Déc", "Mrz", "Dez", "Okt",
                 "Mei", "Xyz", "Ja", "Fe", "J", "Jan-x"):
        with pytest.raises(ValueError):
            parse_datestring(f"1 {word} 2017 00:00:00")
    with pytest.raises(ValueError):
        parse_datestring("Février 1 2017")
