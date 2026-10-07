"""Revisioned metadata is shared safely by independent visualization consumers."""
import json

import pytest

from pynecore import lib
from pynecore.core import viz
from pynecore.types.color import Color
from pynecore.types.na import NA
from pynecore.types.plot_meta import PlotMeta, PlotMetaRegistry


def __test_consumers_share_serialization_without_consuming_changes__():
    registry = PlotMetaRegistry()
    registry['p'] = PlotMeta(id='p', kind='plot', color=Color('FF0000'))
    calls = 0

    def serialize(meta):
        nonlocal calls
        calls += 1
        return viz.serialize_meta(meta)

    first, records = registry.collect_changes(0, serialize)
    second, other = registry.collect_changes(0, serialize)
    assert first == second and records == other
    assert calls == 1
    records[0]['title'] = 'reader-owned'
    assert registry.collect_changes(0, serialize)[1] == other
    for _ in range(5):
        assert registry.collect_changes(first, serialize) == (first, [])
    assert calls == 1

    registry['p'].dynamic = True
    first, records = registry.collect_changes(first, serialize)
    second, other = registry.collect_changes(second, serialize)
    assert first == second and records == other
    assert records[0]['dynamic'] is True and calls == 2


@pytest.mark.parametrize('mutate', ['value', 'a', 't'])
def __test_in_place_color_changes_reach_all_referencing_metas__(mutate):
    registry = PlotMetaRegistry()
    color = Color('FF0000')
    registry['p'] = PlotMeta(id='p', kind='plot', color=color)
    registry['s'] = PlotMeta(id='s', kind='shape', color=color, textcolor=color)
    cursor, _ = registry.collect_changes(0, viz.serialize_meta)
    assert len(registry._colors) == 1
    setattr(color, mutate, {'value': 0x00FF00FF, 'a': 90, 't': 50}[mutate])
    changed, records = registry.collect_changes(cursor, viz.serialize_meta)
    assert changed > cursor
    assert [record['id'] for record in records] == ['p', 's']
    assert records[0]['color'] == records[1]['color'] == viz.color_str(color)
    assert records[1]['textcolor'] == viz.color_str(color)
    assert registry.collect_changes(cursor, viz.serialize_meta)[1] == records
    assert registry.collect_changes(changed, viz.serialize_meta) == (changed, [])


def __test_equal_color_replacement_tracks_the_new_object_and_ignores_old_mutations__():
    registry = PlotMetaRegistry()
    old = Color('FF0000')
    meta = PlotMeta(id='p', kind='plot', color=old)
    registry['p'] = meta
    cursor, _ = registry.collect_changes(0, viz.serialize_meta)
    replacement = Color('FF0000')
    meta.color = replacement
    cursor, records = registry.collect_changes(cursor, viz.serialize_meta)
    assert not records
    old.a = 20
    assert registry.collect_changes(cursor, viz.serialize_meta) == (cursor, [])
    replacement.a = 40
    cursor, records = registry.collect_changes(cursor, viz.serialize_meta)
    assert records[0]['color'] == viz.color_str(replacement)
    meta.color = NA(Color)
    cursor, records = registry.collect_changes(cursor, viz.serialize_meta)
    assert 'color' not in records[0]
    replacement.a = 60
    assert registry.collect_changes(cursor, viz.serialize_meta) == (cursor, [])


def __test_scalar_changes_and_irrelevant_fields_keep_wire_upsert_semantics__():
    registry = PlotMetaRegistry()
    meta = PlotMeta(id='p', kind='plot', color=Color('FF0000'))
    registry['p'] = meta
    cursor, _ = registry.collect_changes(0, viz.serialize_meta)
    meta.linewidth = 3
    cursor, records = registry.collect_changes(cursor, viz.serialize_meta)
    assert records[0]['linewidth'] == 3
    meta.linewidth = 3
    assert registry.collect_changes(cursor, viz.serialize_meta) == (cursor, [])
    meta.wickcolor = Color('0000FF')
    cursor, records = registry.collect_changes(cursor, viz.serialize_meta)
    assert records == []


def __test_reset_and_removal_detach_previous_run_objects__():
    registry = PlotMetaRegistry()
    old_color = Color('FF0000')
    old_meta = PlotMeta(id='p', kind='plot', color=old_color)
    registry['p'] = old_meta
    cursor, _ = registry.collect_changes(0, viz.serialize_meta)
    registry.clear()
    after_reset, records = registry.collect_changes(cursor, viz.serialize_meta)
    assert after_reset > cursor and records == []
    old_meta.linewidth = 4
    old_color.a = 50
    assert registry.collect_changes(after_reset, viz.serialize_meta) == (after_reset, [])
    registry['p'] = PlotMeta(id='p', kind='plot', color=Color('FF0000'))
    cursor, records = registry.collect_changes(after_reset, viz.serialize_meta)
    assert records[0]['id'] == 'p'
    removed = registry.pop('p')
    cursor, _ = registry.collect_changes(cursor, viz.serialize_meta)
    removed.linewidth = 6
    assert registry.collect_changes(cursor, viz.serialize_meta) == (cursor, [])
    assert not registry._colors


def __test_dict_mutations_register_and_detach_metadata__():
    registry = PlotMetaRegistry()
    p = PlotMeta(id='p', kind='plot')
    registry.update({'p': p})
    q = PlotMeta(id='q', kind='plot')
    registry.setdefault('q', q)
    registry |= {'r': PlotMeta(id='r', kind='plot')}
    cursor, records = registry.collect_changes(0, viz.serialize_meta)
    assert [r['id'] for r in records] == ['p', 'q', 'r']
    assert registry.popitem()[0] == 'r'
    assert registry.pop('missing', None) is None
    del registry['q']
    cursor, _ = registry.collect_changes(cursor, viz.serialize_meta)
    q.dynamic = True
    assert registry.collect_changes(cursor, viz.serialize_meta) == (cursor, [])


def __test_native_writer_and_another_reader_observe_mutable_colors__(tmp_path):
    viz.reset_state()
    color = Color('FF0000')
    meta = PlotMeta(id='p', kind='plot', color=color)
    lib._plot_meta['p'] = meta
    lib._plot_meta_new.append(meta)
    path = tmp_path / 'meta.ndjson'
    writer = viz.VizWriter(path)
    writer.open()
    try:
        writer.write_bar(0, 0, {'p': 100.0}, {})
        assert lib._plot_meta_new == []
        cursor, initial = viz.collect_meta_changes()
        assert initial[0]['id'] == 'p'
        color.a = 30
        writer.write_bar(1, 300000, {'p': 101.0}, {})
        cursor, changed = viz.collect_meta_changes(cursor)
        assert changed[0]['color'] == viz.color_str(color)
    finally:
        writer.close()
        viz.reset_state()
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [rec['t'] for rec in records] == ['meta', 'bar', 'meta', 'bar']
    assert records[0] == initial[0] and records[2] == changed[0]
