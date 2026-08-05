"""
@pyne
"""
from pynecore.lib import script, log, bar_index, map, na


@script.indicator(title="Map Basic", shorttitle="map_basic")
def main():
    if bar_index == 0:
        # Create a new map
        m = map.new()
        log.info("map.size(m): {0}", map.size(m))

        # Insert key-value pairs
        map.put(m, "one", 1)
        map.put(m, "two", 2)
        map.put(m, "three", 3)

        # Log key-value pairs individually
        log.info("Map contents - one: {0}", map.get(m, "one"))
        log.info("Map contents - two: {0}", map.get(m, "two"))
        log.info("Map contents - three: {0}", map.get(m, "three"))
        log.info("map.size(m): {0}", map.size(m))

        # Get values
        log.info("map.get(m, 'one'): {0}", map.get(m, "one"))
        log.info("map.get(m, 'two'): {0}", map.get(m, "two"))
        log.info("map.get(m, 'three'): {0}", map.get(m, "three"))

        # Contains key check
        log.info("map.contains(m, 'one'): {0}", map.contains(m, "one"))
        log.info("map.contains(m, 'four'): {0}", map.contains(m, "four"))

        # Put with existing key returns old value
        old_value = map.put(m, "one", 100)
        log.info("Old value for 'one': {0}", old_value)
        log.info("New value for 'one': {0}", map.get(m, "one"))

        # Get keys and values
        keys = map.keys(m)
        log.info("map.keys(m): {0}", keys)
        values = map.values(m)
        log.info("map.values(m): {0}", values)

        # Copy the map
        m2 = map.copy(m)

        # Compare the maps by checking their values
        log.info("m2 (copy of m) - one: {0}", map.get(m2, "one"))
        log.info("m2 (copy of m) - three: {0}", map.get(m2, "three"))

        # Update the copy
        map.put(m2, "four", 4)
        log.info("m2 after adding 'four': {0}", map.get(m2, "four"))
        log.info("m2 size after adding 'four': {0}", map.size(m2))
        log.info("original m size: {0}", map.size(m))

        # Remove a key
        removed = map.remove(m, "two")
        log.info("Removed value: {0}", removed)
        log.info("m contains 'two' after removing: {0}", map.contains(m, "two"))
        log.info("map.size(m): {0}", map.size(m))

        # Try to remove a non-existent key
        removed = map.remove(m, "nonexistent")
        log.info("Removed non-existent value is na: {0}", na(removed))

        # Put all (merge maps)
        other_map = map.new()
        map.put(other_map, "five", 5)
        map.put(other_map, "six", 6)
        map.put_all(m, other_map)

        # Show contents after put_all
        log.info("m after put_all contains 'five': {0} with value: {1}",
                 map.contains(m, "five"), map.get(m, "five"))
        log.info("m after put_all contains 'six': {0} with value: {1}",
                 map.contains(m, "six"), map.get(m, "six"))
        log.info("m size after put_all: {0}", map.size(m))

        # Clear the map
        map.clear(m)
        log.info("m size after clear: {0}", map.size(m))
        log.info("m contains 'one' after clear: {0}", map.contains(m, "one"))


def __test_map_basic__(runner, dummy_ohlcv_iter, file_reader, log_comparator):
    """ Basic """
    tv_log_out = file_reader(subdir="data", suffix=".txt")
    run_iter = runner(dummy_ohlcv_iter).run_iter()
    with log_comparator(tv_log_out):
        next(run_iter)


def __test_map_put_all_id2__():
    """ map.put_all names its second parameter the way Pine names it, so a compiled
    named argument binds """
    from pynecore.lib import map as _map
    target = {"one": 1}
    source = {"two": 2}
    _map.put_all(id=target, id2=source)
    assert target == {"one": 1, "two": 2}
    # The source map is left untouched
    assert source == {"two": 2}
    # Existing keys are overwritten
    _map.put_all(target, {"one": 100})
    assert target["one"] == 100
