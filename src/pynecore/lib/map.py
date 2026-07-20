from typing import TypeVar, Any
from pynecore.types.na import NA

TKey = TypeVar('TKey')
TValue = TypeVar('TValue')


# noinspection PyShadowingBuiltins
def clear(id: dict):
    """
    Clear the map.

    :param id: The map to clear.
    """
    id.clear()


# noinspection PyShadowingBuiltins
def contains(id: dict, key: Any) -> bool:
    """
    Check if the map contains a key.

    :param id: The map to check.
    :param key: The key to check for.
    """
    return key in id


# noinspection PyShadowingBuiltins
def copy(id: dict[TKey, TValue]) -> dict[TKey, TValue]:
    """
    Copy the map.

    :param id: The map to copy.
    """
    return id.copy()


# noinspection PyShadowingBuiltins
def get(id: dict[TKey, TValue], key: TKey) -> TValue:
    """
    Get the value of a key in the map.

    :param id: The map to get the value from.
    :param key: The key to get the value from.
    :return: The value associated with the key, or na when the key is absent.
    """
    try:
        return id[key]
    except KeyError:
        return NA(None)  # value type is unknown at run-time (see ``remove``)


# noinspection PyShadowingBuiltins
def keys(id: dict[TKey, TValue]) -> list[TKey]:
    """
    Get the keys of the map.

    :param id: The map to get the keys from.
    """
    return list(id.keys())


def new() -> dict:
    """
    Create a new map object.

    :return: A new map object.
    """
    return {}


# noinspection PyShadowingBuiltins
def put(id: dict, key: Any, value: TValue) -> TValue:
    """
    Put a key-value pair in the map.

    :param id: The map to put the key-value pair in.
    :param key: The key to put in the map.
    :param value: The value to put in the map.
    :return: The value that was previously in the map.
    """
    try:
        old_value = id[key]
    except KeyError:
        old_value = NA(type(value))
    id[key] = value
    return old_value


# noinspection PyShadowingBuiltins
def put_all(id: dict[TKey, TValue], other: dict[TKey, TValue]):
    """
    Put all the key-value pairs from another map into this map.

    :param id: The map to put the key-value pairs in.
    :param other: The map to put the key-value pairs from.
    """
    id.update(other)


# noinspection PyShadowingBuiltins
def remove(id: dict[TKey, TValue], key: TKey) -> TValue:
    """
    Remove a key-value pair from the map.

    :param id: The map to remove the key-value pair from.
    :param key: The key to remove from the map.
    :return: The value that was removed from the map.
    """
    try:
        return id.pop(key)
    except KeyError:
        return NA(None)  # We don't no the type of the map values here run-time


# noinspection PyShadowingBuiltins
def size(id: dict[TKey, TValue]) -> int:
    """
    Get the size of the map.

    :param id: The map to get the size of.
    :return: The size of the map.
    """
    return len(id)


# noinspection PyShadowingBuiltins
def values(id: dict[TKey, TValue]) -> list[TValue]:
    """
    Get the values of the map.

    :param id: The map to get the values from.
    :return: A list of the values in the map.
    """
    return list(id.values())
