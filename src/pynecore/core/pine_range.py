from pynecore.types.pine_types import PyneFloat, PyneInt


def pine_range(from_num: PyneInt | PyneFloat, to_num: PyneInt | PyneFloat, step_num: PyneInt | PyneFloat | None = None):
    """
    Emulates Pine Script's for loop range behavior.

    :param from_num: Start value (inclusive)
    :param to_num: End value (inclusive)
    :param step_num: Step value (optional, defaults to +1/-1 based on direction)
    :return: A generator that yields values from from_num to to_num (inclusive)
    :raises ValueError: If step_num is zero
    """
    # Determine direction based on from_num and to_num
    direction = 1 if from_num <= to_num else -1

    # Use default step if none provided
    step_val = step_num if step_num is not None else direction

    # Prevent infinite loops
    if step_val == 0:
        raise ValueError("Step cannot be zero in pine_range")

    # Ensure step direction matches the from->to direction
    if (direction > 0 > step_val) or (direction < 0 < step_val):
        step_val = -step_val

    # Generate values
    current = from_num
    if direction > 0:
        # Ascending loop
        while current <= to_num:
            yield current
            current += step_val
            # Safety check to prevent infinite loops due to floating point precision
            if step_val > 0 and current > to_num + abs(step_val):
                break
    else:
        # Descending loop
        while current >= to_num:
            yield current
            current += step_val
            # Safety check to prevent infinite loops due to floating point precision
            if step_val < 0 and current < to_num - abs(step_val):
                break
