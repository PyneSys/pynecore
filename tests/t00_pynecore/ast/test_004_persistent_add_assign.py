"""
@pyne
"""
from pynecore import Persistent


def main():
    cumulative: Persistent[float] = 0.0
    cumulative += some_value
    counter: Persistent[int] = 0
    counter += 1


def __test_persistent_add_assign__(ast_transformed_code, file_reader, log):
    """ += on a Persistent stays a plain augmented assignment on its slot """
    try:
        assert ast_transformed_code == file_reader(subdir="data", suffix="_ast_modified.py")
    except AssertionError:
        log.error("AST transformed code:\n%s\n", ast_transformed_code)
        log.warning("Expected code:\n%s\n", file_reader(subdir="data", suffix="_ast_modified.py"))
        raise