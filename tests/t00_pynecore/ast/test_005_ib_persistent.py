"""
@pyne
"""
from pynecore import Persistent
from pynecore.types import IBPersistent


def main():
    var_count: Persistent[int] = 0
    varip_count: IBPersistent[int] = 0
    varip_total: IBPersistent[float] = 0.0
    var_count += 1
    varip_count += 1
    varip_total += some_value


def __test_ib_persistent__(ast_transformed_code, file_reader, log):
    """ IBPersistent (varip) generates both __persistent__ and __varip__ dicts """
    try:
        assert ast_transformed_code == file_reader(subdir="data", suffix="_ast_modified.py")
    except AssertionError:
        log.error("AST transformed code:\n%s\n", ast_transformed_code)
        log.warning("Expected code:\n%s\n", file_reader(subdir="data", suffix="_ast_modified.py"))
        raise
