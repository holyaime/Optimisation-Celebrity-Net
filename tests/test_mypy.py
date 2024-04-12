# -*- coding: utf-8 -*-
from mypy import api


def test_add_with_execpted_args():
    """
    Here, we test add function with excepted types for args and
    we check if exit_status returns by api.run is equal to 0...
    NB : Function api.run returns a tuple[str, str, int], namely (<normal_report>, <error_report>, <exit_status>)
    """
    result = api.run(["./for_mypy/add_with_execpted_args.py"])
    assert result[2] == 0


def test_add_with_unexecpted_args():
    result = api.run(["./for_mypy/add_with_unexecpted_args.py"])
    assert result[2] != 0
