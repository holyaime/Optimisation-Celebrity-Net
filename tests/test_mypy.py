# -*- coding: utf-8 -*-
from mypy import api


def test_add_with_execpted_args():
    result = api.run(["./for_mypy/add_with_execpted_args.py"])
    assert result[2] == 0


def test_add_with_unexecpted_args():
    result = api.run(["./for_mypy/add_with_unexecpted_args.py"])
    assert result[2] != 0
