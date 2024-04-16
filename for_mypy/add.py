# -*- coding: utf-8 -*-
def add(x: int, y: int) -> int:
    try:
        return x + y
    except TypeError:
        pass
