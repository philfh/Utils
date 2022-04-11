from pytest import approx
import numpy as np
import pandas as pd
import sys

def test1():
    iv = np.array([3.22, 3.22])
    print(iv)
    assert iv == approx(3.22)

def test2():
    iv = np.array([3.1, 3.1])
    assert (iv == 3.10).all()

if __name__ == '__main__':
    pass