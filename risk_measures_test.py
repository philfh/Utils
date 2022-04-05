from pytest import approx
from risk_measures import risk_measure_np
import numpy as np

def test_risk_measure_np_1d():
    hi, lo, mid = 987, 1.4, 501
    dat = 100*[hi] + 100*[lo] + 9800*[mid]
    VaR, ES = risk_measure_np(dat, ci=0.99)
    assert VaR.flatten() == approx(np.array([lo, hi]))

def test_risk_measure_np_2d():
    arr = np.array(10*[range(10_000)])
    VaR, ES = risk_measure_np(arr, ci=0.99)
    assert VaR == approx(np.tile(np.array([99, 9900]), (10, 1)))

if __name__ == '__main__':
    pass