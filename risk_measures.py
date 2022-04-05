import numpy as np

def risk_measure_np(dat, ci=0.99):
    arr = np.array(dat)
    if arr.ndim == 1:
        arr = np.reshape(arr, (1, -1))
    arr_sort = np.sort(arr)
    left_idx, right_idx = int(arr.shape[1] * (1-ci)) - 1, int(arr.shape[1] * ci) # idex = 99, 9900
    VaR = arr_sort[:, [left_idx, right_idx]]
    left_tail, right_tail = arr_sort[:, :left_idx], arr_sort[:, right_idx:]
    ES1, ES99 = np.mean(left_tail, axis=1), np.mean(right_tail, axis=1)
    ES = np.stack((ES1, ES99), axis=1)
    return VaR, ES

def test_risk_measure_np_1d():
    hi, lo, mid = 987, 1.4, 501
    dat = 100*[hi] + 100*[lo] + 9800*[mid]
    VaR, ES = risk_measure_np(dat, ci=0.99)


if __name__ == '__main__':
    arr = np.array(10*[range(10_000)])
    VaR, ES = risk_measure_np(arr, ci=0.99)
    print(VaR, ES)