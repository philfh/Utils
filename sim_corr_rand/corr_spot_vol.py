import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from corr_rand_variables import CorrBiNormT2D, CorrNormSkewT2D

NSCEN = 10000

class CorrSpotVol:
    """ Simulate Correlated Log-returns of Spot and IV """

    def __init__(self, iv, vov, eta=4, lam=0.15):
        self.corr_rv = CorrNormSkewT2D(eta=eta, lam=lam)
        self.iv = iv
        self.vov = vov
        self.corr_rv.rand_name1 = 'Spot Return'
        self.corr_rv.rand_name2 = 'IV Return'
        self.spot_vol_return = pd.DataFrame()

    def gen_corr_spot_vol(self, rho=-0.8):
        self.corr_rv.gen_corr_rand(rho)
        scalar = np.array([self.iv, self.vov]) / np.sqrt(252)  # Scale unit-variance correlated RV with IV and VoV
        self.corr_rv.corr_rands *= np.tile(scalar, (NSCEN, 1))
        self.spot_vol_return = np.exp(self.corr_rv.corr_rands) - 1  # convert log-return to proportional return

    def corr_spot_vol_stats(self):
        print('Statistical Summary of Simulated Spot and Vol Returns \n')
        print(self.spot_vol_return.describe(), '\n', self.spot_vol_return.corr())

    def plot_spot_vol_scatter(self):
        self.corr_rv.plot_marginal_joint()

def main():
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    rhos = [-0.2, -0.4, -0.6, -0.8]
    iv, vov = 0.2, 2
    rho, eta, lam = -0.9, 4, 0.15
    csv = CorrSpotVol(iv=iv, vov=vov, eta=eta, lam=lam)
    for rho, ax in zip(rhos, axes.flatten()):
        csv.gen_corr_spot_vol(rho=rho)
        csv.corr_rv.plot_joint_scatter(ax=ax)
    csv.corr_rv.plot_marginal_hist()
    plt.show()

    print(csv.corr_spot_vol_stats())

if __name__ == '__main__':
    main()