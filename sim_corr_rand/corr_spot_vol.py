import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from corr_rand_variables import CorrBiNormT2D, CorrNormSkewT2D

NSCEN = 10000

class CorrSpotVol:
    """ Simulate Correlated Log-returns of Spot and IV """

    def __init__(self, iv, vov, eta=4, lam=0.15):
        """
        Parameters
        ----------
        iv: implied vol
        vov: vov-of-vol
        eta: degree of freedom for Hanson
        lam: skewness for Hanson
        """
        self.corr_rv = CorrNormSkewT2D(eta=eta, lam=lam)
        self.iv = iv
        self.vov = vov
        self.rand_name1 = 'Spot Return'
        self.rand_name2 = 'IV Return'
        self.spot_vol_return = pd.DataFrame()

    def calc_corr_spot_vol(self, rho=-0.8):
        """ calculate spot and vol returns
        Parameters
        ----------
        rho: float
            linear correlation number
        Returns
        -------
        None
        """
        self.rho = rho
        self.corr_rv.calc_corr_rand(rho)
        scalar = np.array([self.iv, self.vov]) / np.sqrt(252)  # Scale unit-variance correlated RV with IV and VoV
        log_retn = self.corr_rv.corr_rands * np.tile(scalar, (NSCEN, 1))
        self.spot_vol_return = np.exp(log_retn) - 1  # convert log-return to proportional return
        self.spot_vol_return.columns = [self.rand_name1, self.rand_name2]

    def corr_spot_vol_stats(self):
        print('Statistical Summary of Simulated Spot and Vol Returns \n')
        print(self.spot_vol_return.describe(), '\n', self.spot_vol_return.corr())

    def plot_spot_vol_scatter(self, ax=None):
        if self.spot_vol_return.empty:
            print('correlated spot/vol return not calculated yet')
            return
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        sns.scatterplot(x=self.spot_vol_return[self.rand_name1], y=self.spot_vol_return[self.rand_name2], ax=ax)
        ax.set(xlabel=self.rand_name1, ylabel=self.rand_name2, title=f'Correlation = {self.rho}')

def main():
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    rhos = [-0.2, -0.4, -0.6, -0.8]
    iv, vov = 0.2, 2
    eta, lam = 4, 0.15
    csv = CorrSpotVol(iv=iv, vov=vov, eta=eta, lam=lam)
    for rho, ax in zip(rhos, axes.flatten()):
        csv.calc_corr_spot_vol(rho=rho)
        csv.plot_spot_vol_scatter(ax=ax)
    csv.corr_rv.plot_marginal_hist()
    plt.show()
    print(csv.corr_spot_vol_stats())

if __name__ == '__main__':
    main()