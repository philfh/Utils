import pandas as pd
import numpy as np
from functools import partial
import seaborn as sns
import matplotlib.pyplot as plt

from skewstudent import SkewStudent
from Utils.risk_measures import risk_measure_np
from corr_spot_vol import CorrSpotVol

NSCEN = 10000

def rv_skewt(eta, lam, sdv):
    skewt = SkewStudent(eta=eta, lam=lam)
    return skewt.rvs(NSCEN) * sdv / np.sqrt(252)

def rv_corr_spot_vol(iv, vov, eta, lam, rho):
    csv = CorrSpotVol(iv, vov, eta, lam)
    csv.calc_corr_spot_vol(rho)
    return csv.spot_vol_return['IV Return'].to_numpy()

def get_ES(scens, plot_flag=True, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    var, es = risk_measure_np(scens)
    es = pd.DataFrame(es, index=scens.index, columns=['ES01', 'ES99'])
    if plot_flag:
        sns.scatterplot(data=es, ax=ax)
    return es

def plot_kde_es(scens):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    sns.kdeplot(data=scens.T, ax=axes[0])
    axes[0].set(xlim=(-0.5, 0.5), xlabel='IV Return')
    axes[0].xaxis.set_major_formatter("{x:.1%}")
    es = get_ES(scens, ax=axes[1])
    axes[1].set(ylim=(-1, 1), ylabel='IV return')
    # axes[1].xaxis.set_major_locator(MultipleLocator(10))
    axes[1].yaxis.set_major_formatter("{x:.1%}")
    print(es)

if __name__ == '__main__':
    eta, lams, vov = 4, [0.05, 0.1, 0.15, 0.2], 2
    scens = list(map(partial(rv_skewt, eta, sdv=vov), lams))
    scens = pd.DataFrame(scens, index=[f'lam{lam}' for lam in lams])
    plot_kde_es(scens)

    etas, lam, vov = [2.5, 3, 4, 5], 0.15, 2
    scens = list(map(partial(rv_skewt, lam=lam, sdv=vov), etas))
    scens = pd.DataFrame(scens, index=[f'eta{eta}' for eta in etas])
    plot_kde_es(scens)

    iv, rhos = 0.2, [-0.2, -0.4, -0.6, -0.8]
    scens = list(map(partial(rv_corr_spot_vol, iv, vov, eta, lam), rhos))
    scens = pd.DataFrame(scens, index=[f'rho{rho}' for rho in rhos])
    plot_kde_es(scens)
