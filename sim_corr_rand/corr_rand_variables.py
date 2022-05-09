#!/usr/bin/env python
import numpy as np
from scipy.linalg import cholesky
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from skewstudent import SkewStudent

NSCEN = 100000

class CorrRandVariables2D:
    ''' Simulated Correlated 2D Random Variables '''
    def __init__(self, rho, rand_name1='Marginal1', rand_name2='Marginal2'):
        self.rho = rho
        self.rand_name1 = f'{rand_name1} RV'
        self.rand_name2 = f'{rand_name2} RV'

    def gen_rand_num1(self):
        ''' Marginal Distribution 1 '''
        raise NotImplementedError

    def gen_rand_num2(self):
        ''' Marginal Distribution 2 '''
        raise NotImplementedError

    def gen_corr_rand(self):
        ''' Generate Correlated Random Numbers: Corr RV = L x RV'''
        self.marginals = np.stack([self.rv1, self.rv2], axis=0)
        covar_mat = np.array([[1, self.rho], [self.rho, 1]])
        self.chol = cholesky(covar_mat, lower=True)
        self.corr_rands = self.chol @ self.marginals
        return self.corr_rands

    def plot_marginal_hist(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        sns.histplot([self.marginals[0], self.marginals[1]], ax=ax, kde=True)
        ax.legend([self.rand_name1, self.rand_name2])

    def plot_joint_scatter(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        sns.scatterplot(self.corr_rands[0], self.corr_rands[1], ax=ax)
        ax.set(xlabel=self.rand_name1, ylabel=self.rand_name2, title=f'Correlation = {self.rho}')

    def plot_marginal_joint(self):
        fig, axes = plt.subplots(1, 2, figsize=(12,6))
        self.plot_marginal_hist(axes[0])
        self.plot_joint_scatter(axes[1])
        # axes[1].scatter(corr_rands[0], corr_rands[1])
        plt.show()


class CorrNormSkewT2D(CorrRandVariables2D):
    ''' Simulate 2D Correlated Normal and Hanson Skew-T Random Numbers'''
    def __init__(self, rho, eta=5, lam=-0.5):
        super().__init__(rho, 'Normal', 'Hanson Skew-T')
        self.rv1 = self.gen_rand_num1()
        self.rv2 = self.gen_rand_num2(eta=eta, lam=lam)
        self.gen_corr_rand()

    def gen_rand_num1(self):
        return np.random.normal(size=NSCEN)

    def gen_rand_num2(self, eta, lam):
        skewt = SkewStudent(eta=eta, lam=lam)
        return skewt.rvs(NSCEN)

def main():
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    rhos = [-0.2, -0.4, -0.6, -0.8]
    eta, lam = 4, 0.5
    for rho, ax in zip(rhos, axes.flatten()):
        corr_rv = CorrNormSkewT2D(rho=rho, eta=eta, lam=lam)
        corr_rv.plot_joint_scatter(ax=ax)
    corr_rv.plot_marginal_hist()
    plt.show()

if __name__ == '__main__':
    main()





