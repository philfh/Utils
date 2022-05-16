#!/usr/bin/env python
import numpy as np
import pandas as pd
from scipy.linalg import cholesky
import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC, abstractmethod

from skewstudent import SkewStudent
NSCEN = 10000

class CorrRandVariables2D(ABC):
    ''' Simulated Correlated 2D Random Variables '''
    def __init__(self, rand_name1='Marginal1', rand_name2='Marginal2'):
        self.rand_name1 = f'{rand_name1} RV'
        self.rand_name2 = f'{rand_name2} RV'

    @abstractmethod
    def gen_rand_num1(self):
        ''' Marginal Distribution 1 '''
        raise NotImplementedError

    @abstractmethod
    def gen_rand_num2(self):
        ''' Marginal Distribution 2 '''
        raise NotImplementedError

    def gen_corr_rand(self, rho):
        ''' Generate Correlated Random Numbers: Corr RV = L x RV'''
        assert -1 <= rho <= 1, 'Correlation shall be [-1, 1]'
        self.rho = rho
        self.marginals = np.stack([self.rv1, self.rv2], axis=0)
        covar_mat = np.array([[1, self.rho], [self.rho, 1]])
        try:
            self.chol = cholesky(covar_mat, lower=True)
        except np.linalg.LinAlgError as e:
            print(f'Correlation Matrix Cholesky Decomposition Error: {e}')
        self.corr_rands = self.chol @ self.marginals
        self.corr_rands = pd.DataFrame(self.corr_rands, index=[self.rand_name1, self.rand_name2]).T

    def corr_rand_stats(self):
        print('Statistical Summary of Simulated Correlated Random Variables \n')
        print(self.corr_rands.describe(), '\n', self.corr_rands.corr())

    def plot_marginal_hist(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        df = pd.DataFrame(self.marginals, index=[self.rand_name1, self.rand_name2]).T
        sns.histplot(df, ax=ax, kde=True)
        # ax.legend([self.rand_name1, self.rand_name2]) # Caution: Wrong order
        # plt.show()

    def plot_joint_scatter(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        sns.scatterplot(x=self.corr_rands[self.rand_name1], y=self.corr_rands[self.rand_name2], ax=ax)
        ax.set(xlabel=self.rand_name1, ylabel=self.rand_name2, title=f'Correlation = {self.rho}')

    def plot_marginal_joint(self):
        fig, axes = plt.subplots(1, 2, figsize=(12,6))
        self.plot_marginal_hist(axes[0])
        self.plot_joint_scatter(axes[1])
        plt.show()

class CorrBiNormT2D(CorrRandVariables2D):
    ''' Simulate 2D Correlated Normal and Hanson Skew-T Random Numbers'''
    def __init__(self):
        super().__init__('Normal 1', 'Normal 2')
        self.rv1 = self.gen_rand_num1()
        self.rv2 = self.gen_rand_num2()
        self.gen_corr_rand()

    def gen_rand_num1(self):
        return np.random.normal(size=NSCEN)

    def gen_rand_num2(self):
        return np.random.normal(size=NSCEN)

class CorrNormSkewT2D(CorrRandVariables2D):
    ''' Simulate 2D Correlated Normal and Hanson Skew-T Random Numbers'''
    def __init__(self, eta=4, lam=0.15):
        super().__init__('Normal', 'Hanson Skew-T')
        self.rv1 = self.gen_rand_num1()
        self.rv2 = self.gen_rand_num2(eta=eta, lam=lam)

    def gen_rand_num1(self):
        return np.random.normal(size=NSCEN)

    def gen_rand_num2(self, eta, lam):
        skewt = SkewStudent(eta=eta, lam=lam)
        return skewt.rvs(NSCEN)

def main():
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    rhos = [-0.2, -0.4, -0.6, -0.8]
    rho, eta, lam = -0.2, 4, 0.15
    corr_rv = CorrNormSkewT2D(eta=eta, lam=lam)
    for rho, ax in zip(rhos, axes.flatten()):
        corr_rv.gen_corr_rand(rho=rho)
        corr_rv.plot_joint_scatter(ax=ax)
    corr_rv.plot_marginal_hist()
    plt.show()

if __name__ == '__main__':
    main()
    # rv = CorrBiNormT2D(rho=-0.8)
    # rv.plot_marginal_joint()





