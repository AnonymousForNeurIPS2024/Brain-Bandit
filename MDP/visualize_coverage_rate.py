import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':

    Agents = ['EpsilonGreedy', 'PSRL',  'UBE_TS', 'UBE_UCB', 'UBE_BBNoptimistic', 'UBE_BBNneutral', 'UBE_BBNconservative','UCRL2','OptimisticPSRL']
    labels = ['EpsilonGreedy', 'PSRL',  'UBE_TS', 'UBE_UCB', 'UBE_BBN(optimistic)', 'UBE_BBN(neutral)', 'UBE_BBN(conservative)','UCRL2','optimisticPSRL']
    for i,Agent in enumerate(Agents):
        coverage = np.load('saved/L50/coverage_rate_' + Agent + '.npy')
        total_n = coverage.shape[0]
        L = coverage.shape[1]
        coverage_mean = np.mean(coverage, axis=0)
        coverage_std = np.std(coverage, axis=0)
        plt.fill_between(np.arange(1, L+1, 1), coverage_mean + coverage_std, coverage_mean-coverage_std, alpha=0.25)
        plt.plot(coverage_mean, label=labels[i])
    plt.legend()
    plt.xlabel('episodes')
    plt.ylabel('coverage rate')

    plt.show()
