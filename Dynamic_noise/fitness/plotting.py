import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import numpy as np

def plot():

    epsilon_list = []
    population_mean_accuracy = []
    accuracy_without_noise = 90

    with open('results/epsilon_list.txt') as file:
        while (line := file.readline().rstrip()):
            epsilon_list.append(float(line))

    with open('results/population_mean_accuracy.txt') as file:
        while (line := file.readline().rstrip()):
            population_mean_accuracy.append(float(line))    
    
    plt.xscale('log')
    plt.xlabel('\u03B5', fontsize=14)
    plt.ylabel('Population mean of prediction accuracy (%)', fontsize=14)
    spl_1 = UnivariateSpline(epsilon_list, population_mean_accuracy, k=3)
    spl_1.set_smoothing_factor(0.5)
    xs = np.linspace(0, 10, 1000)
    plt.plot(epsilon_list, [accuracy_without_noise]*len(epsilon_list), c="r", linestyle='--', label='Actual')
    plt.plot(xs, spl_1(xs), c="b", label='Noisy')

    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('figures/2005_fitness_eps_accuracy_30_rounds.pdf', format='pdf', bbox_inches='tight')

if __name__ == '__main__':

    plot()
    