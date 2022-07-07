import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import numpy as np

def plot():

    epsilon_list = []
    population_mean_of_percent_num_people_low = []
    population_mean_of_percent_num_people_high = []
    accuracy_on_actual_upper_threshold= 100
    accuracy_on_actual_lower_threshold = 64.71

    with open('results/epsilons.txt') as file:
        while (line := file.readline().rstrip()):
            epsilon_list.append(float(line))

    with open('results/population_mean_percent_num_people_high.txt') as file:
        while (line := file.readline().rstrip()):
            population_mean_of_percent_num_people_high.append(float(line))    
    
    with open('results/population_mean_percent_num_people_low.txt') as file:
        while (line := file.readline().rstrip()):
            population_mean_of_percent_num_people_low.append(float(line))   

    plt.xscale('log')
    plt.xlabel('\u03B5', fontsize=14)
    plt.ylabel('Population mean of prediction accuracy (%)', fontsize=14)
    spl_1 = UnivariateSpline(epsilon_list, population_mean_of_percent_num_people_low, k=3)
    spl_2 = UnivariateSpline(epsilon_list, population_mean_of_percent_num_people_high, k=3)
    xs = np.linspace(0, 10, 1000)
    plt.plot(epsilon_list, [accuracy_on_actual_upper_threshold]*len(epsilon_list), c="r", linestyle='--', label='≥ 3 cm (Actual)')
    plt.plot(xs, spl_1(xs), c="b", label='≥ 3 cm (Noisy)')
    plt.plot(epsilon_list, [accuracy_on_actual_lower_threshold]*len(epsilon_list), c="m", linestyle='--',label='≥ 1 cm (Actual)')
    plt.plot(xs, spl_2(xs), c="c", label='≥ 1 cm (Noisy)')

    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('figures/2002_longer_arm_eps_accuracy.pdf', format='pdf', bbox_inches='tight')

if __name__ == '__main__':

    plot()
    