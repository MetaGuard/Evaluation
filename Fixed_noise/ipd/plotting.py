import matplotlib.pyplot as plt
import seaborn as sns

def plot():

    epsilon_list = []
    noisy_mean_R_2_list = []
    relative_error_list = []
    absolute_error_list = []
    mean_pred_noisy_actual = []

    with open('results/epsilons.txt') as file:
        while (line := file.readline().rstrip()):
            epsilon_list.append(float(line))

    with open('results/relative_error.txt') as file:
        while (line := file.readline().rstrip()):
            relative_error_list.append(float(line))    
    
    with open('results/absolute_error.txt') as file:
        while (line := file.readline().rstrip()):
            absolute_error_list.append(float(line))   
            
    with open('results/noisy_mean_R_2_values.txt') as file:
        while (line := file.readline().rstrip()):
            noisy_mean_R_2_list.append(float(line))

    with open('results/mean_pred_on_noisy_actual.txt') as file:
        while (line := file.readline().rstrip()):
            mean_pred_noisy_actual.append(float(line))

    y = [relative_error_list, absolute_error_list, noisy_mean_R_2_list, mean_pred_noisy_actual]
    x_axis_label = ['Population mean of the relative error (%)', 'Population mean of the absolute error (mm)', 'Population mean of R²', 'Pupulation mean of average ipd prediction']
    file_names = ['2003_ipd_rel_error', '2003_ipd_abs_error', '2003_ipd_R_2', '2003_ipd_mean_pred_noisy_actual']
    y_lims_low = [0, 0, 0, 60]
    y_lims_up = [5, 5, 1, 70]
    for i, color in zip(range(0, 4), sns.color_palette()):

        part_fig, part_ax = plt.subplots(subplot_kw=dict(ylim=(0, 12)))
        part_ax.plot(epsilon_list, y[i], c=color)
        part_ax.set_ylim([y_lims_low[i], y_lims_up[i]])
        # part_ax.set_xticks(np.round(np.arange(0, 10, 0.1),2), minor=False)
        part_ax.set_xscale('log')
        if i == 0:
            # the rel error on prediction on actual data is 0.88%
            part_ax.plot(epsilon_list, [0.88]*len(epsilon_list), c='black')
            part_ax.legend(['Prediction on noisy data', 'Prediction on actual data'])
        if i == 1:
            # the abs error on prediction on actual data is 0.52 mm
            part_ax.plot(epsilon_list, [0.52]*len(epsilon_list), c='black')
            part_ax.legend(['Prediction on noisy data', 'Prediction on actual data'])
        if i == 2:
            # the coef of determination for the prediction on actual data is 0.974
            part_ax.plot(epsilon_list, [0.974]*len(epsilon_list), c='black')
            part_ax.legend(['Prediction on noisy data', 'Prediction on actual data'])
        if i == 3:
            # the predicted mean ipd on actual data is 63.233 mm
            part_ax.plot(epsilon_list, [62.233]*len(epsilon_list), c='black')
            part_ax.legend(['Prediction on noisy data', 'Prediction on actual data'])

        plt.xlabel('\u03B5', fontsize=14)
        plt.ylabel(x_axis_label[i], fontsize=14)
        plt.tight_layout()
        part_fig.savefig("figures/" + file_names[i] + "100_rounds_eps.pdf", format='pdf', bbox_inches='tight')

if __name__ == '__main__':

    plot()

    
