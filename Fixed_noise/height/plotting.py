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
            relative_error_list.append(float(line)*100)    
    
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
    y_axis_label = ['Population mean of the relative error (%)', 'Population mean of the absolute error (m)', 'Population mean of R²', 'Pupulation mean of average height prediction']
    file_names = ['2000_height_rel_error', '2000_height_abs_error', '2000_height_R_2', '2000_height_mean_pred_noisy_actual']
    y_lims_low = [0, 0, 0, 1.55]
    y_lims_up = [15, 0.2, 1, 1.85]
    for i, color in zip(range(0, 4), sns.color_palette()):

        part_fig, part_ax = plt.subplots(subplot_kw=dict(ylim=(0, 12)))
        part_ax.plot(epsilon_list, y[i], c=color)
        part_ax.set_ylim([y_lims_low[i], y_lims_up[i]])
        # part_ax.set_xticks(np.round(np.arange(0, 10, 0.1),2), minor=False)
        part_ax.set_xscale('log')
        if i == 0:
            # the rel error on prediction on actual data is 1.87%
            part_ax.plot(epsilon_list, [1.87]*len(epsilon_list), c='black')
            part_ax.legend(['Prediction on noisy data', 'Prediction on actual data'])
        if i == 1:
            # the rel error on prediction on actual data is 3 cm
            part_ax.plot(epsilon_list, [0.03]*len(epsilon_list), c='black')
            part_ax.legend(['Prediction on noisy data', 'Prediction on actual data'])
        if i == 2:
            # the coef of determination for the prediction on actual data is 0.792
            part_ax.plot(epsilon_list, [0.792]*len(epsilon_list), c='black')
            part_ax.legend(['Prediction on noisy data', 'Prediction on actual data'])
        if i == 3:
            # the predicted mean height on actual data is 1.723
            part_ax.plot(epsilon_list, [1.723]*len(epsilon_list), c='black')
            part_ax.legend(['Prediction on noisy data', 'Prediction on actual data'])

        plt.xlabel('\u03B5', fontsize=14)
        plt.ylabel(y_axis_label[i], fontsize=14)
        plt.tight_layout()
        part_fig.savefig("figures/" + file_names[i] + "100_rounds_eps.pdf", format='pdf', bbox_inches='tight')

if __name__ == '__main__':

    plot()
    
