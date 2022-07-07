import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.interpolate import UnivariateSpline
# Import the privacy parameter from the script that created the data

def plot_room_trajectory(epsilon):
    
    participant_index_list = np.arange(0,32)

    for participant_index in participant_index_list:
        
        camera_x = []
        camera_z = []
        noisy_camera_x = []
        noisy_camera_z = []

        with open('results/camera_x_{}_eps_{}.txt'.format(participant_index, epsilon)) as file:
            while (line := file.readline().rstrip()):
                camera_x.append(float(line))    

        with open('results/camera_z_{}_eps_{}.txt'.format(participant_index, epsilon)) as file:
            while (line := file.readline().rstrip()):
                camera_z.append(float(line))  

        with open('results/noisy_camera_x_{}_eps_{}.txt'.format(participant_index, epsilon)) as file:
            while (line := file.readline().rstrip()):
                noisy_camera_x.append(float(line))  

        with open('results/noisy_camera_z_{}_eps_{}.txt'.format(participant_index, epsilon)) as file:
            while (line := file.readline().rstrip()):
                noisy_camera_z.append(float(line))  
        
        left = np.percentile(camera_x, 0)
        right = np.percentile(camera_x, 99.9)
        bottom = np.percentile(camera_z, 0)
        top = np.percentile(camera_z, 99.9)
        width = right - left
        length = top - bottom
        center_x = left + (width/2)
        center_z = bottom + (length/2)

        noisy_left = np.percentile(noisy_camera_x, 0)
        noisy_right = np.percentile(noisy_camera_x, 99.9)
        noisy_bottom = np.percentile(noisy_camera_z, 0)
        noisy_top = np.percentile(noisy_camera_z, 99.9)
        noisy_width = noisy_right - noisy_left
        noisy_length = noisy_top - noisy_bottom

        # Ground truth
        real_w = [2.9, 1.6, 2.5]
        real_l = [2.1, 2, 1.5]

        ## for the room where the Vive Pro 2 was used, so index in real_l and real_w is 0
        device_index = 0
        part_fig, part_ax = plt.subplots()
        # plt.title('2D Location Data: Actual vs. Predicted Room Size (\u03B5={})'.format(epsilon))
        part_ax.plot(camera_x, camera_z, 'red')
        part_ax.plot(noisy_camera_x, noisy_camera_z, 'm')
        ax = plt.gca()
        # Without protection
        rect = patches.Rectangle((left, bottom), width, length, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        part_ax.annotate("Observed Width: " + str(round(width, 2)) + "m\nObserved Length: " + str(round(length, 2)) + "m", (left + 0.05, bottom + 0.05), color='r')
        # Ground truth
        rect = patches.Rectangle((center_x - real_w[device_index]/2, center_z - real_l[device_index]/2), real_w[device_index], real_l[device_index], linewidth=2, edgecolor='b', facecolor='none')
        ax.add_patch(rect)
        part_ax.annotate("Actual Width: " + str(round(real_w[device_index], 2)) + "m\nActual Length: " + str(round(real_l[device_index], 2)) + "m", (right - 0.05, bottom + 0.05), color='b', ha='right')
        # With protection
        rect = patches.Rectangle((noisy_left, noisy_bottom), noisy_width, noisy_length, linewidth=2, edgecolor='m', facecolor='none')
        ax.add_patch(rect)    
        part_ax.annotate("Observed Width (noisy): " + str(round(noisy_width, 2)) + "m\nObserved Length (noisy): " + str(round(noisy_length, 2)) + "m", (noisy_left + 0.05, noisy_top - 0.25), color='m')
        
        plt.xlabel('X coordinates', fontsize=14)
        plt.ylabel('Z coordinates', fontsize=14)
        plt.tight_layout()
        part_fig.savefig('figures/2004_room_size_trajectory_participant_{}_eps_{}.pdf'.format(participant_index, epsilon), format='pdf', bbox_inches='tight')
        
def plot_R_2():

    epsilon_list = []
    with open('results/epsilon_list.txt') as file:
        while (line := file.readline().rstrip()):
            epsilon_list.append(float(line)) 

    population_mean_noisy_width_R_list = []
    with open('results/population_mean_noisy_width_R_list.txt') as file:
        while (line := file.readline().rstrip()):
            population_mean_noisy_width_R_list.append(float(line)) 

    population_mean_noisy_length_R_list = []
    with open('results/population_mean_noisy_length_R_list.txt') as file:
        while (line := file.readline().rstrip()):
            population_mean_noisy_length_R_list.append(float(line)) 

    population_mean_noisy_area_R_list = []
    with open('results/population_mean_noisy_area_R_list.txt') as file:
        while (line := file.readline().rstrip()):
            population_mean_noisy_area_R_list.append(float(line)) 

    y = [population_mean_noisy_width_R_list, population_mean_noisy_length_R_list, population_mean_noisy_area_R_list]
    y_axis_label = ['Population mean of R² value of width', 'Population mean of R² value of length', 'Population mean of R² value of area']
    file_names = ['2004_width_R2', '2004_length_R2', '2004_area_R2']
    y_lims_low = [0, 0, 0]
    y_lims_up = [1, 1, 1]
    for i in range(0, 3):

        part_fig, part_ax = plt.subplots()
        spl_1 = UnivariateSpline(epsilon_list, y[i])
        spl_1.set_smoothing_factor(0.5)
        xs = np.linspace(0, 10, 1000)
        part_ax.plot(xs, spl_1(xs), c="b")
        part_ax.set_ylim([y_lims_low[i], y_lims_up[i]])
        # part_ax.set_xticks(np.round(np.arange(0, 10, 0.1),2), minor=False)
        part_ax.set_xscale('log')
        if i == 0:
            part_ax.plot(epsilon_list, [0.977]*len(epsilon_list), c='black')
            part_ax.legend(['Prediction on noisy data', 'Prediction on actual data'])
        if i == 1:
            part_ax.plot(epsilon_list, [0.999]*len(epsilon_list), c='black')
            part_ax.legend(['Prediction on noisy data', 'Prediction on actual data'])
        if i == 2:
            part_ax.plot(epsilon_list, [0.974]*len(epsilon_list), c='black')
            part_ax.legend(['Prediction on noisy data', 'Prediction on actual data'])

        plt.xlabel('\u03B5', fontsize=14)
        plt.ylabel(y_axis_label[i], fontsize=14)
        plt.tight_layout()
        part_fig.savefig("figures/" + file_names[i] + "_eps.pdf", format='pdf', bbox_inches='tight')

def plot_accuracies():

    epsilon_list = []
    with open('results/epsilon_list.txt') as file:
        while (line := file.readline().rstrip()):
            epsilon_list.append(float(line)) 

    population_mean_noisy_within_1_m = []
    with open('results/population_mean_noisy_within_1_m.txt') as file:
        while (line := file.readline().rstrip()):
            population_mean_noisy_within_1_m.append(float(line)) 

    population_mean_noisy_within_2_m = []
    with open('results/population_mean_noisy_within_2_m.txt') as file:
        while (line := file.readline().rstrip()):
            population_mean_noisy_within_2_m.append(float(line)) 

    population_mean_noisy_within_3_m = []
    with open('results/population_mean_noisy_within_3_m.txt') as file:
        while (line := file.readline().rstrip()):
            population_mean_noisy_within_3_m.append(float(line)) 

    y = [population_mean_noisy_within_1_m, population_mean_noisy_within_2_m, population_mean_noisy_within_3_m]
    y_axis_label = ['Population mean of accuracy (%)', 'Population mean of accuracy (%)', 'Population mean of accuracy (%)']
    file_names = ['2004_1_m_accuracy', '2004_2_m_accuracy', '2004_3_m_accuracy']
    y_lims_low = [0, 0, 0]
    y_lims_up = [100, 100, 100]
    for i in range(0, 3):

        part_fig, part_ax = plt.subplots()
        spl_1 = UnivariateSpline(epsilon_list, y[i])
        spl_1.set_smoothing_factor(0.5)
        xs = np.linspace(0, 10, 1000)
        part_ax.plot(xs, spl_1(xs), c="b")
        part_ax.set_ylim([y_lims_low[i], y_lims_up[i]])
        # part_ax.set_xticks(np.round(np.arange(0, 10, 0.1),2), minor=False)
        part_ax.set_xscale('log')
        if i == 0:
            part_ax.plot(epsilon_list, [34.4]*len(epsilon_list), c='black')
            part_ax.legend(['Prediction on noisy data', 'Prediction on actual data'])
        if i == 1:
            part_ax.plot(epsilon_list, [78.1]*len(epsilon_list), c='black')
            part_ax.legend(['Prediction on noisy data', 'Prediction on actual data'])
        if i == 2:
            part_ax.plot(epsilon_list, [96.9]*len(epsilon_list), c='black')
            part_ax.legend(['Prediction on noisy data', 'Prediction on actual data'])

        plt.xlabel('\u03B5', fontsize=14)
        plt.ylabel(y_axis_label[i], fontsize=14)
        plt.tight_layout()
        part_fig.savefig("figures/" + file_names[i] + "_eps.pdf", format='pdf', bbox_inches='tight')

if __name__ == '__main__':

    epsilon = 5
    plot_room_trajectory(epsilon)

    plot_R_2()

    plot_accuracies()
    