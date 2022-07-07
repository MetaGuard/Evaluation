# Imports
import numpy as np
import matplotlib.pyplot as plt
import csv

# Source: https://diffprivlib.readthedocs.io/en/latest/modules/mechanisms.html?highlight=bounded#diffprivlib.mechanisms.LaplaceBoundedDomain
from diffprivlib.mechanisms.laplace import LaplaceBoundedDomain

def Laplace_Bounded_Mechanism(epsilon, sensitivity, lower, upper, deterministic_value):
    
    # Initialize differential privacy mechanism
    LapBoundMech = LaplaceBoundedDomain(epsilon=epsilon, sensitivity=sensitivity, lower=lower, upper=upper)
    #print(LapBoundMech.effective_epsilon()) The effective epsilon is the same as the one in the input
    # Produce differentially private output
    noisy_output = LapBoundMech.randomise(deterministic_value)
    return noisy_output

if __name__ == '__main__':

    # Starting and cleaning txt file (if it existed)
    f = open("results/statistics.txt","w")
    f.close()
    f = open("results/statistics.txt","r+")
    f.truncate(0)
    f.close()

    # Starting and cleaning txt file (if it existed)
    f = open("results/eps_selection.csv","w")
    f.close()
    f = open("results/eps_selection.csv","r+")
    f.truncate(0)
    f.close()

    # Epsilons 
    epsilons = [0.1, 0.5, 1, 3, 5, 10]
    # Physical parameters
    ratio_squat_to_height_threshold = 0.5
    height_to_wingspan_ratio = 1.04
    # Bounds: height (m), squat depth (m), wingspan (m), (room length (m) = room width (m)), IPD (mm)
    height_lower = 1.496 # 5th quantile of adult world population
    height_upper = 1.826 # 95th quantile of adult world population
    squat_depth_lower = 0
    squat_depth_upper = height_upper*(1-ratio_squat_to_height_threshold)
    wingspan_lower = height_lower*height_to_wingspan_ratio
    wingspan_upper = height_upper*height_to_wingspan_ratio
    arm_lower = 0.663
    arm_upper = 0.853
    room_dimension_lower = 0
    room_dimension_upper = 5
    ipd_lower = 55.696
    ipd_upper = 71.024
    lower = [height_lower, wingspan_lower, arm_lower, arm_lower, squat_depth_lower, room_dimension_lower, room_dimension_lower, ipd_lower]
    upper = [height_upper,  wingspan_upper, arm_upper, arm_upper, squat_depth_upper, room_dimension_upper, room_dimension_upper, ipd_upper]
    # Ground Truth
    ground_truth_V = [1.79,  1.73, 0.78, 0.82, 0.4475, 2.1, 2.9, 70]
    ground_truth_G = [1.85, 1.88, 0.82, 0.83,  0.4625, 2.1, 2.9, 63]
    # Privacy parameters
    num_rounds = 10000
    # labels
    labels = ["Height (m)", "Wingspan (m)", "Left arm (m)", "Right arm (m)", "Squat depth (m)", "Room length (m)", "Room width (m)", "IPD (mm)"]
    file_labels = ["Height_(m)", "Wingspan_(m)", "Left_arm_(m)", "Right_arm_(m)", "Squat_depth_(m)", "Room_length_(m)", "Room_width_(m)", "IPD_(mm)"]
	
    # CSV
    csv_header = ["Researcher", "Data_point", "Lower_bound", "Upper_bound", "Ground_truth", "ε=0.1_UPPER",	"ε=0.1_LOWER",\
    "ε=0.5_UPPER",	"ε=0.5_LOWER",	"ε=1_UPPER",	"ε=1_LOWER",	"ε=3_UPPER",	"ε=3_LOWER",	"ε=5_UPPER",\
    "ε=5_LOWER",	"ε=10_UPPER",	"ε=10_LOWER"]	

    rows = []
    for i in range(len(upper)):
        sensitivity = abs(upper[i]-lower[i])
        with open("results/statistics.txt", "a") as f:
            print("\n{}".format(labels[i]), file=f)
            f.close()

        noise_list_master_V = []
        noise_list_master_G = []
        for eps in epsilons:
            noisy_attribute_V = []
            noisy_attribute_G = []

            for _ in range(num_rounds):
                noisy_attribute_V.append(Laplace_Bounded_Mechanism(eps, sensitivity, lower[i], upper[i], ground_truth_V[i]))
                noisy_attribute_G.append(Laplace_Bounded_Mechanism(eps, sensitivity, lower[i], upper[i], ground_truth_G[i]))

            noise_list_master_V.append(noisy_attribute_V)
            noise_list_master_G.append(noisy_attribute_G)

            with open("results/statistics.txt", "a") as f:
                print("Epsilon = {}".format(eps), file=f)
                print("Researcher 1 UPPER = {}".format(np.percentile(noisy_attribute_V, 75)), file=f)
                print("Researcher 2 UPPER = {}".format(np.percentile(noisy_attribute_G, 75)), file=f)
                print("Researcher 1 LOWER = {}".format(np.percentile(noisy_attribute_V, 25)), file=f)
                print("Researcher 2 LOWER = {}".format(np.percentile(noisy_attribute_G, 25)), file=f)
                f.close()

        rows.append(["Researcher 1", labels[i], lower[i], upper[i], ground_truth_V[i], np.round(np.percentile(noise_list_master_V[0], 75), 3), np.round(np.percentile(noise_list_master_V[0], 25), 3), \
               np.round(np.percentile(noise_list_master_V[1], 75), 3), np.round(np.percentile(noise_list_master_V[1], 25), 3), np.round(np.percentile(noise_list_master_V[2], 75), 3), np.round(np.percentile(noise_list_master_V[2], 25), 3), \
               np.round(np.percentile(noise_list_master_V[3], 75), 3), np.round(np.percentile(noise_list_master_V[3], 25), 3), np.round(np.percentile(noise_list_master_V[4], 75), 3), np.round(np.percentile(noise_list_master_V[4], 25), 3), \
               np.round(np.percentile(noise_list_master_V[5], 75), 3), np.round(np.percentile(noise_list_master_V[5], 25), 3)])
        rows.append(["Researcher 2", labels[i], lower[i], upper[i], ground_truth_G[i], np.round(np.percentile(noise_list_master_G[0], 75), 3), np.round(np.percentile(noise_list_master_G[0], 25), 3), \
               np.round(np.percentile(noise_list_master_G[1], 75), 3), np.round(np.percentile(noise_list_master_G[1], 25), 3), np.round(np.percentile(noise_list_master_G[2], 75), 3), np.round(np.percentile(noise_list_master_G[2], 25), 3), \
               np.round(np.percentile(noise_list_master_G[3], 75), 3), np.round(np.percentile(noise_list_master_G[3], 25), 3), np.round(np.percentile(noise_list_master_G[4], 75), 3), np.round(np.percentile(noise_list_master_G[4], 25), 3), \
               np.round(np.percentile(noise_list_master_G[5], 75), 3), np.round(np.percentile(noise_list_master_G[5], 25), 3)])

        # Plotting
        fig = plt.figure(figsize =(10, 7))
        # Creating axes instance
        ax = fig.add_axes([0, 0, 1, 1])
        # Creating plot
        bp = ax.boxplot(noise_list_master_V, showmeans=True)
        plt.xticks([1, 2, 3, 4, 5, 6], epsilons)
        plt.xlabel('\u03B5')
        plt.ylabel('{}'.format(labels[i]))
        plt.savefig("figures/" + file_labels[i] + "_eps.pdf", format='pdf', bbox_inches='tight')

    with open('results/eps_selection.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
        for row in rows:
            writer.writerow(row)
        f.close()	
    
        
        
