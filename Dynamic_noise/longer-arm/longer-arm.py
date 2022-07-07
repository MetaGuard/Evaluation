import glob
import os
import scipy.stats
import numpy as np
from DP_guard import Laplace_Bounded_Mechanism
from plotting import plot

# We create a file to store results
f = open("results/avg_statistics.txt","w")
f.close()
f = open("results/avg_statistics.txt","r+")
f.truncate(0)
f.close()

# Calculate confidence interval of the population mean without knowing the std of the population
# From: https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data
def mean_confidence_interval(data, confidence=0.99):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return '{} ±{}'.format(m, h), [m, m-h, m+h]

# This function calculates the prediction accuracy based on a threshold
def longer_arm_prediction_accuracy(threshold, l_actual, r_actual, l_pred, r_pred):
    correct = 0
    incorrect = 0
    for i in range(len(l_actual)):
        if (abs(l_actual[i] - r_actual[i]) <= threshold): continue
        if (l_actual[i] > r_actual[i]):
            if (l_pred[i] > r_pred[i]): correct += 1
            else: incorrect += 1
        elif (l_actual[i] < r_actual[i]):
            if (l_pred[i] < r_pred[i]): correct += 1
            else: incorrect += 1
    
    percent = round((correct / (correct + incorrect))*100,2)
    
    return percent, str(correct) + "/" + str(correct+incorrect) + " (" + str(percent) + "%)"

# Wingspan range. Based on this ref: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7928122/
# Height and wingspan are highly correlated. The coefficient of determination for both male and female is above 0.988 between height and wingspan
# The average ratio (across males and females) is 1.04 (wingspan/height)
height_to_wingspan_ratio = 1.04
height_lower = 1.496 # 5th quantile of adult world population
height_upper = 1.826 # 95th quantile of adult world population
# For DP, we will use these bounds for arm length (we divide by two, as the thresholds are per arm)
wingspan_lower = height_to_wingspan_ratio*height_lower
wingspan_upper = height_to_wingspan_ratio*height_upper
arm_asymmetry_ratio_lower = 0.95 # from in-situ anthropometric survey 
arm_asymmetry_ratio_upper = 1.05 # from in-situ anthropometric survey 

# Privacy parameter
eps_1 = np.arange(0.01,.1, 0.01)
eps_2 = np.arange(0.1,1, 0.1)
eps_3 = np.arange(1,11, 1)
epsilon_list = np.concatenate([eps_1, eps_2, eps_3])
wingspan_sensitivity = np.abs(wingspan_upper-wingspan_lower)
arm_asymmetry_ratio_sensitivity = np.abs(arm_asymmetry_ratio_upper-arm_asymmetry_ratio_lower)

# Number of iterations to calculate average results
num_rounds = 30

# We will measure accuracies for different thresholds of difference between arms
upper_threshold = 0.03
lower_threshold = 0.011

# List to save results
population_mean_of_percent_upper= []
conf_interval_percent_upper = []
population_mean_of_percent_lower = []
conf_interval_percent_lower = []

for epsilon in epsilon_list:

    percent_prediction_on_noisy_upper_threshold = []
    percent_prediction_on_noisy_lower_threshold = []
    noisy_right_list = []
    noisy_left_list = []
    for _ in range(0, num_rounds):

        # Stores the left's and right's arm actual data
        l_actual = []
        r_actual = []
        # Stores the left's and right's arm predicted data
        l_predicted = []
        r_predicted = []
        # Stores the left's and right's arm noisy predicted data
        l_predicted_noisy = []
        r_predicted_noisy = []

        os.chdir("data/")
        for name in glob.glob("*.dat"):
            # Lists used to store the raw signals
            dleft = []
            dright = []
            dist = []
            with open(name) as file:
                # We read the device type
                device = file.readline().strip()
                # We read the ground truth for the left and right arm
                l_actual.append(float(file.readline().strip()))
                r_actual.append(float(file.readline().strip()))
                # We read the signals from the VR device
                for line in file:
                    parts = line.split(', ')
                    dist.append(float(parts[0]))
                    dleft.append(float(parts[1]))
                    dright.append(float(parts[2]))

            # We focus on the middle of the playthrough
            dist = dist[12000:-12000]
            # We eliminte possible outliers, like leaving the cotrollers somewhere esle
            if (max(dist[0:25000]) >= 6): dist = dist[25000:]
            if (max(dist[0:5000]) >= 2.5): dist = dist[5000:]
            while (max(dist) >= 6): dist = dist[:-5000]
            while (max(dist) >= 2): dist = dist[5000:]

            # We select the offset depending on the device
            if device == 'Oculus Quest 2':
                offset_l = 0.31
                offset_r = 0.19
            if device == 'HTC Vive':
                offset_l = 0.59
                offset_r = 0.57
            if device == 'Vive Pro 2':
                offset_l = 0.42
                offset_r = 0.52
            
            # The wingspan prediction will be the maximum distance between controllers
            wingspan = max(dist)
            # based on this signal, we find the left and right arm measurements
            index = dist.index(wingspan)
            left = dleft[index]
            right = dright[index]
            # We apply noise to the arm lengths. 
            arm_asymmetry_ratio = (right+offset_r)/(left+offset_l)
            noisy_arm_asymmetry_ratio = Laplace_Bounded_Mechanism(epsilon, arm_asymmetry_ratio_sensitivity, arm_asymmetry_ratio_lower, arm_asymmetry_ratio_upper, arm_asymmetry_ratio)
            noisy_right = ((wingspan + offset_l + offset_r)/2)*noisy_arm_asymmetry_ratio
            noisy_left = abs((wingspan + offset_l + offset_r) - noisy_right)

            # noisy_right = Laplace_Bounded_Mechanism(epsilon, arm_length_sensitivity, arm_length_lower, arm_length_upper, right+offset_r)
            # noisy_left = Laplace_Bounded_Mechanism(epsilon, arm_length_sensitivity, arm_length_lower, arm_length_upper, left+offset_l)
            # We collect the predictions
            l_predicted.append(left + offset_l)
            r_predicted.append(right + offset_r)
            l_predicted_noisy.append(noisy_left)
            r_predicted_noisy.append(noisy_right)
        

        os.chdir("..")

        longer_arm_prediction_accuracy_upper, _ = longer_arm_prediction_accuracy(upper_threshold, l_actual, r_actual, l_predicted_noisy, r_predicted_noisy)
        percent_prediction_on_noisy_upper_threshold.append(longer_arm_prediction_accuracy_upper)
        longer_arm_prediction_accuracy_in_between, _ = longer_arm_prediction_accuracy(lower_threshold, l_actual, r_actual, l_predicted_noisy, r_predicted_noisy)
        percent_prediction_on_noisy_lower_threshold.append(longer_arm_prediction_accuracy_in_between)

        noisy_left_list.append(l_predicted_noisy)
        noisy_right_list.append(r_predicted_noisy)


    # Accuracies without protection
    _, string_output_high_threshold = longer_arm_prediction_accuracy(upper_threshold, l_actual, r_actual, l_predicted, r_predicted)
    _, string_output_low_threshold = longer_arm_prediction_accuracy(lower_threshold, l_actual, r_actual, l_predicted, r_predicted)

    # Accuracies with protection
    str_conf_inter_upper, conf_inter_upper = mean_confidence_interval(percent_prediction_on_noisy_upper_threshold)
    population_mean_of_percent_upper.append(conf_inter_upper[0])
    conf_interval_percent_upper.append(str(conf_inter_upper[1])+', '+str(conf_inter_upper[2]))
    str_conf_inter_lower, conf_inter_lower = mean_confidence_interval(percent_prediction_on_noisy_lower_threshold)
    population_mean_of_percent_lower.append(conf_inter_lower[0])
    conf_interval_percent_lower.append(str(conf_inter_lower[1])+', '+str(conf_inter_lower[2]))

    with open("results/avg_statistics.txt", "a") as f:
        
        print('\nEpsilon={}\n'.format(epsilon), file=f)
        print('\nStatistics with a sample of {} experiments and a confidence interval of 99%.'.format(num_rounds), file=f) 

        print("\nAccuracies without protection:", file=f)
        print(string_output_high_threshold, file=f)
        print(string_output_low_threshold, file=f)
        
        print("\nAccuracies with protection:", file=f)
        print(str_conf_inter_upper, file=f)
        print(str_conf_inter_lower, file=f)
        f.close()

# Save values to txt files
f = open("results/epsilons.txt","w")
f.close()
f = open("results/epsilons.txt","r+")
f.truncate(0)
f.close()
with open('results/epsilons.txt', 'w') as fp:
    for item in epsilon_list:
        # write each item on a new line
        fp.write("%s\n" % item)
        f.close()

## Save values to txt files
f = open("results/population_mean_percent_num_people_high.txt","w")
f.close()
f = open("results/population_mean_percent_num_people_high.txt","r+")
f.truncate(0)
f.close()
with open('results/population_mean_percent_num_people_high.txt', 'w') as fp:
    for item in population_mean_of_percent_lower:
        # write each item on a new line
        fp.write("%s\n" % item)
        f.close()

## Save values to txt files
f = open("results/interval_population_mean_percent_num_people_high.txt","w")
f.close()
f = open("results/interval_population_mean_percent_num_people_high.txt","r+")
f.truncate(0)
f.close()
with open('results/interval_population_mean_percent_num_people_high.txt', 'w') as fp:
    for item in conf_interval_percent_lower:
        # write each item on a new line
        fp.write("%s\n" % item)
        f.close()

## Save values to txt files
f = open("results/population_mean_percent_num_people_low.txt","w")
f.close()
f = open("results/population_mean_percent_num_people_low.txt","r+")
f.truncate(0)
f.close()
with open('results/population_mean_percent_num_people_low.txt', 'w') as fp:
    for item in population_mean_of_percent_upper:
        # write each item on a new line
        fp.write("%s\n" % item)
        f.close()

## Save values to txt files
f = open("results/interval_population_mean_percent_num_people_low.txt","w")
f.close()
f = open("results/interval_population_mean_percent_num_people_low.txt","r+")
f.truncate(0)
f.close()
with open('results/interval_population_mean_percent_num_people_low.txt', 'w') as fp:
    for item in conf_interval_percent_upper:
        # write each item on a new line
        fp.write("%s\n" % item)
        f.close()

#plotting
plot()