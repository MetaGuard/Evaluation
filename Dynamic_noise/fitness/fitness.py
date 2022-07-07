import numpy as np
import glob
import os
import scipy.stats
from DP_guard import Laplace_Bounded_Mechanism
from plotting import plot

# Calculate confidence interval of the population mean without knowing the std of the population
# From: https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data
def mean_confidence_interval(data, confidence=0.99):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return '{} ±{}'.format(m, h), [m, m-h, m+h]

# Heuristic abot the percentage of the height users can go down 
fitness_threshold = 0.25

# Privacy parameters
eps_1 = np.arange(0.01,.1, 0.01)
eps_2 = np.arange(0.1,1, 0.1)
eps_3 = np.arange(1,11, 1)
epsilon_list = np.concatenate([eps_1, eps_2, eps_3])

# Range - we estimate the depth a human can go as a 50% of the height
squat_depth_as_percent_of_height = 0.5
# Height range
height_lower = 1.496 # 5th quantile of adult world population
height_upper = 1.826 # 95th quantile of adult world population
depth_upper = height_upper*(1-squat_depth_as_percent_of_height) # participant is sitting down
depth_lower = 0 # participant cannot go down
sensitivity = abs(height_upper - height_lower)
# sensitivity is the same for the squat depth

# number of experiemnts per epsilon
num_rounds = 30

population_mean_accuracy = []
interval_population_mean_accuracy = []

# Starting and cleaning txt file (if it existed)
f = open("results/avg_statistics.txt","w")
f.close()
f = open("results/avg_statistics.txt","r+")
f.truncate(0)
f.close()

for epsilon in epsilon_list:    
    predicted_on_noisy_master_list = []
    for _ in range(0, num_rounds):
        
        actual = []
        predicted = []
        predicted_on_noisy = []
        os.chdir("data/")
        for name in glob.glob("*.dat"):
            camera_y = []
            with open(name) as file:
                # Read device
                device = file.readline().strip()
                # Read ground truth - 1 is low fitness, 2 is medium, and 3 is high
                actual.append(float(file.readline().strip()))
                # Read signals
                for line in file:
                    camera_y.append(float(line))

            camera_y = camera_y[15000:-15000]
            height = np.percentile(camera_y, 99.5)
            low = np.percentile(camera_y, 0.05)
            depth = height - low
            predicted.append(depth / height)

            # Noise callibration
            noisy_height = Laplace_Bounded_Mechanism(epsilon, sensitivity, height_lower, height_upper, height)
            noisy_depth = Laplace_Bounded_Mechanism(epsilon, sensitivity, depth_lower, depth_upper, depth)
            noisy_multiple = noisy_depth/depth
            # Add noise
            noisy_camera_y = [noisy_height-((height-y)*(noisy_depth)/depth) for y in camera_y]
            pred_height_on_noisy_data = np.percentile(noisy_camera_y, 99.5)
            pred_low_on_noisy_data  = np.percentile(noisy_camera_y, 0.05)
            noisy_depth = pred_height_on_noisy_data - pred_low_on_noisy_data
            predicted_on_noisy.append(noisy_depth/pred_height_on_noisy_data)

        predicted_on_noisy_master_list.append(predicted_on_noisy)
        os.chdir("..")

    with open("results/avg_statistics.txt", "a") as f:

        correct = 0
        incorrect = 0
        print('\nEpsilon={}'.format(epsilon), file=f)

        print('Without protection', file=f)
        for i in range(0,len(actual)):
            # If the participant has "low fitness"
            if (actual[i] == 1):
                # If the participant has a squat depth below the threshold, ie the participant has not gone low enough to show fitness
                if (predicted[i] < fitness_threshold): correct += 1
                else: incorrect += 1
            else:
                if (predicted[i] > fitness_threshold): correct += 1
                else: incorrect += 1

        percent = round((correct / (correct + incorrect))*100,2)
        print("Fitness: " + str(correct) + "/" + str(correct+incorrect) + " (" + str(percent) + "%)", file=f)
        
       
        print('With protection (%)', file=f)
        accuracy_list = []
        for predicted_on_noisy_data in predicted_on_noisy_master_list:
            correct = 0
            incorrect = 0
            for i in range(0,len(actual)):
                # If the participant has "low fitness"
                if (actual[i] == 1):
                    # If the participant has a squat depth below the threshold, ie the participant has not gone low enough to show fitness
                    if (predicted_on_noisy_data[i] < fitness_threshold): correct += 1
                    else: incorrect += 1
                else:
                    if (predicted_on_noisy_data[i] > fitness_threshold): correct += 1
                    else: incorrect += 1

            percent = round((correct / (correct + incorrect))*100,2)
            accuracy_list.append(percent)
        
        noisy_accuracy = mean_confidence_interval(accuracy_list)
        population_mean_accuracy.append(noisy_accuracy[1][0])
        low, up = round(noisy_accuracy[1][1], 3), round(noisy_accuracy[1][2], 3)
        interval_population_mean_accuracy.append(str(low)+', '+str(up))

        print("Fitness: " + str(noisy_accuracy[0]), file=f)
        f.close()


f = open("results/epsilon_list.txt","w")
f.close()
f = open("results/epsilon_list.txt","r+")
f.truncate(0)
f.close()
with open("results/epsilon_list.txt", 'w') as fp:
    for item in epsilon_list:
        # write each item on a new line
        fp.write("%s\n" % item)
        f.close()

f = open("results/population_mean_accuracy.txt","w")
f.close()
f = open("results/population_mean_accuracy.txt","r+")
f.truncate(0)
f.close()
with open("results/population_mean_accuracy.txt", 'w') as fp:
    for item in population_mean_accuracy:
        # write each item on a new line
        fp.write("%s\n" % item)
        f.close()

f = open("results/interval_population_mean_accuracy.txt","w")
f.close()
f = open("results/interval_population_mean_accuracy.txt","r+")
f.truncate(0)
f.close()
with open("results/interval_population_mean_accuracy.txt", 'w') as fp:
    for item in interval_population_mean_accuracy:
        # write each item on a new line
        fp.write("%s\n" % item)
        f.close()

plot()