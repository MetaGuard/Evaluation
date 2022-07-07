import numpy as np
import statistics
import os
import glob
import matplotlib.patches as patches
from DP_guard import Laplace_Bounded_Mechanism
import scipy.stats


# Calculate confidence interval of the population mean without knowing the std of the population
# From: https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data
def mean_confidence_interval(data, confidence=0.99):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return '{} ±{}'.format(m, h), [m, m-h, m+h]


# Room size range: https://help.steampowered.com/en/faqs/view/14AE-8D60-24E6-AA67
# https://www.vive.com/us/accessory/base-station2/
# We divide them by two because the coordinate systems is in the middle of the room
# The user can at most travel the diagonal of the room
# the size of a VR room is at least 2mx1.5m (min distance between base stations is 2.5m) and at most 10mx10m
# the max distance between base stations is 10 m. Due to the symmetry of the area, we can use half the maximum distance for the upper bound
lower = 0 # (m) 
upper = 10/2 # (m) 

# Privacy parameters
eps_1 = np.arange(0.01,.1, 0.01)
eps_2 = np.arange(0.1,1, 0.1)
eps_3 = np.arange(1,11, 1)
epsilon_list = np.concatenate([eps_1, eps_2, eps_3])
sensitivity = abs(upper-lower)

# number of experiemnts per epsilon
num_rounds = 40

# Starting and cleaning txt file (if it existed)
f = open("results/avg_statistics.txt","w")
f.close()
f = open("results/avg_statistics.txt","r+")
f.truncate(0)
f.close()

# Lists to store data in files for later visualizations
population_mean_noisy_width_R_list = []
interval_population_mean_noisy_width_R_list = []
population_mean_noisy_length_R_list = []
interval_population_mean_noisy_length_R_list= []
population_mean_noisy_area_R_list = []
interval_population_mean_noisy_area_R_list = []
population_mean_noisy_within_1_m = []
interval_population_mean_noisy_within_1_m = []
population_mean_noisy_within_2_m = []
interval_population_mean_noisy_within_2_m = []
population_mean_noisy_within_3_m = []
interval_population_mean_noisy_within_3_m = []

for epsilon in epsilon_list:

    noisy_found_w_master_list = []
    noisy_found_l_master_list = []
    noisy_found_a_master_list = []
    for _ in range(0, num_rounds):

        noisy_found_w = []
        noisy_found_l = []
        noisy_found_a = []
        real_w = []
        real_l = []
        real_a = []
        found_w = []
        found_l = []
        found_a = []
        device = []
        os.chdir("data/")
        for name in glob.glob("*.dat"):
            camera_x = []
            camera_z = []
            with open(name) as file:
                # Reads device type
                device.append(file.readline().strip())
                # Reads ground truth
                rw = float(file.readline().strip())
                rl = float(file.readline().strip())
                real_w.append(rw)
                real_l.append(rl)
                real_a.append(rw*rl)
                # Reads telemetry signals
                for line in file:
                    coords = line.strip().split(', ')
                    camera_x.append(float(coords[0]))
                    camera_z.append(float(coords[1]))
            
            camera_x = camera_x[1200:-2400]
            camera_z = camera_z[1200:-2400]

            # Predictions are percentiles
            left = np.percentile(camera_x, 0)
            right = np.percentile(camera_x, 99.9)
            bottom = np.percentile(camera_z, 0)
            top = np.percentile(camera_z, 99.9)
            width = right - left
            length = top - bottom
            center_x = left + (width/2)
            center_z = bottom + (length/2)
            found_w.append(width)
            found_l.append(length)
            found_a.append(width * length)
            grountruth_collected = True

            # Calculate the callibrated noise
            noisy_w = Laplace_Bounded_Mechanism(epsilon, sensitivity, lower, upper, rw/2)
            noisy_l = Laplace_Bounded_Mechanism(epsilon, sensitivity, lower, upper, rl/2)
            noisy_multiple_x = noisy_w/(rw/2)
            noisy_multiple_z = noisy_l/(rl/2)

            # Apply noise to coordinates
            noisy_camera_x = [center_x+(noisy_multiple_x*(x-center_x)) for x in camera_x]
            noisy_camera_z = [center_z+(noisy_multiple_z*(z-center_z))for z in camera_z]
            # Predictions for noisy data
            noisy_left = np.percentile(noisy_camera_x, 0)
            noisy_right = np.percentile(noisy_camera_x, 99.9)
            noisy_bottom = np.percentile(noisy_camera_z, 0)
            noisy_top = np.percentile(noisy_camera_z, 99.9)
            noisy_width = abs(noisy_right - noisy_left)
            noisy_length = abs(noisy_top - noisy_bottom)
            noisy_center_x = noisy_left + (noisy_width/2)
            noisy_center_z = noisy_bottom + (noisy_length/2)
            noisy_found_w.append(noisy_width)
            noisy_found_l.append(noisy_length)
            noisy_found_a.append(noisy_width * noisy_length)
            
        os.chdir("..")
        # The other lists (real_w, found_w, ...) will be the same in the last roun as they are not stochastic. 
        # Thus, we do not need specific lists for them
        noisy_found_w_master_list.append(noisy_found_w)
        noisy_found_l_master_list.append(noisy_found_l)
        noisy_found_a_master_list.append(noisy_found_a)

    N = len(real_a)

    with open("results/avg_statistics.txt", "a") as f:

        print('\nEpsilon = {}'.format(epsilon), file=f)

        print('\nError ranges for width (%): \nNo protection', file=f)
        errs = np.absolute(np.subtract(real_w, found_w))
        print('Width within 0.5m:', len([x for x in errs if x <= 0.5])/N*100, file=f)
        print('Width within 1m:', len([x for x in errs if x <= 1.0])/N*100, file=f)
        
        print('\nError ranges for length (%): \nNo protection', file=f)
        errs = np.absolute(np.subtract(real_l, found_l))
        print('Length within 0.5m:', len([x for x in errs if x <= 0.5])/N*100, file=f)
        print('Length within 1m:', len([x for x in errs if x <= 1.0])/N*100, file=f)

        print('\nError ranges for room area (%): \nNo protection', file=f)
        errs = np.absolute(np.subtract(real_a, found_a))
        print('Area within 1m²:', len([x for x in errs if x <= 1.0])/N*100, file=f)
        print('Area within 2m²:', len([x for x in errs if x <= 2.0])/N*100, file=f)
        print('Area within 3m²:', len([x for x in errs if x <= 3.0])/N*100, file=f)

        list_0_5_m = []
        list_1_m = []
        for noisy_w in noisy_found_w_master_list:
            errs = np.absolute(np.subtract(real_w, noisy_w))
            list_0_5_m.append(len([x for x in errs if x <= 1.0])/N*100)
            list_1_m.append(len([x for x in errs if x <= 2.0])/N*100)

        print('\nWith protection (%)', file=f)
        noisy_accuracy = mean_confidence_interval(list_0_5_m)
        print('Width accuracy within 0.5m: {}'.format(noisy_accuracy[0]), file=f)

        noisy_accuracy = mean_confidence_interval(list_1_m)
        print('Width accuracy within 1m: {}'.format(noisy_accuracy[0]), file=f)

        list_0_5_m = []
        list_1_m = []
        for noisy_l in noisy_found_l_master_list:
            errs = np.absolute(np.subtract(real_l, noisy_l))
            list_0_5_m.append(len([x for x in errs if x <= 1.0])/N*100)
            list_1_m.append(len([x for x in errs if x <= 2.0])/N*100)

        noisy_accuracy = mean_confidence_interval(list_0_5_m)
        print('Length accuracy within 0.5m: {}'.format(noisy_accuracy[0]), file=f)

        noisy_accuracy = mean_confidence_interval(list_1_m)
        print('Length accuracy within 1m: {}'.format(noisy_accuracy[0]), file=f)

        list_1_m = []
        list_2_m = []
        list_3_m = []
        for noisy_a in noisy_found_a_master_list:
            errs = np.absolute(np.subtract(real_a, noisy_a))
            list_1_m.append(len([x for x in errs if x <= 1.0])/N*100)
            list_2_m.append(len([x for x in errs if x <= 2.0])/N*100)
            list_3_m.append(len([x for x in errs if x <= 3.0])/N*100)

        noisy_accuracy = mean_confidence_interval(list_1_m)
        population_mean_noisy_within_1_m.append(noisy_accuracy[1][0])
        low, up = round(noisy_accuracy[1][1], 3), round(noisy_accuracy[1][2], 3)
        interval_population_mean_noisy_within_1_m.append(str(low)+', '+str(up))
        print('Area accuracy within 1m²: {}'.format(noisy_accuracy[0]), file=f)

        noisy_accuracy = mean_confidence_interval(list_2_m)
        population_mean_noisy_within_2_m.append(noisy_accuracy[1][0])
        low, up = round(noisy_accuracy[1][1], 3), round(noisy_accuracy[1][2], 3)
        interval_population_mean_noisy_within_2_m.append(str(low)+', '+str(up))
        print('Area accuracy within 2m²: {}'.format(noisy_accuracy[0]), file=f)

        noisy_accuracy = mean_confidence_interval(list_3_m)
        population_mean_noisy_within_3_m.append(noisy_accuracy[1][0])
        low, up = round(noisy_accuracy[1][1], 3), round(noisy_accuracy[1][2], 3)
        interval_population_mean_noisy_within_3_m.append(str(low)+', '+str(up))
        print('Area accuracy within 3m²: {}'.format(noisy_accuracy[0]), file=f)
        
        # Ground truth
        real_w = [2.9, 1.6, 2.5]
        real_l = [2.1, 2, 1.5]
        real_a = [6.09, 3.2, 3.75]

        # Without protection
        # These lines aggregate the measurements into the mean - The mean is not the prediction of each individual, we could also just pick one individual
        # Each device was used in one room. Three devices - three rooms. 
        found_w_1 = statistics.mean([found_w[i] for i in range(N) if device[i] == 'Vive Pro 2'])
        found_l_1 = statistics.mean([found_l[i] for i in range(N) if device[i] == 'Vive Pro 2'])
        found_a_1 = statistics.mean([found_a[i] for i in range(N) if device[i] == 'Vive Pro 2'])

        found_w_2 = statistics.mean([found_w[i] for i in range(N) if device[i] == 'HTC Vive'])
        found_l_2 = statistics.mean([found_l[i] for i in range(N) if device[i] == 'HTC Vive'])
        found_a_2 = statistics.mean([found_a[i] for i in range(N) if device[i] == 'HTC Vive'])

        found_w_3 = statistics.mean([found_w[i] for i in range(N) if device[i] == 'Oculus Quest 2'])
        found_l_3 = statistics.mean([found_l[i] for i in range(N) if device[i] == 'Oculus Quest 2'])
        found_a_3 = statistics.mean([found_a[i] for i in range(N) if device[i] == 'Oculus Quest 2'])

        found_w = [found_w_1, found_w_2, found_w_3]
        found_l = [found_l_1, found_l_2, found_l_3]
        found_a = [found_a_1, found_a_2, found_a_3]

        print('\nCoefficients of determinations without protection (per device)', file=f)
        correlation_matrix = np.corrcoef(real_w, found_w)
        correlation_xy = correlation_matrix[0,1]
        w_r_squared = correlation_xy**2
        print('Width R²=' + str(w_r_squared), file=f)

        correlation_matrix = np.corrcoef(real_l, found_l)
        correlation_xy = correlation_matrix[0,1]
        l_r_squared = correlation_xy**2
        print('Length R²=' + str(l_r_squared), file=f)

        correlation_matrix = np.corrcoef(real_a, found_a)
        correlation_xy = correlation_matrix[0,1]
        a_r_squared = correlation_xy**2
        print('Area R²=' + str(a_r_squared), file=f)

        print('\nCoefficients of determinations with protection', file=f)
        noisy_width_R_list = []
        noisy_length_R_list = []
        noisy_area_R_list = []
        
        noisy_found_w_master_list_users = noisy_found_w_master_list
        noisy_found_l_master_list_users = noisy_found_w_master_list
        noisy_found_a_master_list_users = noisy_found_w_master_list

        for j in range(0, len(noisy_found_a_master_list)):

            # With protection
            noisy_found_w_1 = statistics.mean([noisy_found_w_master_list[j][i] for i in range(N) if device[i] == 'Vive Pro 2'])
            noisy_found_l_1 = statistics.mean([noisy_found_l_master_list[j][i] for i in range(N) if device[i] == 'Vive Pro 2'])
            noisy_found_a_1 = statistics.mean([noisy_found_a_master_list[j][i] for i in range(N) if device[i] == 'Vive Pro 2'])

            noisy_found_w_2 = statistics.mean([noisy_found_w_master_list[j][i] for i in range(N) if device[i] == 'HTC Vive'])
            noisy_found_l_2 = statistics.mean([noisy_found_l_master_list[j][i] for i in range(N) if device[i] == 'HTC Vive'])
            noisy_found_a_2 = statistics.mean([noisy_found_a_master_list[j][i] for i in range(N) if device[i] == 'HTC Vive'])

            noisy_found_w_3 = statistics.mean([noisy_found_w_master_list[j][i] for i in range(N) if device[i] == 'Oculus Quest 2'])
            noisy_found_l_3 = statistics.mean([noisy_found_l_master_list[j][i] for i in range(N) if device[i] == 'Oculus Quest 2'])
            noisy_found_a_3 = statistics.mean([noisy_found_a_master_list[j][i] for i in range(N) if device[i] == 'Oculus Quest 2'])
        
            noisy_found_w = [noisy_found_w_1, noisy_found_w_2, noisy_found_w_3]
            noisy_found_l = [noisy_found_l_1, noisy_found_l_2, noisy_found_l_3]
            noisy_found_a = [noisy_found_a_1, noisy_found_a_2, noisy_found_a_3]

            correlation_matrix = np.corrcoef(real_w, noisy_found_w)
            correlation_xy = correlation_matrix[0,1]
            noisy_w_r_squared = correlation_xy**2
            noisy_width_R_list.append(noisy_w_r_squared)

            correlation_matrix = np.corrcoef(real_l, noisy_found_l)
            correlation_xy = correlation_matrix[0,1]
            noisy_l_r_squared = correlation_xy**2
            noisy_length_R_list.append(noisy_l_r_squared)

            correlation_matrix = np.corrcoef(real_a, noisy_found_a)
            correlation_xy = correlation_matrix[0,1]
            noisy_a_r_squared = correlation_xy**2
            noisy_area_R_list.append(noisy_a_r_squared)

        # Width
        noisy_R_2 = mean_confidence_interval(noisy_width_R_list)
        population_mean_noisy_width_R_list.append(noisy_R_2[1][0])
        low, up = round(noisy_R_2[1][1], 3), round(noisy_R_2[1][2], 3)
        interval_population_mean_noisy_width_R_list.append(str(low)+', '+str(up))
        print('Noisy width R²:' + str(noisy_R_2[0]), file=f)

        # Length
        noisy_R_2 = mean_confidence_interval(noisy_length_R_list)
        population_mean_noisy_length_R_list.append(noisy_R_2[1][0])
        low, up = round(noisy_R_2[1][1], 3), round(noisy_R_2[1][2], 3)
        interval_population_mean_noisy_length_R_list.append(str(low)+', '+str(up))
        print('Noisy length R²:' + str(noisy_R_2[0]), file=f)

        # Area
        noisy_R_2 = mean_confidence_interval(noisy_area_R_list)
        population_mean_noisy_area_R_list.append(noisy_R_2[1][0])
        low, up = round(noisy_R_2[1][1], 3), round(noisy_R_2[1][2], 3)
        interval_population_mean_noisy_area_R_list.append(str(low)+', '+str(up))
        print('Noisy area R²:' + str(noisy_R_2[0]), file=f)       
  
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

f = open("results/population_mean_noisy_width_R_list.txt","w")
f.close()
f = open("results/population_mean_noisy_width_R_list.txt","r+")
f.truncate(0)
f.close()
with open("results/population_mean_noisy_width_R_list.txt", 'w') as fp:
    for item in population_mean_noisy_width_R_list:
        # write each item on a new line
        fp.write("%s\n" % item)
        f.close()

f = open("results/interval_population_mean_noisy_width_R_list.txt","w")
f.close()
f = open("results/interval_population_mean_noisy_width_R_list.txt","r+")
f.truncate(0)
f.close()
with open("results/interval_population_mean_noisy_width_R_list.txt", 'w') as fp:
    for item in interval_population_mean_noisy_width_R_list:
        # write each item on a new line
        fp.write("%s\n" % item)
        f.close()

f = open("results/population_mean_noisy_length_R_list.txt","w")
f.close()
f = open("results/population_mean_noisy_length_R_list.txt","r+")
f.truncate(0)
f.close()
with open("results/population_mean_noisy_length_R_list.txt", 'w') as fp:
    for item in population_mean_noisy_length_R_list:
        # write each item on a new line
        fp.write("%s\n" % item)
        f.close()

f = open("results/interval_population_mean_noisy_length_R_list.txt","w")
f.close()
f = open("results/interval_population_mean_noisy_length_R_list.txt","r+")
f.truncate(0)
f.close()
with open("results/interval_population_mean_noisy_length_R_list.txt", 'w') as fp:
    for item in interval_population_mean_noisy_length_R_list:
        # write each item on a new line
        fp.write("%s\n" % item)
        f.close()

f = open("results/population_mean_noisy_area_R_list.txt","w")
f.close()
f = open("results/population_mean_noisy_area_R_list.txt","r+")
f.truncate(0)
f.close()
with open("results/population_mean_noisy_area_R_list.txt", 'w') as fp:
    for item in population_mean_noisy_area_R_list:
        # write each item on a new line
        fp.write("%s\n" % item)
        f.close()

f = open("results/interval_population_mean_noisy_area_R_list.txt","w")
f.close()
f = open("results/interval_population_mean_noisy_area_R_list.txt","r+")
f.truncate(0)
f.close()
with open("results/interval_population_mean_noisy_area_R_list.txt", 'w') as fp:
    for item in interval_population_mean_noisy_area_R_list:
        # write each item on a new line
        fp.write("%s\n" % item)
        f.close()


f = open("results/population_mean_noisy_within_1_m.txt","w")
f.close()
f = open("results/population_mean_noisy_within_1_m.txt","r+")
f.truncate(0)
f.close()
with open("results/population_mean_noisy_within_1_m.txt", 'w') as fp:
    for item in population_mean_noisy_within_1_m:
        # write each item on a new line
        fp.write("%s\n" % item)
        f.close()

f = open("results/interval_population_mean_noisy_within_1_m.txt","w")
f.close()
f = open("results/interval_population_mean_noisy_within_1_m.txt","r+")
f.truncate(0)
f.close()
with open("results/interval_population_mean_noisy_within_1_m.txt", 'w') as fp:
    for item in interval_population_mean_noisy_within_1_m:
        # write each item on a new line
        fp.write("%s\n" % item)
        f.close()

f = open("results/population_mean_noisy_within_2_m.txt","w")
f.close()
f = open("results/population_mean_noisy_within_2_m.txt","r+")
f.truncate(0)
f.close()
with open("results/population_mean_noisy_within_2_m.txt", 'w') as fp:
    for item in population_mean_noisy_within_2_m:
        # write each item on a new line
        fp.write("%s\n" % item)
        f.close()

f = open("results/interval_population_mean_noisy_within_2_m.txt","w")
f.close()
f = open("results/interval_population_mean_noisy_within_2_m.txt","r+")
f.truncate(0)
f.close()
with open("results/interval_population_mean_noisy_within_2_m.txt", 'w') as fp:
    for item in interval_population_mean_noisy_within_2_m:
        # write each item on a new line
        fp.write("%s\n" % item)
        f.close()

f = open("results/population_mean_noisy_within_3_m.txt","w")
f.close()
f = open("results/population_mean_noisy_within_3_m.txt","r+")
f.truncate(0)
f.close()
with open("results/population_mean_noisy_within_3_m.txt", 'w') as fp:
    for item in population_mean_noisy_within_3_m:
        # write each item on a new line
        fp.write("%s\n" % item)
        f.close()

f = open("results/interval_population_mean_noisy_within_3_m.txt","w")
f.close()
f = open("results/interval_population_mean_noisy_within_3_m.txt","r+")
f.truncate(0)
f.close()
with open("results/interval_population_mean_noisy_within_3_m.txt", 'w') as fp:
    for item in interval_population_mean_noisy_within_3_m:
        # write each item on a new line
        fp.write("%s\n" % item)
        f.close()