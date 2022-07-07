import numpy as np
import statistics
import os
import glob
import matplotlib.patches as patches
from DP_guard import Laplace_Bounded_Mechanism

# Stores the ground truth
real_w = []
real_l = []
real_a = []
# Store the predictions on the raw signals
found_w = []
found_l = []
found_a = []
# Stores the device type
device = []
# Stores predictions on noisy signals
noisy_found_w = []
noisy_found_l = []
noisy_found_a = []

# Room size range: https://help.steampowered.com/en/faqs/view/14AE-8D60-24E6-AA67
# https://www.vive.com/us/accessory/base-station2/
# We divide them by two because the coordinate systems is in the middle of the room
# The user can at most travel the diagonal of the room
# the size of a VR room is at least 2mx1.5m (min distance between base stations is 2.5m) and at most 10mx10m
# the max distance between base stations is 10 m. Due to symmetry, we can focus on protecting half the room 
lower = 0 # (m) 
upper = 10/2 # (m) 

# Privacy parameters
epsilon = 5
sensitivity = abs(upper-lower)

# Starting and cleaning txt file (if it existed)
f = open("results/statistics_eps_{}.txt".format(epsilon),"w")
f.close()
f = open("results/statistics_eps_{}.txt".format(epsilon),"r+")
f.truncate(0)
f.close()

# Lists to store data in files for later visualizations
camera_x_list = []
camera_z_list = []
noisy_camera_x_list = []
noisy_camera_z_list = []

# introduce the number of rounds and the different epsilons
noisy_width_R_list = []
noisy_length_R_list = []
noisy_area_R_list = []
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

    camera_x_list.append(camera_x)
    camera_z_list.append(camera_z)
    noisy_camera_x_list.append(noisy_camera_x)
    noisy_camera_z_list.append(noisy_camera_z)
    
os.chdir("..")

# Saving results for plotting
for i, camera_x in enumerate(camera_x_list):
    f = open("results/camera_x_{}_eps_{}.txt".format(i, epsilon),"w")
    f.close()
    f = open("results/camera_x_{}_eps_{}.txt".format(i, epsilon),"r+")
    f.truncate(0)
    f.close()
    with open("results/camera_x_{}_eps_{}.txt".format(i, epsilon), 'w') as fp:
        for item in camera_x:
            # write each item on a new line
            fp.write("%s\n" % item)
            f.close()

for i, camera_z in enumerate(camera_z_list):
    f = open("results/camera_z_{}_eps_{}.txt".format(i, epsilon),"w")
    f.close()
    f = open("results/camera_z_{}_eps_{}.txt".format(i, epsilon),"r+")
    f.truncate(0)
    f.close()
    with open("results/camera_z_{}_eps_{}.txt".format(i, epsilon), 'w') as fp:
        for item in camera_z:
            # write each item on a new line
            fp.write("%s\n" % item)
            f.close()

for i, noisy_camera_x in enumerate(noisy_camera_x_list):
    f = open("results/noisy_camera_x_{}_eps_{}.txt".format(i, epsilon),"w")
    f.close()
    f = open("results/noisy_camera_x_{}_eps_{}.txt".format(i, epsilon),"r+")
    f.truncate(0)
    f.close()
    with open("results/noisy_camera_x_{}_eps_{}.txt".format(i, epsilon), 'w') as fp:
        for item in noisy_camera_x:
            # write each item on a new line
            fp.write("%s\n" % item)
            f.close()

for i, noisy_camera_z in enumerate(noisy_camera_z_list):
    f = open("results/noisy_camera_z_{}_eps_{}.txt".format(i, epsilon),"w")
    f.close()
    f = open("results/noisy_camera_z_{}_eps_{}.txt".format(i, epsilon),"r+")
    f.truncate(0)
    f.close()
    with open("results/noisy_camera_z_{}_eps_{}.txt".format(i, epsilon), 'w') as fp:
        for item in noisy_camera_z:
            # write each item on a new line
            fp.write("%s\n" % item)
            f.close()

