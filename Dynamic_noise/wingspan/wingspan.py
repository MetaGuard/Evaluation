import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from DP_guard import Laplace_Bounded_Mechanism


# List to store groundtruth
actual = []
# List containing the noise added 
noise_list = []
# List to store predictions of the groundtruth
predicted = []
# List to store predictions of the noisy groundtruth
predicted_on_noisy = []

# Wingspan range. Based on this ref: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7928122/
# Height and wingspan are highly correlated. The coefficient of determination for both male and female is above 0.988 between height and wingspan
# The average ratio (across males and females) is 1.04 (wingspan/height)
ratio = 1.04
height_lower = 1.496 # 5th quantile of adult world population
height_upper = 1.826 # 95th quantile of adult world population
# For DP, we will use these bounds for wingspan
lower = ratio*height_lower
upper = ratio*height_upper

# Privacy parameter
epsilon = 1
sensitivity = np.abs(upper-lower)

# Create folder to store results
f = open("results/statistics.txt","w")
f.close()
f = open("results/statistics.txt","r+")
f.truncate(0)
f.close()

os.chdir("data/")
for name in glob.glob("*.dat"):
    # array to store the distance between controllers
    dist = []
    with open(name) as file:
        # The first line is the VR device type
        device = file.readline().strip()
        # The second line is the wingspan ground truth
        actual.append(float(file.readline().strip()))
        # The rest of lines are the distance between controllers
        for line in file:
            dist.append(float(line))

    # We capture only the values in the middle of the playthrough
    dist = dist[12000:-12000]
    # We discard any potential outliers, during the playthrough, 
    # e.g., putting the controllers down in separate places
    if (max(dist[0:25000]) >= 6): dist = dist[25000:]
    if (max(dist[0:5000]) >= 2.5): dist = dist[5000:]
    while (max(dist) >= 6): dist = dist[:-5000]
    while (max(dist) >= 2): dist = dist[5000:]

    # Depending on the device, the offsets are different
    if device == 'Oculus Quest 2': offset = -0.01
    if device == 'HTC Vive': offset = -0.03
    if device == 'Vive Pro 2': offset = 0.1
    # The prediction is the maximum distance between controllers after removing outliers
    wingspan = max(dist)
    # We apply noise to the wingspan value
    noisy_wingspan = round(Laplace_Bounded_Mechanism(epsilon, sensitivity, lower, upper, wingspan+offset), 2)
    noise = abs(noisy_wingspan-wingspan)
    noise_list.append(noise)
    # We include the prediction in the list
    predicted.append(wingspan + offset)
    # All the distances between the controllers, including the max distance (which is in what the attacker is interested),
    # will be offset in practice in a manner that the retrieved distance after noise "addition" will also be the max among noisy ones
    predicted_on_noisy.append(noisy_wingspan)

os.chdir("..")
with open("results/statistics.txt", "a") as f:
    print('\nError ranges: \nNo protection', file=f)
    errs = np.absolute(np.subtract(predicted, actual))
    print('within 5cm:', len([x for x in errs if x <= 0.05])/30, file=f)
    print('within 7cm:', len([x for x in errs if x <= 0.07])/30, file=f)
    print('within 12cm:', len([x for x in errs if x <= 0.12])/30, file=f)

    print('\nError ranges: \With protection', file=f)
    errs = np.absolute(np.subtract(predicted_on_noisy, actual))
    print('within 5cm:', len([x for x in errs if x <= 0.05])/30, file=f)
    print('within 7cm:', len([x for x in errs if x <= 0.07])/30, file=f)
    print('within 12cm:', len([x for x in errs if x <= 0.12])/30, file=f)
    print('within 20cm:', len([x for x in errs if x <= 0.20])/30, file=f)
    print('within 30cm:', len([x for x in errs if x <= 0.30])/30, file=f)

    print('\nAverage noise in absolute terms:', round(np.mean(np.abs(noise_list)), 2), file=f)

    # Normalization
    print('\nCoefficients of determination:', file=f)
    actual = np.divide(np.subtract(actual, min(actual)), max(actual) - min(actual))
    predicted = np.divide(np.subtract(predicted, min(predicted)), max(predicted) - min(predicted))
    predicted_on_noisy = np.divide(np.subtract(predicted_on_noisy, min(predicted_on_noisy)), max(predicted_on_noisy) - min(predicted_on_noisy))

    # Correlation analysis on actual data
    correlation_matrix = np.corrcoef(actual, predicted)
    correlation_xy = correlation_matrix[0,1]
    r_squared = correlation_xy**2
    print('R²=' + str(round(r_squared, 3)), file=f)

    # Correlation analysis on noisy actual data
    correlation_matrix = np.corrcoef(actual, predicted_on_noisy)
    correlation_xy = correlation_matrix[0,1]
    noisy_r_squared = correlation_xy**2
    print('Noisy R²=' + str(round(noisy_r_squared, 3)), file=f)
    f.close()

# Plotting
plt.title('Actual vs. Predicted Wingspan\n(n=' + str(len(actual)) + ', R²=' + str(round(r_squared, 2)) + ', Noisy (eps=' + str(epsilon)+ ') R²=' + str(round(noisy_r_squared, 2)) + ')', fontsize=16)
plt.xlabel('Actual Wingspan (Norm)', fontsize=14)
plt.ylabel('Predicted Wingspan (Norm)', fontsize=14)
plt.scatter(actual, predicted, s=15, c="b", label='Actual')
plt.scatter(actual, predicted_on_noisy, s=15, c="r", label='Noisy')
z = np.polyfit(actual, predicted, 1)
p = np.poly1d(z)
noisy_z = np.polyfit(actual, predicted_on_noisy, 1)
noisy_p = np.poly1d(noisy_z)
plt.plot(actual,p(actual),"b")
plt.plot(actual,noisy_p(actual),"r")
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('figures/2001_wingspan_regression_{}_R2_{}_eps_{}_nR2_{}.pdf'.format(len(actual), round(r_squared, 2), epsilon, round(noisy_r_squared, 2)), format='pdf', bbox_inches='tight')