import numpy as np
import matplotlib.pyplot as plt
import os 
import glob
from DP_guard import Laplace_Bounded_Mechanism

# List to store groundtruth
actual = []
# List to store predictions from 
predicted_on_actual = []
predicted_on_noisy_actual = []
# List containing the noise added to each participant
noise_list = []

# Height range
lower = 1.496 # 5th quantile of adult world population
upper = 1.826 # 95th quantile of adult world population

# Privacy parameters
epsilon = 1
sensitivity = round(abs(upper - lower), 2)


f = open("results/statistics.txt","w")
f.close()
f = open("results/statistics.txt","r+")
f.truncate(0)
f.close()

# Data  collection and prediction
os.chdir("data/")
for name in glob.glob("*.dat"):
    # Array to store raw data collected from the VR camera's y-coordinate
    camera_y = []
    with open(name) as file:
        # Get device name from the first line  
        device = file.readline().strip()
        # Get ground truth from the second line 
        ground_truth_height = float(file.readline().strip())
        actual.append(ground_truth_height)
        # Get VR camera's y-coordinate measurements during playthrough
        for line in file:
            camera_y.append(float(line))
    # Select offset depending on device
    if device == 'Oculus Quest 2': offset = 0.12
    if device == 'HTC Vive': offset = 0.11
    if device == 'Vive Pro 2': offset = 0.09
    # Determine height from raw signals
    camera_y = camera_y[15000:-15000]
    camera_y = [y for y in camera_y if y > 1]
    height = np.percentile(camera_y, 99.5)
    # Applying differential privacy to height's clamped raw signals. Fix that noise and add to raw signals
    # Locally, we estimate height as in line 63, and then protect it with clamping and DP.
    noisy_height = round(Laplace_Bounded_Mechanism(epsilon, sensitivity, lower-offset, upper-offset, height), 2)
    noise = noisy_height - height 
    noise_list.append(noise) 
    noisy_camera_y = [y+noise for y in camera_y]
    # Noisy prediction
    noisy_height_prediction = np.percentile(noisy_camera_y, 99.5)
    # Store predictions
    predicted_on_actual.append(height + offset)
    predicted_on_noisy_actual.append(noisy_height_prediction + offset)

os.chdir("..")
with open("results/statistics.txt", "a") as f:
    print('Ground truth range (clamping):', file=f)
    print('lower = {}'.format(lower), file=f)
    print('upper = {}'.format(upper), file=f)
    print('Privacy parameters:', file=f)
    print('epsilon = {}'.format(epsilon), file=f)
    print('sensitivity = {}'.format(sensitivity), file=f)

    # Prediction errors grouped by precision
    errs = np.absolute(np.subtract(predicted_on_actual, actual))
    print('\nError ranges: \nNo protection', file=f)
    print('within 5cm: {}%'.format(round(len([x for x in errs if x <= 0.05])/30*100)), file=f)
    print('within 6cm: {}%'.format(round(len([x for x in errs if x <= 0.06])/30*100)), file=f)
    print('within 7cm: {}%'.format(round(len([x for x in errs if x <= 0.07])/30*100)), file=f)

    # Prediction errors grouped by precision
    errs = np.absolute(np.subtract(predicted_on_noisy_actual, actual))
    print('With protection', file=f)
    print('within 5cm: {}%'.format(round(len([x for x in errs if x <= 0.05])/30*100)), file=f)
    print('within 6cm: {}%'.format(round(len([x for x in errs if x <= 0.06])/30*100)), file=f)
    print('within 7cm: {}%'.format(round(len([x for x in errs if x <= 0.07])/30*100)), file=f)
    print('within 10cm: {}%'.format(round(len([x for x in errs if x <= 0.10])/30*100)), file=f)
    print('within 15cm: {}%'.format(round(len([x for x in errs if x <= 0.15])/30*100)), file=f)
    print('within 20cm: {}%'.format(round(len([x for x in errs if x <= 0.20])/30*100)), file=f)
    print('within 30cm: {}%'.format(round(len([x for x in errs if x <= 0.30])/30*100)), file=f)
    print('within 50cm: {}%'.format(round(len([x for x in errs if x <= 0.50])/30*100)), file=f)

    # Sample statistics
    print('\nSample statistics (m):', file=f)
    print('Mean of actual:', round(np.mean(actual), 3), file=f)
    print('Std of actual:', round(np.std(actual), 3), file=f)
    print('Mean of prediction on actual:', round(np.mean(predicted_on_actual), 3), file=f)
    print('Std of prediction on actual:', round(np.std(predicted_on_actual), 3), file=f)
    print('Relative error of mean prediction on actual: {}%'.format(round(abs(np.mean(predicted_on_actual)-np.mean(actual))/np.mean(actual)*100, 3)), file=f)
    print('Mean of prediction on noisy actual:', round(np.mean(predicted_on_noisy_actual), 3), file=f)
    print('Std of prediction on noisy actual:', round(np.std(predicted_on_noisy_actual), 3), file=f)
    print('Relative error of mean prediction on noisy actual: {}%'.format(round(abs(np.mean(predicted_on_noisy_actual)-np.mean(actual))/np.mean(actual)*100, 3)), file=f)
    print('Absolute error of mean prediction on noisy actual: {}'.format(round(abs(np.mean(predicted_on_noisy_actual)-np.mean(actual)), 2)), file=f)

    # Data min and max
    print('\nMin and max of actual data and predictions (m):', file=f)
    print('Actual data (min, max): ({}, {})'.format(min(actual), max(actual)), file=f)
    print('Prediction on actual data (min, max): ({}, {})'.format(min(predicted_on_actual), max(predicted_on_actual)), file=f)
    print('Prediction on noisy actual data (min, max): ({}, {})'.format(round(min(predicted_on_noisy_actual), 2), round(max(predicted_on_noisy_actual), 2)), file=f)

    # Avergae noise 
    print('\nAverage noise in absolute terms:', round(np.mean(np.abs(noise_list)), 2), file=f)

    # Normalization
    print('\nCoefficients of determination:', file=f)
    actual = np.divide(np.subtract(actual, min(actual)), max(actual) - min(actual))
    predicted_on_actual = np.divide(np.subtract(predicted_on_actual, min(predicted_on_actual)), max(predicted_on_actual) - min(predicted_on_actual))
    predicted_on_noisy_actual = np.divide(np.subtract(predicted_on_noisy_actual, min(predicted_on_noisy_actual)), max(predicted_on_noisy_actual) - min(predicted_on_noisy_actual))

    # Correlation analysis for actual data
    correlation_matrix = np.corrcoef(actual, predicted_on_actual)
    correlation_xy = correlation_matrix[0,1]
    r_squared = correlation_xy**2
    print('R²=' + str(round(r_squared, 3)), file=f)

    # Correlation analysis for noisy data
    correlation_matrix = np.corrcoef(actual, predicted_on_noisy_actual)
    correlation_xy = correlation_matrix[0,1]
    noisy_r_squared = correlation_xy**2
    print('Noisy R²=' + str(round(noisy_r_squared, 3)), file=f)
    f.close()

# Plotting
plt.title('Actual vs. Predicted Height\n(n=' + str(len(actual)) + ', R²=' + str(round(r_squared, 2)) + ', Noisy (\u03B5=' + str(epsilon)+ ') R²=' + str(round(noisy_r_squared, 2)) + ')', fontsize=16)
plt.xlabel('Actual Height (Normalized)', fontsize=14)
plt.ylabel('Predicted Height (Normalized)', fontsize=14)
plt.scatter(actual, predicted_on_actual, s=15, c="b", label='Actual')
plt.scatter(actual, predicted_on_noisy_actual, s=15, c="r", label='Noisy')
z = np.polyfit(actual, predicted_on_actual, 1)
p = np.poly1d(z)
noisy_z = np.polyfit(actual, predicted_on_noisy_actual, 1)
noisy_p = np.poly1d(noisy_z)
plt.plot(actual,p(actual),"b")
plt.plot(actual,noisy_p(actual),"r")
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('figures/2000_height_regression_sample_{}_R2_{}_eps_{}_nR2_{}.pdf'.format(len(actual), round(r_squared, 2), epsilon, round(noisy_r_squared, 2)), format='pdf', bbox_inches='tight')


