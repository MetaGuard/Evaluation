import numpy as np
import scipy.stats
import glob
import os 
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

# Height range
lower = 1.496 # 5th quantile of adult world population
upper = 1.826 # 95th quantile of adult world population

# Privacy parameters
eps_1 = np.arange(0.01,.1, 0.01)
eps_2 = np.arange(0.1,1, 0.1)
eps_3 = np.arange(1,11, 1)
epsilon_list = np.concatenate([eps_1, eps_2, eps_3])
sensitivity = round(abs(upper - lower), 2)

# Number of iterations to calculate average results
num_rounds = 100

# Store noisy R^2 and error statistics
noisy_mean_R_2_list = []
noisy_interval_R_2_list = []
mean_relative_error_master_list = []
interval_mean_relative_error_master_list = []
mean_absolute_error_master_list = []
interval_mean_absolute_error_master_list = []
std_absolute_error_master_list = []
interval_std_absolute_error_master_list = []
mean_pred_noisy_actual_master_list = []
interval_mean_pred_noisy_actual_master_list = []

# Starting and cleaning txt file (if it existed)
f = open("results/avg_statistics.txt","w")
f.close()
f = open("results/avg_statistics.txt","r+")
f.truncate(0)
f.close()

for epsilon in epsilon_list:

    # Array to store arrays of predictions on noisy data
    predicted_on_noisy_actual_list = []
    avg_noise_list = []

    for _ in range(0, num_rounds):

        # Data  collection and prediction
        # Array to store predictions
        predicted_on_actual = []
        predicted_on_noisy_actual_list_temp = []
        # Array to store groundtruth
        actual = []
        noise_list = []

        # We move to the data folder to get the data
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
            predicted_on_noisy_actual_list_temp.append(noisy_height_prediction + offset)

        avg_noise_list.append(np.mean(np.abs(noise_list)))

        predicted_on_noisy_actual_list.append(predicted_on_noisy_actual_list_temp)
    
        # Move one folder back
        os.chdir("..")

    with open("results/avg_statistics.txt", "a") as f:

        print('Epsilon={}\n'.format(epsilon), file=f)
        print('Height ground truth range (clamping):', file=f)
        print('lower = {}'.format(lower), file=f)
        print('upper = {}'.format(upper), file=f)
        print('Privacy parameters:', file=f)
        print('epsilon = {}'.format(epsilon), file=f)
        print('sensitivity = {}'.format(sensitivity), file=f)
        print('\nStatistics with a sample of {} experiments and a confidence interval of 99%.'.format(num_rounds), file=f)

        # Prediction errors grouped by precision
        errs = np.absolute(np.subtract(predicted_on_actual, actual))
        print('\nError ranges (%): \nNo protection', file=f)
        print('within 5cm: {}'.format(round(len([x for x in errs if x <= 0.05])/30*100)), file=f)
        print('within 6cm: {}'.format(round(len([x for x in errs if x <= 0.06])/30*100)), file=f)
        print('within 7cm: {}'.format(round(len([x for x in errs if x <= 0.07])/30*100)), file=f)

        list_5_cm = []
        list_6_cm = []
        list_7_cm = []
        list_10_cm = []
        list_15_cm = []
        list_20_cm = []
        list_30_cm = []
        list_50_cm = []
        for predicted_on_noisy_actual in predicted_on_noisy_actual_list:

            # Prediction errors grouped by precision
            errs = np.absolute(np.subtract(predicted_on_noisy_actual, actual))
            list_5_cm.append(len([x for x in errs if x <= 0.05])/30*100)
            list_6_cm.append(len([x for x in errs if x <= 0.06])/30*100)
            list_7_cm.append(len([x for x in errs if x <= 0.07])/30*100)
            list_10_cm.append(len([x for x in errs if x <= 0.10])/30*100)
            list_15_cm.append(len([x for x in errs if x <= 0.15])/30*100)
            list_20_cm.append(len([x for x in errs if x <= 0.20])/30*100)
            list_30_cm.append(len([x for x in errs if x <= 0.30])/30*100)
            list_50_cm.append(len([x for x in errs if x <= 0.50])/30*100)

        print('With Protection (%)', file=f)
        print('within 5cm: {}'.format(mean_confidence_interval(list_5_cm)[0]), file=f)
        print('within 6cm: {}'.format(mean_confidence_interval(list_6_cm)[0]), file=f)
        print('within 7cm: {}'.format(mean_confidence_interval(list_7_cm)[0]), file=f)
        print('within 10cm: {}'.format(mean_confidence_interval(list_10_cm)[0]), file=f)
        print('within 15cm: {}'.format(mean_confidence_interval(list_15_cm)[0]), file=f)
        print('within 20cm: {}'.format(mean_confidence_interval(list_20_cm)[0]), file=f)
        print('within 30cm: {}'.format(mean_confidence_interval(list_30_cm)[0]), file=f)
        print('within 50cm: {}'.format(mean_confidence_interval(list_50_cm)[0]), file=f)

        # Data min and max
        min_list = []
        max_list = []
        for predicted_on_noisy_actual in predicted_on_noisy_actual_list:
            min_list.append(round(min(predicted_on_noisy_actual), 2))
            max_list.append(round(max(predicted_on_noisy_actual), 2))

        print('\nMin and max of actual data and predictions (m):', file=f)
        print('Actual data (min, max): ({}, {})'.format(min(actual), max(actual)), file=f)
        print('Prediction on actual data (min, max): ({}, {})'.format(min(predicted_on_actual), max(predicted_on_actual)), file=f)
        print('Prediction on noisy actual data: \nMin: {}. \nMax: {}.'.format(mean_confidence_interval(min_list)[0], mean_confidence_interval(max_list)[0]), file=f)

        # Avergae noise 
        print('\nAverage noise in absolute terms:', mean_confidence_interval(avg_noise_list)[0], file=f)

        mean_abs_error_list = []
        std_abs_error_list = []
        mean_rel_error_list = []
        mean_prediction_on_noisy_actual_list = []
        for predicted_on_noisy_actual in predicted_on_noisy_actual_list:
            
            mean_prediction_on_noisy_actual_list.append(np.mean(predicted_on_noisy_actual))

            abs_errs = np.absolute(np.subtract(predicted_on_noisy_actual, actual))
            rel_errs = np.divide(abs_errs, actual)

            mean_abs_error = np.mean(abs_errs)
            std_abs_error = np.std(abs_errs)
            mean_rel_error = np.mean(rel_errs)

            mean_abs_error_list.append(mean_abs_error)
            std_abs_error_list.append(std_abs_error)
            mean_rel_error_list.append(mean_rel_error)

            
       # Sample statistics
        print('\nSample statistics (m):', file=f)
        print('Mean of actual:', round(np.mean(actual), 3), file=f)
        print('Std of actual:', round(np.std(actual), 3), file=f)
        print('Mean of prediction on actual:', round(np.mean(predicted_on_actual), 3), file=f)
        print('Std of prediction on actual:', round(np.std(predicted_on_actual), 3), file=f)
        # The next two lines consider the error 1:1 between noisy and actual data point. Not the difference between the means.
        print('Mean absolute error of prediction on actual: {}'.format(round(np.mean(np.absolute(np.subtract(predicted_on_actual, actual))), 2)), file=f)
        print('Mean relative error of prediction on actual: {}%'.format(round(np.mean(np.divide(np.absolute(np.subtract(predicted_on_actual, actual)), actual))*100, 2)), file=f)
        # Predictions on noisy actual
        conf_int_mean_pred_on_noisy_actual = mean_confidence_interval(mean_prediction_on_noisy_actual_list)
        mean_pred_noisy_actual_master_list.append(conf_int_mean_pred_on_noisy_actual[1][0])
        interval_mean_pred_noisy_actual_master_list.append(str(conf_int_mean_pred_on_noisy_actual[1][1])+', '+str(conf_int_mean_pred_on_noisy_actual[1][2]))
        print('Population mean of prediction on noisy actual:',conf_int_mean_pred_on_noisy_actual[0], file=f)     
        # abs err
        conf_int_abs_err_mean = mean_confidence_interval(mean_abs_error_list)
        mean_absolute_error_master_list.append(conf_int_abs_err_mean[1][0])
        interval_mean_absolute_error_master_list.append(str(conf_int_abs_err_mean[1][1])+', '+str(conf_int_abs_err_mean[1][2]))
        print('Population mean of abs errors of prediction on noisy actual:',conf_int_abs_err_mean[0], file=f)
        # std of abs err
        conf_int_abs_err_std = mean_confidence_interval(std_abs_error_list)
        std_absolute_error_master_list.append(conf_int_abs_err_std[1][0])
        interval_std_absolute_error_master_list.append(str(conf_int_abs_err_std[1][1])+', '+ str(conf_int_abs_err_std[1][2]))
        print('Population std of abs errors of prediction on noisy actual:', conf_int_abs_err_std[0], file=f)
        # rel err
        conf_int_rel_err_mean = mean_confidence_interval(mean_rel_error_list)
        mean_relative_error_master_list.append(conf_int_rel_err_mean[1][0])
        interval_mean_relative_error_master_list.append(str(conf_int_rel_err_mean[1][1])+', '+str(conf_int_rel_err_mean[1][2]))   
        print('Population mean of relative errors of prediction on noisy actual:', conf_int_rel_err_mean[0], file=f)
   
        # Normalization
        print('\nCoefficients of determination:', file=f)
        actual = np.divide(np.subtract(actual, min(actual)), max(actual) - min(actual))
        
        # Correlation analysis for actual data
        predicted_on_actual = np.divide(np.subtract(predicted_on_actual, min(predicted_on_actual)), max(predicted_on_actual) - min(predicted_on_actual))
        correlation_matrix = np.corrcoef(actual, predicted_on_actual)
        correlation_xy = correlation_matrix[0,1]
        r_squared = correlation_xy**2

        noisy_coef_determination_list = []
        for predicted_on_noisy_actual in predicted_on_noisy_actual_list:
            predicted_on_noisy_actual = np.divide(np.subtract(predicted_on_noisy_actual, min(predicted_on_noisy_actual)), max(predicted_on_noisy_actual) - min(predicted_on_noisy_actual))

            # Correlation analysis for noisy data
            correlation_matrix = np.corrcoef(actual, predicted_on_noisy_actual)
            correlation_xy = correlation_matrix[0,1]
            noisy_r_squared = correlation_xy**2
            noisy_coef_determination_list.append(noisy_r_squared)
        
        noisy_R_2 = mean_confidence_interval(noisy_coef_determination_list)
        noisy_mean_R_2_list.append(noisy_R_2[1][0])
        low, up = round(noisy_R_2[1][1], 3), round(noisy_R_2[1][2], 3)
        noisy_interval_R_2_list.append(str(low)+', '+str(up))
        print('R²=' + str(round(r_squared, 3)), file=f)
        print('Noisy R²:' + str(noisy_R_2[0])+'\n', file=f)
        f.close()

## Save values to txt files
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

f = open("results/mean_pred_on_noisy_actual.txt","w")
f.close()
f = open("results/mean_pred_on_noisy_actual.txt","r+")
f.truncate(0)
f.close()
with open('results/mean_pred_on_noisy_actual.txt', 'w') as fp:
    for item in mean_pred_noisy_actual_master_list:
        # write each item on a new line
        fp.write("%s\n" % item)
        f.close()

f = open("results/interval_mean_pred_on_noisy_actual.txt","w")
f.close()
f = open("results/interval_mean_pred_on_noisy_actual.txt","r+")
f.truncate(0)
f.close()
with open('results/interval_mean_pred_on_noisy_actual.txt', 'w') as fp:
    for item in interval_mean_pred_noisy_actual_master_list:
        # write each item on a new line
        fp.write("%s\n" % item)
        f.close()

f = open("results/relative_error.txt","w")
f.close()
f = open("results/relative_error.txt","r+")
f.truncate(0)
f.close()
with open('results/relative_error.txt', 'w') as fp:
    for item in mean_absolute_error_master_list:
        # write each item on a new line
        fp.write("%s\n" % item)
        f.close()

f = open("results/absolute_error.txt","w")
f.close()
f = open("results/absolute_error.txt","r+")
f.truncate(0)
f.close()
with open('results/absolute_error.txt', 'w') as fp:
    for item in mean_absolute_error_master_list:
        # write each item on a new line
        fp.write("%s\n" % item)
        f.close()

f = open("results/noisy_mean_R_2_values.txt","w")
f.close()
f = open("results/noisy_mean_R_2_values.txt","r+")
f.truncate(0)
f.close()
with open('results/noisy_mean_R_2_values.txt', 'w') as fp:
    for item in noisy_mean_R_2_list:
        # write each item on a new line
        fp.write("%s\n" % item)
        f.close()

f = open("results/noisy_interval_R_2_values.txt","w")
f.close()
f = open("results/noisy_interval_R_2_values.txt","r+")
f.truncate(0)
f.close()
with open('results/noisy_interval_R_2_values.txt', 'w') as fp:
    for item in noisy_interval_R_2_list:
        # write each item on a new line
        fp.write("%s\n" % item)
        f.close()

f = open("results/interval_mean_relative_error.txt","w")
f.close()
f = open("results/interval_mean_relative_error.txt","r+")
f.truncate(0)
f.close()
with open('results/interval_mean_relative_error.txt', 'w') as fp:
    for item in interval_mean_relative_error_master_list:
        # write each item on a new line
        fp.write("%s\n" % item)
        f.close()

f = open("results/interval_mean_absolute_error.txt","w")
f.close()
f = open("results/interval_mean_absolute_error.txt","r+")
f.truncate(0)
f.close()
with open('results/interval_mean_absolute_error.txt', 'w') as fp:
    for item in interval_mean_absolute_error_master_list:
        # write each item on a new line
        fp.write("%s\n" % item)
        f.close()
        
f = open("results/std_absolute_error.txt","w")
f.close()
f = open("results/std_absolute_error.txt","r+")
f.truncate(0)
f.close()
with open('results/std_absolute_error.txt', 'w') as fp:
    for item in std_absolute_error_master_list:
        # write each item on a new line
        fp.write("%s\n" % item)
        f.close()

f = open("results/interval_std_absolute_error.txt","w")
f.close()
f = open("results/interval_std_absolute_error.txt","r+")
f.truncate(0)
f.close()
with open('results/interval_std_absolute_error.txt', 'w') as fp:
    for item in interval_std_absolute_error_master_list:
        # write each item on a new line
        fp.write("%s\n" % item)
        f.close()

# plotting
plot()
