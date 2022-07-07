import numpy as np
import matplotlib.pyplot as plt
import statistics
import glob
import os
from DP_guard import Laplace_Bounded_Mechanism

# Interpupillary distance (IPD) range
lower = 55.696 # 5th quantiles IPD https://www.spiedigitallibrary.org/conference-proceedings-of-spie/5291/1/Variation-and-extrema-of-human-interpupillary-distance/10.1117/12.529999.short
upper = 71.024 # 95th quantile IPD

# Privacy parameters
epsilon = 1
sensitivity = abs(upper - lower)

actual = []
predicted = []
noisy_predicted = []
device = []

os.chdir("data/")
for name in glob.glob("*.dat"):
    ipd = []
    with open(name) as file:
        # Read device name
        device.append(file.readline().strip())
        # Read ground truth
        actual.append(float(file.readline().strip()))
        # Read IPD measures
        for line in file:
            ipd_value = float(line)
            ipd.append(ipd_value)
            
    # We predict the IPD with the media (mm)
    ipd_prediction = statistics.median(ipd) * 1000
    predicted.append(ipd_prediction)
    # The method of protection is measuring the IPD from the device and add callibrated noise.
    # This noise will be fixed and will offset the IPD of the user using VR
    # Because the prediction is based on the median, offsetting the list "ipd" and calculating the median would yield
    # the same data point as in "predicted" with an offset. Thus, we consider only offsetting the predicted value
    # In the implementation, the ipd will not be the median, but a snapshot of the IPD value when the protection is activated 
    noisy_predicted.append(Laplace_Bounded_Mechanism(epsilon, sensitivity, lower, upper, ipd_prediction))
os.chdir("..")

# Starting and cleaning txt file (if it existed)
f = open("results/ipd_regression.txt","w")
f.close()
f = open("results/ipd_regression.txt","r+")
f.truncate(0)
f.close()

N = len(actual)

with open("results/ipd_regression.txt", "a") as f:

    print('\nEpsilon={}\n'.format(epsilon), file=f)
    print('Ground truth range (clamping):', file=f)
    print('lower = {}'.format(lower), file=f)
    print('upper = {}'.format(upper), file=f)
    print('Privacy parameters:', file=f)
    print('sensitivity = {}'.format(sensitivity), file=f)

    print('\nError ranges (%): \nNo protection', file=f)
    errs = np.absolute(np.subtract(predicted, actual))
    print('within 0.2mm:', (len([x for x in errs if x <= 0.2])+1)/N, file=f)
    print('within 0.5mm:', (len([x for x in errs if x <= 0.5])+1)/N, file=f)
    print('within 1mm:', (len([x for x in errs if x <= 1])+1)/N, file=f)
    
    print('\nError ranges (%): \nWith protection', file=f)
    errs = np.absolute(np.subtract(noisy_predicted, actual))
    print('within 0.2mm:', (len([x for x in errs if x <= 0.2])+1)/N, file=f)
    print('within 0.5mm:', (len([x for x in errs if x <= 0.5])+1)/N, file=f)
    print('within 1mm:', (len([x for x in errs if x <= 1])+1)/N, file=f)
    print('within 2mm:', (len([x for x in errs if x <= 2])+1)/N, file=f)
    print('within 4mm:', (len([x for x in errs if x <= 4])+1)/N, file=f)
    print('within 8mm:', (len([x for x in errs if x <= 8])+1)/N, file=f)

    # Normalization
    actual = np.divide(np.subtract(actual, min(actual)), max(actual) - min(actual))
    predicted = np.divide(np.subtract(predicted, min(predicted)), max(predicted) - min(predicted))
    noisy_predicted = np.divide(np.subtract(noisy_predicted, min(noisy_predicted)), max(noisy_predicted) - min(noisy_predicted))

    pq = [predicted[i] for i in range(len(actual)) if device[i] == 'Vive Pro 2']
    aq = [actual[i] for i in range(len(actual)) if device[i] == 'Vive Pro 2']
    z = np.polyfit(aq, pq, 1)
    p = np.poly1d(z)

    pq_noisy = [noisy_predicted[i] for i in range(len(actual)) if device[i] == 'Vive Pro 2']
    aq = [actual[i] for i in range(len(actual)) if device[i] == 'Vive Pro 2']
    z = np.polyfit(aq, pq_noisy, 1)
    p_noisy = np.poly1d(z)

    # Correlation analysis
    correlation_matrix = np.corrcoef(actual, predicted)
    correlation_xy = correlation_matrix[0,1]
    r_squared = correlation_xy**2
    print('R²=' + str(r_squared), file=f)

    # Correlation analysis
    correlation_matrix = np.corrcoef(actual, noisy_predicted)
    correlation_xy = correlation_matrix[0,1]
    noisy_r_squared = correlation_xy**2
    print('Noisy R²=' + str(noisy_r_squared), file=f)

# Plotting
plt.title('Actual vs. Predicted IPD\n(n=' + str(len(actual)) + ', R²=' + str(round(r_squared, 2)) + ', Noisy (eps=' + str(epsilon)+ ') R²=' + str(round(noisy_r_squared, 2)) + ')', fontsize=16)
plt.xlabel('Actual IPD (Normalized)', fontsize=14)
plt.ylabel('Predicted IPD (Normalized)', fontsize=14)
for l in ['Vive Pro 2', 'HTC Vive', 'Oculus Quest 2']:
    plt.scatter(
        [actual[i] for i in range(len(actual)) if device[i] == l],
        [predicted[i] for i in range(len(actual)) if device[i] == l],
        label=l
    )
noisy_labels = ['Vive Pro 2 (Noisy)', 'HTC Vive (Noisy)', 'Oculus Quest 2 (Noisy)']
i = 0
for l in ['Vive Pro 2', 'HTC Vive', 'Oculus Quest 2']:
    plt.scatter(
        [actual[i] for i in range(len(actual)) if device[i] == l],
        [noisy_predicted[i] for i in range(len(actual)) if device[i] == l],
        label=noisy_labels[i]
    )
    i += 1

plt.plot(actual,p(actual),"b")
plt.plot(actual, p_noisy(actual), "r")
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('figures/2003_ipd_regression_{}_R2_{}_eps_{}_nR2_{}.pdf'.format(len(actual), round(r_squared, 2), epsilon, round(noisy_r_squared, 2)), format='pdf', bbox_inches='tight')


