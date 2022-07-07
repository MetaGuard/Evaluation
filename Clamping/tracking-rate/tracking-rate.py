import statistics
import glob
import os

device = []
predicted = []
clamped_rate = []
predicted_on_guarded = []

# Clamping value
clamp_value = 90

os.chdir("data/")
for name in glob.glob("*.dat"):
    rate = []
    with open(name) as file:
        # Read the device type
        device.append(file.readline().strip())
        for line in file:
            # Read the refresh rates
            rate.append(float(line))
            clamped_rate.append(clamp_value)
    # Predict refresh rate with the median
    predicted.append(statistics.median(rate))
    # Predict clamped refresh rate with the median
    predicted_on_guarded.append(statistics.median(clamped_rate))
os.chdir("..")

correct = 0
incorrect = 0
for i in range(len(device)):
    err = 3
    if (device[i] == 'HTC Vive'):
        if (abs(predicted[i] - 90) <= err): correct += 1
        else: incorrect += 1
    if (device[i] == 'Vive Pro 2'):
        if (abs(predicted[i] - 120) <= err): correct += 1
        else: incorrect += 1
    if (device[i] == 'Oculus Quest 2'):
        if (abs(predicted[i] - 38) <= err): correct += 1
        else: incorrect += 1

f = open("avg_statistics.txt","w")
f.close()
f = open("avg_statistics.txt","r+")
f.truncate(0)
f.close()
with open("avg_statistics.txt", "a") as f:
    percent = round((correct / (correct + incorrect))*100,2)
    print("Prediction on tracking rate: " + str(correct) + "/" + str(correct+incorrect) + " (" + str(percent) + "%) within 3 Hz", file=f)
    f.close()

correct = 0
incorrect = 0
for i in range(len(device)):
    err = 3
    if (device[i] == 'HTC Vive'):
        if (abs(predicted_on_guarded[i] - 90) <= err): correct += 1
        else: incorrect += 1
    if (device[i] == 'Vive Pro 2'):
        if (abs(predicted_on_guarded[i] - 120) <= err): correct += 1
        else: incorrect += 1
    if (device[i] == 'Oculus Quest 2'):
        if (abs(predicted_on_guarded[i] - 38) <= err): correct += 1
        else: incorrect += 1

with open("avg_statistics.txt", "a") as f:
    percent = round((correct / (correct + incorrect))*100,2)
    print("Prediction on noisy tracking rate: " + str(correct) + "/" + str(correct+incorrect) + " (" + str(percent) + "%) within 3 Hz", file=f)
    f.close()