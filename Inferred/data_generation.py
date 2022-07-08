# Imports
import csv
import os
import random


# Source: https://diffprivlib.readthedocs.io/en/latest/modules/mechanisms.html?highlight=bounded#diffprivlib.mechanisms.LaplaceBoundedDomain
from diffprivlib.mechanisms.laplace import LaplaceBoundedDomain

def Laplace_Bounded_Mechanism(epsilon, sensitivity, lower, upper, deterministic_value):
    
    # Initialize differential privacy mechanism
    LapBoundMech = LaplaceBoundedDomain(epsilon=epsilon, sensitivity=sensitivity, lower=lower, upper=upper)

    # Produce differentially private output
    noisy_output = LapBoundMech.randomise(deterministic_value)
    return noisy_output

if __name__ == '__main__':

    # Starting and cleaning txt file (if it existed)
    f = open("noisy_data/noisy_gender.csv","w")
    f.close()
    f = open("noisy_data/noisy_gender.csv","r+")
    f.truncate(0)
    f.close()

    f = open("noisy_data/noisy_age.csv","w")
    f.close()
    f = open("noisy_data/noisy_age.csv","r+")
    f.truncate(0)
    f.close()

    f = open("noisy_data/noisy_ethnicity.csv","w")
    f.close()
    f = open("noisy_data/noisy_ethnicity.csv","r+")
    f.truncate(0)
    f.close()

    f = open("noisy_data/noisy_ID.csv","w")
    f.close()
    f = open("noisy_data/noisy_ID.csv","r+")
    f.truncate(0)
    f.close()

    # Epsilons 
    privacy_levels = 3
    height_epsilons = [1, 3, 5]
    wingspan_epsilons = [0.5, 1, 3]
    ipd_epsilons = [1, 3, 5]
    voice_coin_flip = [0.6125, 0.65, 0.725]
    handedness_RR_coin_flip = [0.05, 0.45, 0.63]
    arm_epsilons = [0.1, 1, 3]
    reaction_time_offset = [100, 20, 10]
    
    # Physical parameters
    height_to_wingspan_ratio = 1.04
    # Bounds
    height_lower = 1.496 
    height_upper = 1.826 
    height_sensitivity = abs(height_upper-height_lower)
    wingspan_lower = height_lower*height_to_wingspan_ratio
    wingspan_upper = height_upper*height_to_wingspan_ratio
    wingspan_sensitivity = abs(wingspan_upper-wingspan_lower)
    arm_ratio_lower = 0.95
    arm_ratio_upper = 1.05
    arm_ratio_sensitivity = abs(arm_ratio_upper-arm_ratio_lower)
    ipd_lower = 55.696
    ipd_upper = 71.024
    ipd_sensitivity = abs(ipd_upper-ipd_lower)

    # Privacy parameters
    num_rounds = 100

    # labels
    labels_gender = ["id", "gender", "voice", "height", "wingspan", "ipd"]
    file_labels = ["id", "gender", "voice", "height", "wingspan", "ipd"]
	
    # CSVs
    gender_csv_header = ["id", "round", "privacy level", "gender", "voice", "height", "wingspan", "ipd"]
    age_csv_header = ["id", "round", "privacy level","age", "close", "reaction", "height", "duration", "moca"]
    ethnicity_csv_header = ["id", "round", "privacy level","ethnicity", "voice", "language1", "language2", "height"]
    ID_csv_header = ["id", "round", "privacy level", "height", "wingspan", "left", "right", "hand", "ipd"]

# ID CSV
    rows = []
    os.chdir("data/")
    with open('MetaGuard Derived - ID.csv', 'r') as f:
        my_reader = csv.reader(f, delimiter=',')
        next(f)
        for row in my_reader:
            row.insert(1, 0)
            row.insert(2, 0)
            rows.append(row)
    f.close()
    os.chdir("..")

    with open('noisy_data/noisy_ID.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(ID_csv_header)
        for row in rows:
            writer.writerow(row)
        f.close()	


    rows = []
    for i in range(privacy_levels):

        for j in range(num_rounds):
            
            os.chdir("data/")
            with open('MetaGuard Derived - ID.csv', 'r') as f:

                my_reader = csv.reader(f, delimiter=',')
                next(f)
                for line in my_reader:
                    row = []
                    row.append(line[0]) # append ID
                    row.append(j+1) # append experiment round
                    row.append(i+1) # append privacy level
                    # append noisy height
                    row.append(round(100*Laplace_Bounded_Mechanism(height_epsilons[i], height_sensitivity, height_lower, height_upper, float(line[1])), 1))
                    # append wingspan
                    wingspan = float(line[2])  
                    noisy_wingspan = round(100*Laplace_Bounded_Mechanism(wingspan_epsilons[i], wingspan_sensitivity, wingspan_lower, wingspan_upper, wingspan), 1)
                    row.append(noisy_wingspan)
                    # append left and right arm
                    arm_ratio = float(line[3]) / float(line[4])
                    noisy_arm_ratio = Laplace_Bounded_Mechanism(arm_epsilons[i], arm_ratio_sensitivity, arm_ratio_lower, arm_ratio_upper, arm_ratio)
                    row.append(round(noisy_wingspan/2*noisy_arm_ratio, 1))
                    row.append(round(noisy_wingspan/2*(1/noisy_arm_ratio), 1))
                    # append handedness
                    if random.uniform(0, 1) < handedness_RR_coin_flip[i]:
                        row.append(line[5])
                    elif random.uniform(0, 1) < handedness_RR_coin_flip[i]:
                        row.append('right')
                    else:
                        row.append('left')
                    # append ipd
                    row.append(round(Laplace_Bounded_Mechanism(ipd_epsilons[i], ipd_sensitivity, ipd_lower, ipd_upper, float(line[6])), 1))

                    rows.append(row)

            f.close()
            os.chdir("..")

    with open('noisy_data/noisy_ID.csv', 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)
        f.close()	

   # Ethnicity CSV
    rows = []
    os.chdir("data/")
    with open('MetaGuard Derived - Ethnicity.csv', 'r') as f:
        my_reader = csv.reader(f, delimiter=',')
        next(f)
        for row in my_reader:
            row.insert(1, 0)
            row.insert(2, 0)
            rows.append(row)
    f.close()
    os.chdir("..")

    with open('noisy_data/noisy_ethnicity.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(ethnicity_csv_header)
        for row in rows:
            writer.writerow(row)
        f.close()	


    rows = []
    for i in range(privacy_levels):

        for j in range(num_rounds):
            
            os.chdir("data/")
            with open('MetaGuard Derived - Ethnicity.csv', 'r') as f:

                my_reader = csv.reader(f, delimiter=',')
                next(f)
                for line in my_reader:
                    row = []
                    row.append(line[0]) # append ID
                    row.append(j+1) # append experiment round
                    row.append(i+1) # append privacy level
                    row.append(line[1]) # append ground truth
                    row.append(line[2]) # append voice-ethnicity
                    row.append(line[3]) # append language 1
                    row.append(line[4]) # append language 2
                    # append noisy height
                    row.append(round(100*Laplace_Bounded_Mechanism(height_epsilons[i], height_sensitivity, height_lower, height_upper, float(line[5])), 1))
                    rows.append(row)

            f.close()
            os.chdir("..")

    with open('noisy_data/noisy_ethnicity.csv', 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)
        f.close()	 


    # Age CSV
    rows = []
    os.chdir("data/")
    with open('MetaGuard Derived - Age.csv', 'r') as f:
        my_reader = csv.reader(f, delimiter=',')
        next(f)
        for row in my_reader:
            row.insert(1, 0)
            row.insert(2, 0)
            rows.append(row)
    f.close()
    os.chdir("..")

    with open('noisy_data/noisy_age.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(age_csv_header)
        for row in rows:
            writer.writerow(row)
        f.close()	


    rows = []
    for i in range(privacy_levels):

        for j in range(num_rounds):
            
            os.chdir("data/")
            with open('MetaGuard Derived - Age.csv', 'r') as f:

                my_reader = csv.reader(f, delimiter=',')
                next(f)
                for line in my_reader:
                    row = []
                    row.append(line[0]) # append ID
                    row.append(j+1) # append experiment round
                    row.append(i+1) # append privacy level
                    row.append(line[1]) # append ground truth
                    row.append(line[2]) # append close vision
                    # append noisy reaction time
                    row.append(float(line[3]) + round(random.uniform(0, 1)*reaction_time_offset[i] -reaction_time_offset[i]/2)) 
                    # append noisy height
                    row.append(round(100*Laplace_Bounded_Mechanism(height_epsilons[i], height_sensitivity, height_lower, height_upper, float(line[4])), 1))
                    row.append(line[5]) # append duraton
                    row.append(line[6]) # append moca
                    rows.append(row)

            f.close()
            os.chdir("..")

    with open('noisy_data/noisy_age.csv', 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)
        f.close()	        


#    Gender CSV
    rows = []
    os.chdir("data/")
    with open('MetaGuard Derived - Gender.csv', 'r') as f:
        my_reader = csv.reader(f, delimiter=',')
        next(f)
        for row in my_reader:
            row.insert(1, 0)
            row.insert(2, 0)
            rows.append(row)
    f.close()
    os.chdir("..")

    with open('noisy_data/noisy_gender.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(gender_csv_header)
        for row in rows:
            writer.writerow(row)
        f.close()	


    rows = []
    for i in range(privacy_levels):

        for j in range(num_rounds):

            os.chdir("data/")
            with open('MetaGuard Derived - Gender.csv', 'r') as f:

                my_reader = csv.reader(f, delimiter=',')
                next(f)
                for line in my_reader:
                    row = []
                    row.append(line[0]) # append ID
                    row.append(j+1) # append experiment round
                    row.append(i+1) # append privacy level
                    row.append(line[1]) # append ground truth
                    # append noisy voice-gender
                    if random.uniform(0, 1) < voice_coin_flip[i]:
                        row.append(line[2])
                    elif line[2] == 'm':
                        row.append('f')
                    else:
                        row.append('m')
                    # append noisy height
                    row.append(round(100*Laplace_Bounded_Mechanism(height_epsilons[i], height_sensitivity, height_lower, height_upper, float(line[3])), 1))
                    # append wingspan
                    row.append(round(100*Laplace_Bounded_Mechanism(wingspan_epsilons[i], wingspan_sensitivity, wingspan_lower, wingspan_upper, float(line[4])), 1))
                    # append ipd
                    row.append(round(Laplace_Bounded_Mechanism(ipd_epsilons[i], ipd_sensitivity, ipd_lower, ipd_upper, float(line[4])), 1))
                    rows.append(row)

            f.close()
            os.chdir("..")

    with open('noisy_data/noisy_gender.csv', 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)
        f.close()	