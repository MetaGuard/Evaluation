# Imports
import csv
import os
import random
import scipy.stats
import numpy as np
import pandas as pd

# Calculate confidence interval of the population mean without knowing the std of the population
# From: https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data
def mean_confidence_interval(data, confidence=0.99):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return '{} ±{}'.format(m*100, h*100), [m*100, (m-h)*100, (m+h)*100]

f = open("results/statistics.txt","w")
f.close()
f = open("results/statistics.txt","r+")
f.truncate(0)
f.close()

sample_size = 32


# # Age CSV
df_age_1 = pd.read_csv("MetaGuard_Predictions/age_predictions.csv") 
df_age_1 = df_age_1[["age_orig",	"age_predicted"]].copy()
df_age_2 = pd.read_csv("noisy_data/noisy_age.csv") 
df_age_2 = df_age_2[["round",	"privacy level"]].copy()
df_age = df_age_1.join(df_age_2)
df_age["difference"] = df_age["age_orig"] - df_age["age_predicted"]
df_age['Correct'] = df_age["difference"].apply(lambda diff: 1 if (diff < 1 and diff > -1) else 0) # within 1 year
df_age = df_age.drop(columns=["difference", "age_orig", "age_predicted"])
df_age = df_age.groupby(['privacy level', 'round']).agg({'Correct': 'sum'})

# Ethnicity CSV
df_ethnicity_1 = pd.read_csv("MetaGuard_Predictions/ethnicity_predictions.csv") 
df_ethnicity_1 = df_ethnicity_1[["ethnicity_orig",	"ethnicity_predicted"]].copy()
df_ethnicity_2 = pd.read_csv("noisy_data/noisy_ethnicity.csv") 
df_ethnicity_2 = df_ethnicity_2[["round",	"privacy level"]].copy()
df_ethnicity = df_ethnicity_1.join(df_ethnicity_2)
df_ethnicity["Correct"] = df_ethnicity.apply(lambda row: 1 if (row["ethnicity_orig"] == row["ethnicity_predicted"]) else 0, axis=1)
df_ethnicity = df_ethnicity.drop(columns=["ethnicity_orig", "ethnicity_predicted"])
df_ethnicity = df_ethnicity.groupby(['privacy level', 'round']).agg({'Correct': 'sum'})

# Gender CSV
df_gender_1 = pd.read_csv("MetaGuard_Predictions/gender_predictions.csv") 
df_gender_1 = df_gender_1[["gender_orig",	"gender_predicted"]].copy()
df_gender_2 = pd.read_csv("noisy_data/noisy_gender.csv") 
df_gender_2 = df_gender_2[["round",	"privacy level"]].copy()
df_gender = df_gender_1.join(df_gender_2)
df_gender["Correct"] = df_gender.apply(lambda row: 1 if (row["gender_orig"] == row["gender_predicted"]) else 0, axis=1)
df_gender = df_gender.drop(columns=["gender_orig", "gender_predicted"])
df_gender = df_gender.groupby(['privacy level', 'round']).agg({'Correct': 'sum'})

# ID deep CSV
df_ID_deep_1 = pd.read_csv("MetaGuard_Predictions/id_predictions_deep.csv") 
df_ID_deep_1 = df_ID_deep_1[["id_orig",	"id_predicted"]].copy()
df_ID_deep_2 = pd.read_csv("noisy_data/noisy_ID.csv") 
df_ID_deep_2 = df_ID_deep_2[["round",	"privacy level"]].copy()
df_ID_deep = df_ID_deep_1.join(df_ID_deep_2)
df_ID_deep["Correct"] = df_ID_deep.apply(lambda row: 1 if (row["id_orig"] == row["id_predicted"]) else 0, axis=1)
df_ID_deep = df_ID_deep.drop(columns=["id_orig", "id_predicted"])
df_ID_deep = df_ID_deep.groupby(['privacy level', 'round']).agg({'Correct': 'sum'})

# ID tree CSV
df_ID_trees_1 = pd.read_csv("MetaGuard_Predictions/id_predictions.csv") 
df_ID_trees_1 = df_ID_trees_1[["id_orig",	"id_predicted"]].copy()
df_ID_trees_2 = pd.read_csv("noisy_data/noisy_ID.csv") 
df_ID_trees_2 = df_ID_trees_2[["round",	"privacy level"]].copy()
df_ID_trees = df_ID_trees_1.join(df_ID_trees_2)
df_ID_trees["Correct"] = df_ID_trees.apply(lambda row: 1 if (row["id_orig"] == row["id_predicted"]) else 0, axis=1)
df_ID_trees = df_ID_trees.drop(columns=["id_orig", "id_predicted"])
df_ID_trees = df_ID_trees.groupby(['privacy level', 'round']).agg({'Correct': 'sum'})

# Calculate accuracies
with open("results/statistics.txt", "a") as f:

    print('Age prediction', file=f)
    print('High privacy level:', mean_confidence_interval(df_age['Correct'][1]/sample_size)[0], file=f)
    print('Medium privacy level:', mean_confidence_interval(df_age['Correct'][2]/sample_size)[0], file=f)
    print('Low privacy level:', mean_confidence_interval(df_age['Correct'][3]/sample_size)[0], file=f)

    print('\nEthnicity prediction', file=f)
    print('High privacy level:', mean_confidence_interval(df_ethnicity['Correct'][1]/sample_size)[0], file=f)
    print('Medium privacy level:', mean_confidence_interval(df_ethnicity['Correct'][2]/sample_size)[0], file=f)
    print('Low privacy level:', mean_confidence_interval(df_ethnicity['Correct'][3]/sample_size)[0], file=f)

    print('\nGender prediction', file=f)
    print('High privacy level:', mean_confidence_interval(df_gender['Correct'][1]/sample_size)[0], file=f)
    print('Medium privacy level:', mean_confidence_interval(df_gender['Correct'][2]/sample_size)[0], file=f)
    print('Low privacy level:', mean_confidence_interval(df_gender['Correct'][3]/sample_size)[0], file=f)

    print('\nID prediction DNN', file=f)
    print('High privacy level:', mean_confidence_interval(df_ID_deep['Correct'][1]/sample_size)[0], file=f)
    print('Medium privacy level:', mean_confidence_interval(df_ID_deep['Correct'][2]/sample_size)[0], file=f)
    print('Low privacy level:', mean_confidence_interval(df_ID_deep['Correct'][3]/sample_size)[0], file=f)

    print('\nID prediction Extreme random trees', file=f)
    print('High privacy level', mean_confidence_interval(df_ID_trees['Correct'][1]/sample_size)[0], file=f)
    print('Medium privacy level', mean_confidence_interval(df_ID_trees['Correct'][2]/sample_size)[0], file=f)
    print('Low privacy level', mean_confidence_interval(df_ID_trees['Correct'][3]/sample_size)[0], file=f)