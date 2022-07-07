import numpy as np
import statistics
from geopy.distance import geodesic
import glob
import scipy.stats
import os

# Calculate confidence interval of the population mean without knowing the std of the population
# From: https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data
def mean_confidence_interval(data, confidence=0.99):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return '{} ±{}'.format(m, h), [m, m-h, m+h]

### No VR privacy protection ####################

us_east_2_avg = []
us_west_1_avg = []
us_west_2_avg = []

noisy_us_east_2_avg = []
noisy_us_west_1_avg = []
noisy_us_west_2_avg = []

os.chdir("data/")
for name in glob.glob("*.dat"):
    us_east_2 = []
    us_west_1 = []
    us_west_2 = []
    noisy_us_east_2 = []
    noisy_us_west_1 = []
    noisy_us_west_2 = []
    with open(name) as file:
        for line in file:
            parts = line.split(', ')
            us_east_2.append(int(parts[0]))
            us_west_1.append(int(parts[1]))
            us_west_2.append(int(parts[2]))

    us_east_2_avg.append(statistics.median(us_east_2))
    us_west_1_avg.append(statistics.median(us_west_1))
    us_west_2_avg.append(statistics.median(us_west_2))
os.chdir("..")

def transpose_1D(M):
    return M.reshape(len(M), 1)

def multilaterate(anchor_positions, distances):
    N = anchor_positions.shape[0]
    A = np.vstack([np.ones(N), -2 * anchor_positions[:, 0], -2 * anchor_positions[:, 1]]).T
    B = distances ** 2 - anchor_positions[:, 0] ** 2 - anchor_positions[:, 1] ** 2
    X = np.dot(A.T, A)
    xp = np.dot(np.dot(np.linalg.inv(X), A.T), B)
    return xp[1:]

N = len(us_east_2_avg)
dist = []
lat_dist = []
long_dist = []
for i in range(0,N):
    d = 0.496
    loc = multilaterate(np.array([
        [40.093655163828615, -82.75021108220088],
        [37.413769057554184, -121.97182817337978],
        [45.921586440357295, -119.26565997256668],
    ]), np.array([
        (us_east_2_avg[i]-2)*d,
        (us_west_1_avg[i]-2)*d,
        (us_west_2_avg[i]-2)*d,
    ]))
    # Find the shortest path between the predicted point (loc[0], loc[1]) and the actual location (37.87680775, -122.2563914)
    lat_dist.append(abs(loc[0]-37.87680775))
    long_dist.append(abs(loc[1]-(-122.2563914)))   
    dist.append(geodesic((loc[0], loc[1]), (37.87680775, -122.2563914)).km)

f = open("avg_statistics.txt","w")
f.close()
f = open("avg_statistics.txt","r+")
f.truncate(0)
f.close()

with open("avg_statistics.txt", "a") as f:
    print('No protection', file=f)

    print('Latitude:', file=f)
    print('Within 2.5°:', len([x for x in lat_dist if x <= 2.5])/N, file=f)
    print('Within 3°:', len([x for x in lat_dist if x <= 3])/N, file=f)
    print('Within 4°:', len([x for x in lat_dist if x <= 4])/N, file=f)

    print('Longitude:', file=f)
    print('Within 2.5°:', len([x for x in long_dist if x <= 2.5])/N, file=f)
    print('Within 3°:', len([x for x in long_dist if x <= 3])/N, file=f)
    print('Within 4°:', len([x for x in long_dist if x <= 4])/N, file=f)

    print('Coordinates:', file=f)
    print('Within 400km:', len([x for x in dist if x <= 400])/N, file=f)
    print('Within 500km:', len([x for x in dist if x <= 500])/N, file=f)
    f.close()

### With VR privacy protection ####################
# Set clamping value for latency (ms)
clamping_value = 25

os.chdir("data/")
for name in glob.glob("*.dat"):

    us_east_2 = []
    us_west_1 = []
    us_west_2 = []
    noisy_us_east_2 = []
    noisy_us_west_1 = []
    noisy_us_west_2 = []
    with open(name) as file:
        for line in file:
            parts = line.split(', ')
            us_east_2.append(int(parts[0]))
            us_west_1.append(clamping_value)
            us_west_2.append(int(parts[2]))

    noisy_us_east_2_avg.append(statistics.median(us_east_2))
    noisy_us_west_1_avg.append(statistics.median(us_west_1))
    noisy_us_west_2_avg.append(statistics.median(us_west_2))
os.chdir("..")

N = len(us_east_2_avg)
dist = []
lat_dist = []
long_dist = []
for i in range(0,N):
    d = 0.496
    loc = multilaterate(np.array([
        [40.093655163828615, -82.75021108220088],
        [37.413769057554184, -121.97182817337978],
        [45.921586440357295, -119.26565997256668],
    ]), np.array([
        (noisy_us_east_2_avg[i]-2)*d,
        (noisy_us_west_1_avg[i]-2)*d,
        (noisy_us_west_2_avg[i]-2)*d,
    ]))
    # Find the shortest path between the predicted point (loc[0], loc[1]) and the actual location (37.87680775, -122.2563914)
    lat_dist.append(abs(loc[0]-37.87680775))
    long_dist.append(abs(loc[1]-(-122.2563914)))   
    dist.append(geodesic((loc[0], loc[1]), (37.87680775, -122.2563914)).km)

    
with open("avg_statistics.txt", "a") as f:
    print('With protection', file=f)

    print('Latitude:', file=f)
    print('Within 2.5°:', len([x for x in lat_dist if x <= 2.5])/N, file=f)
    print('Within 3°:', len([x for x in lat_dist if x <= 3])/N, file=f)
    print('Within 4°:', len([x for x in lat_dist if x <= 4])/N, file=f)

    print('Longitude:', file=f)
    print('Within 2.5°:', len([x for x in long_dist if x <= 2.5])/N, file=f)
    print('Within 3°:', len([x for x in long_dist if x <= 3])/N, file=f)
    print('Within 4°:', len([x for x in long_dist if x <= 4])/N, file=f)

    print('Coordinates:', file=f)
    print('Within 400km:', len([x for x in dist if x <= 400])/N, file=f)
    print('Within 500km:', len([x for x in dist if x <= 500])/N, file=f)
    print('Within 700km:', len([x for x in dist if x <= 700])/N, file=f)
    print('Within 900km:', len([x for x in dist if x <= 900])/N, file=f)
    print('Within 1700km:', len([x for x in dist if x <= 1700])/N, file=f)
    f.close()