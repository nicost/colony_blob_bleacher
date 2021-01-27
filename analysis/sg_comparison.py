import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os

# --------------------------
# PARAMETERS
# --------------------------
# paths
# data source
data_path = "/Users/xiaoweiyan/Dropbox/LAB/ValeLab/Projects/Blob_bleacher/SG_scoring/dataAnalysis/"
data1_path = "%s/WT/data_single-FOV_WT.txt" % data_path  # ctrl data (wild type data)
data2_path = "%s/CX/data_single-FOV_CX.txt" % data_path  # tested data (mutant data)
# save location
save_path = "/Users/xiaoweiyan/Dropbox/LAB/ValeLab/Projects/Blob_bleacher/SG_scoring/dataAnalysis/"

# --------------------------
# LOAD DATA
# --------------------------
data1 = pd.read_csv(data1_path, na_values=['.'], sep='\t')
data2 = pd.read_csv(data2_path, na_values=['.'], sep='\t')

# --------------------------
# OUTPUT DIR/NAMES
# --------------------------
sample_name1 = data1_path.split('"')[0].split('.')[0].split('_')[-1]
sample_name2 = data2_path.split('"')[0].split('.')[0].split('_')[-1]
storage_path = '%s/comparison_%s-%s.txt' % (save_path, sample_name1, sample_name2)
if not os.path.exists(storage_path):
    os.makedirs(storage_path)

# --------------------------
# OUTPUT FILE
# --------------------------
# Images
# circularity
fig, ax = plt.subplots(figsize=(8, 6))
sample_num = min([len(data1[data1['size'] > 50]), len(data2[data2['size'] > 50])])
histrange = (0, 1.3)
num_bin = 50
ax = plt.hist(data2[data2['size'] > 50]['circ'].sample(n=sample_num), bins=num_bin, range=histrange,
              alpha=1.0, color=(0.85, 0.35, 0.25), label=sample_name2, edgecolor=(0.2, 0.2, 0.2))
ax = plt.hist(data1[data1['size'] > 50]['circ'].sample(n=sample_num), bins=num_bin, range=histrange,
              alpha=0.5, color=(0.8, 0.8, 0.8), label=sample_name1, edgecolor=(0.2, 0.2, 0.2))
plt.xlabel('Circularity')
plt.ylabel('Counts')
plt.legend(loc=2, bbox_to_anchor=(0.02, 0.99))
plt.savefig('%s/circularity_size>50_n-%d.pdf' % (storage_path, sample_num))

# eccentricity
fig, ax = plt.subplots(figsize=(8, 6))
sample_num = min([len(data1), len(data2)])
histrange = (0, 1.0)
num_bin = 50
ax = plt.hist(data2['eccentricity'].sample(n=sample_num), bins=num_bin, range=histrange,
              alpha=1.0, color=(0.85, 0.35, 0.25), label=sample_name2, edgecolor=(0.2, 0.2, 0.2))
ax = plt.hist(data1['eccentricity'].sample(n=sample_num), bins=num_bin, range=histrange,
              alpha=0.5, color=(0.8, 0.8, 0.8), label=sample_name1, edgecolor=(0.2, 0.2, 0.2))
plt.xlabel('Eccentricity')
plt.ylabel('Counts')
plt.legend(loc=2, bbox_to_anchor=(0.02, 0.99))
plt.savefig('%s/eccentricity_n-%d.pdf' % (storage_path, sample_num))

# size (in ln scale)
fig, ax = plt.subplots(figsize=(8, 6))
sample_num = min([len(data1), len(data2)])
histrange = (1, 6)
num_bin = 50
ax = plt.hist(np.log(data2['size'].sample(n=sample_num)), bins=num_bin, range=histrange,
              alpha=1.0, color=(0.85, 0.35, 0.25), label=sample_name2, edgecolor=(0.2, 0.2, 0.2))
ax = plt.hist(np.log(data1['size'].sample(n=sample_num)), bins=num_bin, range=histrange,
              alpha=0.5, color=(0.8, 0.8, 0.8), label=sample_name1, edgecolor=(0.2, 0.2, 0.2))
plt.xlabel('ln(Size(pixel))')
plt.ylabel('Counts')
plt.legend(loc=2, bbox_to_anchor=(0.02, 0.99))
plt.savefig('%s/ln(size)_n-%d.pdf' % (storage_path, sample_num))

# intensity (in ln scale)
fig, ax = plt.subplots(figsize=(8, 6))
sample_num = min([len(data1), len(data2)])
histrange = (6, 10)
num_bin = 50
ax = plt.hist(np.log(data2['int'].sample(n=sample_num)), bins=num_bin, range=histrange,
              alpha=1.0, color=(0.85, 0.35, 0.25), label=sample_name2, edgecolor=(0.2, 0.2, 0.2))
ax = plt.hist(np.log(data1['int'].sample(n=sample_num)), bins=num_bin, range=histrange,
              alpha=0.5, color=(0.8, 0.8, 0.8), label=sample_name1, edgecolor=(0.2, 0.2, 0.2))
plt.xlabel('ln(Intensity(AU))')
plt.ylabel('Counts')
plt.legend(loc=2, bbox_to_anchor=(0.02, 0.99))
plt.savefig('%s/ln(intensity)_n-%d.pdf' % (storage_path, sample_num))