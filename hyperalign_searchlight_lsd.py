

import os
import nibabel as nib
import numpy as np
from mvpa2.suite import *
#import mvpa2.suite


### note that pyMVPA2 requires a python 2.7 environment to run 

LSD_brain_dir = '/Users/gcooper/Downloads/FD_040_NoScrub/LSD/rest2/'

S1_subj_list = ["S01", "S04", "S09", "S13", "S17", "S20"]
# S2_subj_list = ["S02", "S06", "S10", "S11", "S18", "S19"]


mask='/Users/gcooper/Downloads/FD_040_NoScrub/LSD/rest2/nzmask.nii.gz'
datasets = []

from nibabel import Nifti1Image

print("loading in datasets")
num_features = None  # Number of features in the datasets
for subj in S1_subj_list:
    filename = os.path.join(LSD_brain_dir, "{}_LSD_rest2_clean.nii.gz".format(subj))
    if os.path.exists(filename):
        ds = fmri_dataset(samples=filename, mask=mask)
        if num_features is None:
            num_features = ds.shape[1]  # Get the number of features from the first dataset
            print(num_features)
        elif num_features != ds.shape[1]:
            print("Number of features in the datasets is not consistent.")
            exit(1)
        
        
        ds.sa['subject'] = [subj] * len(ds)
        ds.sa['chunks'] = [1] * len(ds)  # Assigning a 'chunks' attribute with a dummy value
        datasets.append(ds)

# Check the number of voxels per subject after reloading the datasets
print("Number of voxels per subject after reloading the datasets:")
for i, ds in enumerate(datasets):
    num_voxels = ds.shape[1]
    print("Subject {}: {}".format(S1_subj_list[i], num_voxels))

for i, ds in enumerate(datasets):
    print("Subject {} has {} invariant features.".format(i, np.sum(np.std(ds.samples, axis=0) == 0)))

# After loading all datasets, calculate the common mask
common_mask = np.ones((num_features,), dtype=bool)
for ds in datasets:
    invariant_voxels_mask = np.std(ds.samples, axis=0) == 0  # Mask for invariant voxels in the current dataset
    common_mask = np.logical_and(common_mask, ~invariant_voxels_mask)  # Update the common mask

# Apply the common mask to all datasets
for i in range(len(datasets)):
    datasets[i] = datasets[i][:, common_mask]

# Check the number of voxels per subject after applying the common mask
print("Number of voxels per subject after applying the common mask:")
for i, ds in enumerate(datasets):
    num_voxels = ds.shape[1]
    print("Subject {}: {}".format(S1_subj_list[i], num_voxels))


# Z-score all datasets individually
_ = [zscore(ds, chunks_attr='chunks') for ds in datasets]

# Inject the subject ID into all datasets
for i, sd in enumerate(datasets):
    sd.sa['subject'] = np.repeat(i, len(sd))

# Number of subjects
nsubjs = len(datasets)

verbose(1, "Performing hyperalignment...")
verbose(2, "between-subject (searchlight hyperaligned)...", cr=False, lf=False)

# Feature selection helpers
slhyper_start_time = time.time()
bsc_slhyper_results = []


# No need for leave-one-run-out here, instead we perform leave-one-subject-out
for test_subj in range(nsubjs):
    # Split into training and testing set
    ds_train = [sd.copy() for i, sd in enumerate(datasets) if i != test_subj]
    ds_test = [datasets[test_subj].copy()]

    # Remove any extra features from the datasets
    for ds in ds_train + ds_test:
        ds.samples = ds.samples[:, :num_features]

    # Initialize Searchlight Hyperalignment with Sphere searchlights of 3 voxel radius.
    # Using 40% features in each SL and spacing centers at 3-voxels distance.
    slhyper = SearchlightHyperalignment(radius=3, featsel=0.4, sparse_radius=3)

    # Perform searchlight hyperalignment on training data.
    slhypmaps = slhyper(ds_train)

    # Apply hyperalignment parameters by running the test dataset
    # through the forward() function of the mapper.
    ds_hyper = [h.forward(sd) for h, sd in zip(slhypmaps, ds_test)]

    # Save the hyperaligned dataset as a new Nifti file
    # Map the hyperaligned dataset back to Nifti space
    nifti_img = map2nifti(ds_hyper[0])

    # Save the Nifti image to a file
    hyperaligned_filename = os.path.join(LSD_brain_dir, "{}_hyperaligned.nii.gz".format(S1_subj_list[test_subj]))
    nifti_img.to_filename(hyperaligned_filename)
    verbose(2, "Hyperaligned dataset saved for subject {} at: {}".format(S1_subj_list[test_subj], hyperaligned_filename))


verbose(2, "Hyperalignment done in %.1f seconds" % (time.time() - slhyper_start_time,))


PCB_brain_dir = '/Users/gcooper/Downloads/FD_040_NoScrub/PCB/rest2/'
#S1_subj_list = ["S01", "S04", "S09", "S13", "S17", "S20"]
S1_subj_list = ["S02", "S06", "S10", "S11", "S18", "S19"]


mask='/Users/gcooper/Downloads/FD_040_NoScrub/LSD/rest2/nzmask.nii.gz'
datasets = []

from nibabel import Nifti1Image

print("loading in datasets")
num_features = None  # Number of features in the datasets
for subj in S1_subj_list:
    filename = os.path.join(PCB_brain_dir, "{}_PCB_rest2_clean.nii.gz".format(subj))
    if os.path.exists(filename):
        ds = fmri_dataset(samples=filename, mask=mask)
        if num_features is None:
            num_features = ds.shape[1]  # Get the number of features from the first dataset
            print(num_features)
        elif num_features != ds.shape[1]:
            print("Number of features in the datasets is not consistent.")
            exit(1)
        
        
        ds.sa['subject'] = [subj] * len(ds)
        ds.sa['chunks'] = [1] * len(ds)  # Assigning a 'chunks' attribute with a dummy value
        datasets.append(ds)

# Check the number of voxels per subject after reloading the datasets
print("Number of voxels per subject after reloading the datasets:")
for i, ds in enumerate(datasets):
    num_voxels = ds.shape[1]
    print("Subject {}: {}".format(S1_subj_list[i], num_voxels))

for i, ds in enumerate(datasets):
    print("Subject {} has {} invariant features.".format(i, np.sum(np.std(ds.samples, axis=0) == 0)))

# After loading all datasets, calculate the common mask
common_mask = np.ones((num_features), dtype=bool)
for ds in datasets:
    invariant_voxels_mask = np.std(ds.samples, axis=0) == 0  # Mask for invariant voxels in the current dataset
    common_mask = np.logical_and(common_mask, ~invariant_voxels_mask)  # Update the common mask

# Apply the common mask to all datasets
for i in range(len(datasets)):
    datasets[i] = datasets[i][:, common_mask]

# Check the number of voxels per subject after applying the common mask
print("Number of voxels per subject after applying the common mask:")
for i, ds in enumerate(datasets):
    num_voxels = ds.shape[1]
    print("Subject {}: {}".format(S1_subj_list[i], num_voxels))


# Z-score all datasets individually
_ = [zscore(ds, chunks_attr='chunks') for ds in datasets]

# Inject the subject ID into all datasets
for i, sd in enumerate(datasets):
    sd.sa['subject'] = np.repeat(i, len(sd))

# Number of subjects
nsubjs = len(datasets)

verbose(1, "Performing hyperalignment...")
verbose(2, "between-subject (searchlight hyperaligned)...", cr=False, lf=False)

# Feature selection helpers
slhyper_start_time = time.time()
bsc_slhyper_results = []


# No need for leave-one-run-out here, instead we perform leave-one-subject-out
for test_subj in range(nsubjs):
    # Split into training and testing set
    ds_train = [sd.copy() for i, sd in enumerate(datasets) if i != test_subj]
    ds_test = [datasets[test_subj].copy()]

    # Remove any extra features from the datasets
    for ds in ds_train + ds_test:
        ds.samples = ds.samples[:, :num_features]

    # Initialize Searchlight Hyperalignment with Sphere searchlights of 3 voxel radius.
    # Using 40% features in each SL and spacing centers at 3-voxels distance.
    slhyper = SearchlightHyperalignment(radius=3, featsel=0.4, sparse_radius=3)

    # Perform searchlight hyperalignment on training data.
    slhypmaps = slhyper(ds_train)

    # Apply hyperalignment parameters by running the test dataset
    # through the forward() function of the mapper.
    ds_hyper = [h.forward(sd) for h, sd in zip(slhypmaps, ds_test)]

    # Save the hyperaligned dataset as a new Nifti file
    # Map the hyperaligned dataset back to Nifti space
    nifti_img = map2nifti(ds_hyper[0])

    # Save the Nifti image to a file
    hyperaligned_filename = os.path.join(PCB_brain_dir, "{}_hyperaligned.nii.gz".format(S1_subj_list[test_subj]))
    nifti_img.to_filename(hyperaligned_filename)
    verbose(2, "Hyperaligned dataset saved for subject {} at: {}".format(S1_subj_list[test_subj], hyperaligned_filename))


verbose(2, "Hyperalignment done in %.1f seconds" % (time.time() - slhyper_start_time,))

