import os
import numpy as np
import pandas as pd
from nilearn.input_data import NiftiLabelsMasker, NiftiSpheresMasker
from nilearn.plotting import plot_glass_brain, plot_stat_map
from nltools.data import Brain_Data, Adjacency
from nltools.mask import roi_to_brain, expand_mask
from nltools.stats import threshold
from sklearn.metrics import pairwise_distances
from scipy.stats import ttest_1samp
from nilearn import datasets
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm
import pickle
from nibabel.affines import apply_affine
from nilearn.image import load_img
from nilearn.image import new_img_like


#note that nilearn requires a python 3.8+ environment to run

# -------- Configuration and Initializations ----------

LSD_brain_dir = '/Users/gcooper/Downloads/FD_040_NoScrub/LSD/rest2/hyperaligned'
behav_dir =('/Users/gcooper/Documents/LSD_RSA/behavioural')

S1_subj_list = [ "S01","S04", "S09", "S13", "S17", "S20" ]
S2_subj_list = ["S02", "S06", "S10", "S11","S18","S19"]

template = datasets.load_mni152_gm_mask()
mask_img = template
mask_img = load_img('/Users/gcooper/Downloads/FD_040_NoScrub/LSD/rest2/nzmask.nii.gz')
seeds=[]

# --------- Functions Definitions ------------

def scale_mtx(mtx):
    return (mtx-np.min(mtx))/(np.max(mtx)-np.min(mtx))

# def create_searchlight_time_series(subj):
#     searchlight_time_series_fname = os.path.join(LSD_brain_dir, subj+'_hyperaligned_searchlightTimeSeries.csv')
    
#     if os.path.exists(searchlight_time_series_fname):
#         print(f"Searchlight time series for {subj} already exists!")
#     else:
#         print(f"Creating searchlight time series for {subj}")
#         # Pass the seeds to the NiftiSpheresMasker
#         searchlight_masker = NiftiSpheresMasker(seeds, radius=3, standardize=True, allow_overlap=True)
#         searchlight_time_series = searchlight_masker.fit_transform(os.path.join(LSD_brain_dir, subj+'_hyperaligned.nii.gz'))
#         pd.DataFrame(searchlight_time_series).to_csv(searchlight_time_series_fname, index=False)

# Add this function to your script
def create_seeds(mask_img):
    # Load the mask image
    mask_img_obj = load_img(mask_img)
    mask_data = mask_img_obj.get_fdata()
    mask_affine = mask_img_obj.affine

    # Loop over the entire volume and add the coordinates to seeds list
    for i in range(mask_data.shape[0]):
        for j in range(mask_data.shape[1]):
            for k in range(mask_data.shape[2]):
                # Only add the coordinates if the voxel is inside the mask
                if mask_data[i, j, k] > 0:
                    # Transform voxel coordinates to world coordinates
                    world_coords = apply_affine(mask_affine, [i, j, k])
                    seeds.append(world_coords)

    # Now 'seeds' contains the coordinates of all voxels in the mask
    print(f"Created {len(seeds)} seeds")
    return seeds

def create_searchlight_time_series(subj):
    searchlight_time_series_fname = os.path.join(LSD_brain_dir, subj+'_hyperaligned_searchlightTimeSeries.csv')
    
    if os.path.exists(searchlight_time_series_fname):
        print(f"Searchlight time series for {subj} already exists!")

    else:
        print(f"Creating searchlight time series for {subj}")
        print(f"Nifti file path: {os.path.join(LSD_brain_dir, subj+'_hyperaligned.nii.gz')}")
        searchlight_masker = NiftiSpheresMasker(seeds, radius=3, standardize=True, allow_overlap=True)
        searchlight_time_series = searchlight_masker.fit_transform(os.path.join(LSD_brain_dir, subj+'_hyperaligned.nii.gz'))
        pd.DataFrame(searchlight_time_series).to_csv(searchlight_time_series_fname, index=False)


# Function to map values to the brain
def map_values_to_brain(values, seeds, template):
    result_data = np.zeros(template.shape)
    for seed, value in zip(seeds, values):
        voxel_coords = apply_affine(np.linalg.inv(template.affine), seed)
        voxel_coords = [int(round(coord)) for coord in voxel_coords] # Ensure the voxel coordinates are integers
        result_data[voxel_coords[0], voxel_coords[1], voxel_coords[2]] = value
    result_img = new_img_like(template, result_data)
    return result_img


# --------- Function to Run Analysis ------------

def run_analysis(song_number, subj_list, behav_column):
    # Load data
    data = pd.read_csv(f"{behav_dir}/song{song_number}_asc.csv")
    behav = data[behav_column].values
    n_subs = len(behav)

    # Compute Behavioral Similarity Matrix
    behav_mtx = np.zeros((n_subs, n_subs))
    for i in range(n_subs):
        for j in range(n_subs):
            if i < j:
                dist_ij = np.mean([behav[i]/n_subs, behav[j]/n_subs])
                behav_mtx[i,j] = dist_ij
                behav_mtx[j,i] = dist_ij
    behav_mtx = scale_mtx(behav_mtx)
    np.fill_diagonal(behav_mtx, 1)

    # Get Searchlight Time Series Data
    seeds = create_seeds(mask_img)

    Parallel(n_jobs=-1)(delayed(create_searchlight_time_series)(subj) for subj in tqdm(subj_list))

    data = []
    for subj in subj_list:
        print (f'reading in searchlight time series for sub-{subj}')
        searchlight_time_series_fname = os.path.join(LSD_brain_dir, subj+f'_hyperaligned_searchlightTimeSeries.csv')
        data.append(pd.read_csv(searchlight_time_series_fname).values)

    print('preparing arrays for ISC')
    data = np.array(data)
    n_subs, n_ts, n_voxels = data.shape

    # Compute Inter-Subject Correlation (ISC)
    # Compute Inter-Subject Correlation (ISC)
    print('Calculating Similarity Matrices  ...')

    similarity_matrices_path = f'{LSD_brain_dir}/results/similarity_matrices_song{song_number}.pickle'
    isc_path = f'{LSD_brain_dir}/results/isc_song{song_number}.pickle'
    
    if os.path.exists(similarity_matrices_path):
        with open(similarity_matrices_path, 'rb') as handle:
            similarity_matrices = pickle.load(handle)
        print('Loaded similarity matrices from disk.')
    else:
        similarity_matrices = [Adjacency(1 - pairwise_distances(data[:, :, voxel], metric='correlation'), matrix_type='similarity') for voxel in tqdm(range(n_voxels), desc='computing similarity matrices')]
        similarity_matrices = Adjacency(similarity_matrices)

        print('Saving Similarity Matrix')
        with open(similarity_matrices_path, 'wb') as handle:
            pickle.dump(similarity_matrices, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('Calculating Inter-Subject Correlation (ISC) ...')
    if os.path.exists(isc_path):
        with open(isc_path, 'rb') as handle:
            isc = pickle.load(handle)
        print('Loaded ISC from disk.')
    else:
        isc = {voxel: similarity_matrices[voxel].isc(metric='mean', n_bootstraps=1, n_jobs=-1)['isc'] for voxel in range(n_voxels)}
        print('Saving ISC')
        with open(isc_path, 'wb') as handle:
            pickle.dump(isc, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # After computing the ISC values, call the map_values_to_brain function:
    isc_brain = map_values_to_brain(list(isc.values()), seeds, template)

    # Now you can save the ISC brain map to a NIfTI file
    isc_brain.to_filename(f'{LSD_brain_dir}/results/isc_brain_song{song_number}.nii.gz')

    # Compute Inter-Subject Representational Similarity Analysis (IS-RSA)
    isrsa = {}
    for voxel in tqdm(range(len(similarity_matrices)), desc='computing non-perm IS-RSA'):
        isrsa[voxel] = similarity_matrices[voxel].similarity(behav_mtx, metric='spearman', n_permute=1, n_jobs=-1 )['correlation']

    isrsa_brain = map_values_to_brain(list(isrsa.values()), seeds, template)
    print('saving IS-RSA_non-perm')
    with open(f'{LSD_brain_dir}/results/isrsa_brain_song{song_number}_non-perm', 'wb') as handle:
        pickle.dump(isrsa_brain, handle, protocol=pickle.HIGHEST_PROTOCOL)

    isrsa_brain.to_filename(f'{LSD_brain_dir}/results/isrsa_brain_song{song_number}_{behav_column}.nii.gz')


# Run analysis for Song 1 and Song 2

#run_analysis(1, S1_subj_list, "Ego Dissolution")
# run_analysis(2, S2_subj_list, "Ego Dissolution")
# run_analysis(1, S1_subj_list, "Simple")
# run_analysis(2, S2_subj_list, "Simple")
run_analysis(1, S1_subj_list, "Cmplx")
run_analysis(2, S2_subj_list, "Cmplx")