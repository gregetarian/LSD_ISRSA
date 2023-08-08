# LSD_ISRSA
Code to run searchlight-wise functional hyperalignment and inter-subject representational similarity analysis on the Carhart-Harris 2016 LSD fMRI dataset (unscrubbed version provided in personal communication)

First run hyperalign_searchlight_LSD.py in a python 2.7 environment
- this step hyperaligns data to a common space in a searchlight-wise fashion. Implemented in pyMVPA2 as per Guntapalli et al., 2016

Second, run LSDhyperaligned_is_RSA_searchlight.py in a python 3.8+ environment. 
- this step performs a searchlight-wise intersubject representational similarity analysis to determine which hyperaligned voxels encode intersubject variation in a given behavioural score as per Finn et al., 2020.

- This is performed in 3 steps:
i). An interparticipant similarity matrix is computed for participant's behavioural scores for a chosen dimension (in this case, ego dissolution).
  
ii). Timeseries are created for 3mm radii spheres (aka searchlights), over which an intersubject similarity matrix is computed for each searchlight individually. This step is parallelised across all available cores by default.
  
iii). The euclidian distance between each searchlight-wise intersubject similarity matrix computed in step 2, and behavioural intersubject similarity matrix computed in step 1 is computed. This outputs a nifti image of voxelwise representational similarity between intersubject variation in voxelwise activity and intersubject variation in subjective experience across the chosen behavioural dimension. This step is parallelised across all available cores by default.

Finally, run FWHMx_hyperaligned_multithresh_rsa.sh for cluster-size correction on outputs.

**References**

Finn, E.S., Glerean, E., Khojandi, A.Y., Nielson, D., Molfese, P.J., Handwerker, D.A. and Bandettini, P.A., 2020. Idiosynchrony: From shared responses to individual differences during naturalistic neuroimaging. NeuroImage, 215, p.116828.

Guntupalli, J.S., Hanke, M., Halchenko, Y.O., Connolly, A.C., Ramadge, P.J. and Haxby, J.V., 2016. A model of representational spaces in human cortex. Cerebral cortex, 26(6), pp.2919-2934.
