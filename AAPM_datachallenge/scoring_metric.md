## Scoring Metrics

## Overview
In the final evaluation, the user’s MAR algorithms will be evaluated in a total of 29 patient datasets with simulated metal implants, covering the five metal categories described in Gjesteby et al. [1]. Datasets will include either single or multiple metal implants, ranging from small metal objects with low attenuation to large metal objects with high attenuation. Different, region-specific evaluation metrics will be combined to calculate an overall performance score over all patients.

## Scoring metrics 
A total of eight metrics is used to evaluate the MAR images relative to the ground truth images. For each metric, a score between 0 and 4 is assigned. A score of 0 corresponds to no relevant differences to ground truth observed. If not otherwise noted, a score of 2 corresponds to the MAR capability of a state-of-the-art analytical NMAR algorithm if applicable and a score of 4 corresponds to no improvement over the uncorrected images or still remaining large quality degradation. To allow for the distinction of smaller differences between participants, fractional scores are assigned. 

1. **CT number accuracy** is measured as the root-mean-square error (RMSE) within the patient, excluding the metal ground truth geometry.   
2. **Noise** is defined as 1 standard deviation in a circular region of interest (ROI) in a homogenous tissue region that is the least affected by the metal artifacts. For the scoring, a linear increase from the ground truth noise to approximately 2x the ground truth noise is used. 	
3. **Image sharpness**: Image sharpness is quantified as the preservation of gradients within specified ROIs containing an edge between soft tissue and bone as well as between soft and adipose tissue. For this, a Sobel filter is applied to the image data to determine the approximate absolute gradient magnitude within the ROI. To minimize the effect of outliers, the 90th percentile value within the respective ROIs are used. For scoring, the obtained values are then compared to those from the ground truth image. A score of 0 corresponds to an unaltered ground truth, whereas for a score from 1 to 4, a Gaussian blur with a sigma of 0.4, 0.5, 0.75 and 1 is introduced, respectively.   
4. **Streak amplitude**: Remaining streaks are assessed by the maximum CT number error deviation in a region of interest perpendicular to strong artifacts in the uncorrected images. For positive and negative deviations, voxels with the 5% highest error deviations are averaged, respectively, and the difference of the two values determined.
5. **Overall integrity**: The structural similarity index measure (SSIM) is used to evaluate the anatomical integrity within the patient geometry, excluding the metal ground truth. The implementation of Wang et al. [2] is followed, with the data range set to 5000.  
6. **Bone integrity**: All voxels with a CT number above 150 HU, excluding those in the metal ground truth geometry, are considered bone. Integrity is determined as the volume change relative to the ground truth image without metal artifacts and by applying the Sørensen–Dice coefficient to assess the similarity of the bone volumes. The overall bone integrity score is averaged over the two metrics.    
7. **Metal integrity**: To quantify the **metal integrity**, all voxels above a certain threshold are considered metal in the scoring program. The threshold is the highest non-metal value near the metal (listed below) **plus a margin 250 HU** for each case.  We recommend setting the metal value in the final images 1 HU higher than the threshold, as listed below.  Integrity is determined as the volume change relative to the ground truth image without metal artifacts and by applying the Sørensen–Dice coefficient to assess the similarity of the metal volumes. The analysis is limited to a ROI covering the metal ground truth and the surrounding tissue.  
01.raw    highest value 1600     (metal 1851) <br>
02.raw    highest value  1650    (metal 1901) <br>
03.raw    highest value 550       (metal 801) <br>
04.raw    highest value  100      (metal 351) <br>
05.raw    highest value 100       (metal 351) <br>
06.raw    highest value 2500     (metal 2751) <br>
07.raw    highest value 2500     (metal 2751) <br>
08.raw    highest value 2500     (metal 2751) <br>
09.raw    highest value 2500     (metal 2751) <br>
10.raw    highest value 2500     (metal 2751) <br>
11.raw    highest value 100       (metal 351) <br>
12.raw    highest value 100       (metal 351) <br>
13.raw    highest value 1400     (metal 1651) <br>
14.raw    highest value 1400     (metal 1651) <br>
15.raw    highest value 2500     (metal 2751) <br>
16.raw    highest value 2500     (metal  2751) <br>
17.raw    highest value 2500     (metal 2751)   <br>
18.raw    highest value 1900     (metal 2151) <br>
19.raw    highest value 1900     (metal (2151) <br>
20.raw    highest value 1900     (metal 2151) <br>
21.raw    highest value 1400     (metal 1651) <br>
22.raw    highest value 1400     (metal 1651) <br>
23.raw    highest value 600       (metal 851) <br>
24.raw    highest value 1000     (metal (1251) <br>
25.raw    highest value 1400     (metal 1651) <br>
26.raw    highest value 1400     (metal 1651) <br>
27.raw    highest value 1050     (metal 1301) <br>
28.raw    highest value 1050     (metal 1301) <br>
29.raw    highest value 1050     (metal 1301)

8. **Proton beam range for radiotherapy**: For radiotherapy with protons, CT numbers are translated into the different tissues’ respective stopping power relative to water (SPR). The SPR multiplied with the voxel size corresponds to the water-equivalent thickness (WET) of the tissue. The clinically validated CTN-to-SPR translation curve from Massachusetts General Hospital is applied to the MAR scans and the resulting SPR is integrated along simplified in-plane beam paths to obtain the WET. WET differences to the artifact-free ground truth are used for scoring, where a score of zero corresponds to no range shift, linearly increasing to a score of 4 corresponding to a range shift of 2% relative to the largest beam range of the treatment field. 

### Patient cases
The patient scenarios covered in the final evaluation are listed in Table 1.

<img src="https://qtim-challenges.southcentralus.cloudapp.azure.com:9000/public/ct-mar/Eval_Table1.png" width=660px>


## References
[1]: Gjesteby L, Man BDE, Jin Y, Paganetti H, Verburg J, Giantsoudi D, et al. Metal Artifact Reduction in CT : Where Are We After Four Decades ? IEEE Access 2016;4:5826–49. https://doi.org/10.1109/ACCESS.2016.2608621.

[2]: Wang Z, Bovik AC, Sheikh HR, Simoncelli EP. Image Quality Assessment: From Error Visibility to Structural Similarity. IEEE Trans Image Process 2004;13:600–12. https://doi.org/10.1109/TIP.2003.819861.
