## AAPM CT Metal Artifact Reduction Grand Challenge Benchmark Tool
The CT Metal Artifact Reduction (CT-MAR) benchmark tool (training/validation datasets, scoring datasets, scoring program) used for the AAPM CT-MAR Grand Challenge (2024) is now available to everyone. You can use the CT-MAR benchmark tool to evaluate your developed MAR offline. This guide explains how to download the datasets and use the scoring program. 

### Background (AAPM CT Metal Artifact Reduction Grand Challenge)
The AAPM CT Metal Artifact Reduction (CT-MAR) challenge was organized by GE HealthCare Technology and Innovation Center, Massachusetts General Hospital, and Rensselaer Polytechnic Institute between Oct 2023 - Jul 2024. Participants were invited to develop a 2D metal artifact reduction (MAR) algorithm. This challenge provided 14,000 CT training datasets generated with the open-source CT simulation environment XCIST [2], using a hybrid data simulation framework that combines publicly available clinical images [3, 4] and virtual metal objects. Each dataset includes 5 components: CT sinograms (with and without metals), CT reconstructed images (with and without metal artifacts), and metal masks. In the final evaluation, a total of 29 clinical uncorrected datasets was provided in both in sinograms and reconstructed images. The inserted metals include surgical clips, dental fillings, and hip prosthesis and others. The participants’ submitted images were evaluated using our defined scoring metrics.

The AAPM CT-MAR challenge consisted of 3 phases: 
* Phase 1: Training & development phase (14,000 training/validation datasets were provided)
* Phase 2: Feedback & refinement phase (5 feedback datasets were provided)
* Phase 3: Final scoring phase (29 scoring datasets were provided)

During this challenge, the feedback datasets were provided in Phase 2 so that the participants could receive a preliminary score. For this benchmark tool, you don’t need the feedback datasets. Only the training/validation datasets and scoring datasets are needed to evaluate your MAR algorithm. 

### Get Started 
To start your benchmark, please follow the steps below.

1)	Read the [“Benchmark Tool Overview” page](Benchmark_Tool_Overview.md).
2)	Download 14,000 training/validation datasets from here: https://rpi.box.com/s/7p8tkqj5ewhtdad2h8kx975i9qg6b7a4. Please read “README_training_data.txt” on the link for the details. If you want to know how the data is generated, please read the [“Data_generation” page](data_generation.md).
3)	Train your MAR algorithm. 
    * You can download Python-based 2D Projection & Reconstruction code from here: https://github.com/xcist/example/tree/main/AAPM_datachallenge/proj_and_recon
	* The output of your MAR algorithm should be a metal-artifact-free recon image but containing metals (512x512 pixels, same format as the recon images provided in the training datasets). Make sure to use the provided recon FOVs.

4)	Download the 29 scoring datasets from https://rpi.box.com/s/p8aayubdww9tav66urn9tvpsv2bwyxar. Please read “README.txt” on the link. 

5)	Process the 29 scoring datasets with your MAR algorithm to generate metal-artifact-free recon images but containing metals (512x512 pixels, same format as the recon images provided in the training and scoring datasets). Make sure to use the provided recon FOVs in README.txt.

6)	Evaluate your MAR algorithm.
	* Download the Python scoring program from https://github.com/xcist/example/tree/main/AAPM_datachallenge/scoring.
 	* Read the instructions and run the scoring program on your 29 processed datasets. The evaluation results will be saved as “scores.html” and “scores.txt”.
    * The details of scoring metrics can be found on [“Scoring_metrics” page](scoring_metric.md).
    * For your reference, the scores of AAPM CT-MAR Challenge participants can be found on this page.  https://qtim-challenges.southcentralus.cloudapp.azure.com/competitions/1/. Visit the “Results” tab, and select “Phase 3 (Final scoring)” tab. Click eye-shaped button to see their detailed scores.

### Acknowledgements 
In your publications, please acknowledge the AAPM CT-MAR challenge data and benchmark tool using the following sentences and references:
`This work used the AAPM CT-MAR Grand Challenge datasets and benchmark tool [1, 2]. The AAPM CT-MAR Grand Challenge datasets were generated with the open-source CT simulation environment XCIST [3], using a hybrid data simulation framework that combines publicly available clinical images [4, 5] and virtual metal objects.`
1. AAPM CT Metal Artifact Reduction (CT-MAR) Grand Challenge: https://www.aapm.org/GrandChallenge/CT-MAR
2. AAPM CT Metal Artifact Reduction (CT-MAR) Grand Challenge Benchmark Tool: https://github.com/xcist/example/tree/main/AAPM_datachallenge/
3. M. Wu, P. FitzGerald, J. Zhang, W.P. Segars, H. Yu, Y. Xu, B. De Man, "XCIST - an open access x-ray/CT simulation toolkit," Phys Med Biol. 2022 Sep 28;67(19)
4. Yan K, Wang X, Lu L, Summers RM, "DeepLesion: Automated Mining of Large-Scale Lesion Annotations and Universal Lesion Detection with Deep Learning", Journal of Medical Imaging 5(3), 036501 (2018), doi: 10.1117/1.JMI.5.3.036501
5. Goren N, Dowrick T, Avery J & Holder D. (2017). UCLH Stroke EIT Dataset - Radiology Data (CT). Zenodo. https://doi.org/10.5281/zenodo.838704
