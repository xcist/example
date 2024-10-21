## CT MAR training data and scoring benchmark

The CT Metal Artifact Reduction (MAR) training data and scoring benchmark used for the AAPM CT MAR Grand Challenge (2024) are now available to everyone. The CT MAR training data can be used to train 2D MAR algorithms. The CT MAR scoring benchmark can be used to objectively and comparatively evaluate new 2D MAR approaches. This guide explains how to download the datasets and how to use the scoring benchmark. 

### Background (AAPM CT Metal Artifact Reduction Grand Challenge)

The AAPM CT Metal Artifact Reduction (CT MAR) challenge was organized by GE HealthCare Technology and Innovation Center, Massachusetts General Hospital, and Rensselaer Polytechnic Institute from Oct 2023 until Jul 2024. Participants were invited to develop a 2D MAR algorithm. This challenge provided 14,000 CT training datasets generated with the CatSim simulator in the open-source toolkit XCIST [3,4], using a hybrid data simulation framework that combines publicly available clinical images [5, 6] and virtual metal objects. Each dataset includes 5 components: CT sinograms (with and without metals), CT reconstructed images (with and without metal artifacts), and metal masks (images). For the final evaluation a scoring benchmark was provided, including 29 clinical uncorrected datasets (collected at Massachusetts General Hospital) as both sinograms and reconstructed images. The inserted metals include surgical clips, dental fillings, and hip prosthesis and others. The participants’ submitted images were evaluated using our custom-defined scoring routines.

The AAPM CT-MAR challenge consisted of 3 phases: 
* Phase 1: Training & development phase (14,000 training/validation datasets were provided)
* Phase 2: Feedback & refinement phase (5 feedback datasets were provided. Not included in this package)
* Phase 3: Final scoring phase (29 scoring datasets were provided)

The next sections provide more details on how to download and use the training/validation datasets and the scoring benchmark. 

### Downloading and using the MAR training datasets 
1)	Read [“Overview”](Overview.md) page.
2)	Download 14,000 training/validation datasets from here: https://rpi.box.com/s/7p8tkqj5ewhtdad2h8kx975i9qg6b7a4. Please read “README_training_data.txt” on the link for the details. If you want to know how the data is generated, please read [“Data_generation”](data_generation.md) page.
3)	Train your MAR algorithm. 
	* You can download Python-based 2D Projection & Reconstruction code from here: https://github.com/xcist/example/tree/main/AAPM_datachallenge/proj_and_recon
	* The output of your MAR algorithm should be a metal-artifact-free recon image but containing metals (512x512 pixels, same format as the recon images provided in the training datasets). Make sure to use the provided recon FOVs.

4)	Acknowledgements

In any publications that use these training datasets, please acknowledge the AAPM CT-MAR challenge data using the following sentences and references: *This work used the AAPM CT-MAR Grand Challenge datasets [1, 2]. The AAPM CT-MAR Grand Challenge datasets were generated with the open-source CT simulation environment XCIST [3], using a hybrid data simulation framework that combines publicly available clinical images [4, 5] and virtual metal objects.*

[1] AAPM CT Metal Artifact Reduction (CT-MAR) Grand Challenge: https://www.aapm.org/GrandChallenge/CT-MAR/

[2] AAPM CT Metal Artifact Reduction (CT-MAR) Grand Challenge Benchmark Tool: https://github.com/xcist/example/tree/main/AAPM_datachallenge/

[3] Wu M, FitzGerald P, Zhang J, Segars WP, Yu J, Xu Y, De Man B. XCIST - an open access x-ray/CT simulation toolkit. Phys Med Biol. 2022 Sep 28;67(19)

[4] Yan K, Wang X, Lu L, Summers RM, "DeepLesion: Automated Mining of Large-Scale Lesion Annotations and Universal Lesion Detection with Deep Learning", Journal of Medical Imaging 5(3), 036501 (2018), doi: 10.1117/1.JMI.5.3.036501

[5] Goren N, Dowrick T, Avery J & Holder D. (2017). UCLH Stroke EIT Dataset - Radiology Data (CT). Zenodo. https://doi.org/10.5281/zenodo.838704

### Downloading and using the MAR scoring benchmark

1)	Read [“Overview”](Overview.md) page.

2)	Download the 29 scoring datasets from https://rpi.box.com/s/p8aayubdww9tav66urn9tvpsv2bwyxar. **Please read “README.txt” on the link.**

3)	Process the 29 scoring datasets with your MAR algorithm to generate metal-artifact-free recon images but containing metals (512x512 pixels, same format as the recon images provided in the training datasets). Make sure to use the provided recon FOVs in README.txt.

4)	Evaluate your MAR algorithm. Download the Python scoring program from https://github.com/xcist/example/tree/main/AAPM_datachallenge/proj_and_recon

5)	Read the instructions and run the scoring program on your 29 processed datasets. The evaluation results will be saved as “scores.html” and “scores.txt”. 

	* The details of scoring metrics can be found on [“Scoring_metrics”](scoring_metric.md) page.

	* For your reference, the scores of AAPM CT-MAR Challenge participants can be found on this page.  https://qtim-challenges.southcentralus.cloudapp.azure.com/competitions/1/. Visit the “Results” tab, and select “Phase 3 (Final scoring)” tab. Click eye-shaped button to see their detailed scores.

6)	 Acknowledgements

In any publications that use the MAR scoring benchmark, please acknowledge the AAPM CT-MAR benchmark tool using the following sentences and references: *This work used the AAPM CT-MAR Grand Challenge benchmark tool [1, 2]. The AAPM CT-MAR Grand Challenge datasets were generated with the CatSim simulator in the open-source toolkit XCIST [3], using a hybrid data simulation framework that combines clinical images acquired at Massachusetts General Hospital and virtual metal objects.*

[1] AAPM CT Metal Artifact Reduction (CT-MAR) Grand Challenge: https://www.aapm.org/GrandChallenge/CT-MAR/

[2] AAPM CT Metal Artifact Reduction (CT-MAR) Grand Challenge Benchmark Tool: https://github.com/xcist/example/tree/main/AAPM_datachallenge/

[3] M. Wu, P. FitzGerald, J. Zhang, W.P. Segars, H. Yu, Y. Xu, B. De Man, "XCIST - an open access x-ray/CT simulation toolkit," Phys Med Biol. 2022 Sep 28;67(19)

