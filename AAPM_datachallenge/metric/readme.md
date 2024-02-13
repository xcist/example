**Usage**: `python AAPMmetric.py`

**Output**: `results.txt`, which stores the RMSE, SSIM, PSNR scores between MAR results and Target.

Please make sure that you have the following file structure when running this code, and also make sure that you follow the naming convention here. Note that the "MAR" folder holds your MAR results.
```
.
├── AAPMmetric.py
├── Baseline
│   ├── test_body_metalart_img44_512x512x1.raw
│   ├── ...
├── MAR
│   ├── test_body_metal_img44_512x512x1.raw
│   ├── ... (place your results here)
├── Mask
│   ├── test_body_metalonlymask_img44_512x512x1.raw
│   ├── ...
├── Target
│   ├── test_body_nometal_img44_512x512x1.raw
│   ├── ...
```

**How to install** This code needs `gecatsim`, `tqdm` and `skimage` packages installed first:
```
pip install gecatsim
pip install tqdm
pip install scikit-image
```


**How the code works** (you don’t need to read this to run the code, just for information purpose):
1. apply max FOV mask: all values outside of the mask will be 0
2. exclude metal region by setting pixels within the ground truth metal mask to 0
3. normalize all data with respect to the dynamic range of [-2000, 6000]; clip values beyond 6000 or below -2000
4. calculate PSNR and SSIM using the scikit-image package
