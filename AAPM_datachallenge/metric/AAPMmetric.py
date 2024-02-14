import numpy as np
import os
from tqdm import tqdm
import sys
import gecatsim as xc
from skimage.metrics import structural_similarity as SKssim
from skimage.metrics import peak_signal_noise_ratio as SKpsnr
import glob

rootpath = "./"

# ground truth path
GT_root = os.path.join(rootpath, "Target")
# MAR output path
my_root = os.path.join(rootpath, "MAR")
# metal mask path
mask_root = os.path.join(rootpath, "Mask")

def get_ring_mask():
    #mask = np.zeros((512, 512), dtype=int)
    mask_diam = 470 # in mm
    mask_diam_pix = mask_diam/(400/512) # in pixels
    indice = np.indices((512, 512))
    mask = (indice[0]-255.5)**2 + (indice[1]-255.5)**2 < mask_diam_pix**2/4.
    return mask

allpaths = glob.glob(GT_root+"/*512x512x1.raw")

FOVmask = get_ring_mask()

ssim_list, psnr_list = [], []
out_list = []
myimg_paths = []
rmse_list = []

for thissub in tqdm(allpaths):
    path_split = thissub.split('/')
    # check difference
    mypath = glob.glob(my_root+"/*"+path_split[-1].split('_')[3]+"*")[0]
    myimg_paths.append(mypath)
    myrecon = xc.rawread(mypath, [512, 512, 1], 'float')
    myrecon += 1000
    gt_path = thissub
    gtrecon = xc.rawread(gt_path, [512, 512, 1], 'float')
    gtrecon += 1000

    # metal mask (ground truth)
    gt_metal_mask_path = gt_path.replace("Target", "Mask").replace("nometal", "metalonlymask")
    gt_metal_mask = xc.rawread(gt_metal_mask_path, [512, 512, 1], 'float')
    gt_metal_mask = gt_metal_mask>0.5

    # calc PSNR
    myrecon[gt_metal_mask] = 0
    gtrecon[gt_metal_mask] = 0
    min2 = -2000
    max2 = 20000
    gtrecon = (gtrecon-min2)/(max2-min2)
    myrecon = (myrecon-min2)/(max2-min2)
    gtrecon = np.clip(gtrecon, 0, 1)[:,:,0]
    myrecon = np.clip(myrecon, 0, 1)[:,:,0]
    # only body has max FOV limit
    if 'body' in thissub:
        myrecon[FOVmask<0.5] = 0
        gtrecon[FOVmask<0.5] = 0

    if my_root == GT_root:
        psnr = SKpsnr(gtrecon, myrecon-0.001, data_range=1)
    else:
        psnr = SKpsnr(gtrecon, myrecon, data_range=1)
    # calc SSIM
    if my_root == GT_root:
        ssim = SKssim(gtrecon, myrecon-0.001, win_size=11, data_range=1, gaussian_weights=True)
    else:
        ssim = SKssim(gtrecon, myrecon, win_size=11, data_range=1, gaussian_weights=True)
    rmse = np.sqrt(np.mean((gtrecon-myrecon)**2))

    ssim_list.append(ssim)
    psnr_list.append(psnr)
    rmse_list.append(rmse)
    out_list.append([ssim, psnr])

with open('results.txt', 'w') as f:
    f.write("# rmse, ssim, psnr, gt_path, my_path\n")
    for i in range(len(psnr_list)):
        f.write("{:10.6f} {:10.6f} {:10.6f} {} {}\n".format(rmse_list[i], ssim_list[i], psnr_list[i], allpaths[i], myimg_paths[i]))
