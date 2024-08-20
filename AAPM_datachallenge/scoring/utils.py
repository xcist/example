# -*- coding: utf-8 -*-
"""
Metric evaluation program for the AAPM Grand Challenge. 
Image to be analyzed are stored in a subfolder "submission" in 
"folder_allresults". Data is submitted in seperate ".raw" files for each case, 
the data format is determined by the recon export (to be further specified by
Eri). 

@author: Nils Peters, npeters8@mgh.harvard.edu 
"""
import pdb
import os # to create file lists
import sys # for sanity check quit
import pickle # for dictionary import
import struct # to load raw data GE style
import numpy as np
from skimage.metrics import structural_similarity as ssim # for ssim calc
from sklearn.metrics import mean_absolute_error as mae # for mae calc
from sklearn.metrics import mean_squared_error as mse # for mse calc
from skimage.metrics import hausdorff_distance as hdd # for Hausdorff calc
from scipy.ndimage import rotate # for image rotation
from scipy import interpolate # for HLUT fit
from scipy import ndimage # for Sobel filter

# from PIL import Image 
debug = False # for prints

def rawread(fname, dataShape, dataType):
    # dataType is for numpy, ONLY allows: 'float'/'single', 'double', 'int'/'int32', 'uint'/'uint32', 'int8', 'int16' 
    #          they are single, double, int32, uin32, int8, int16
    with open(fname, 'rb') as fin:
        data = fin.read()
    
    # https://docs.python.org/3/library/struct.html
    switcher = {'float': ['f', 4, np.single], 
                'single': ['f', 4, np.single], 
                'double': ['d', 8, np.double], 
                'int': ['i', 4, np.int32], 
                'uint': ['I', 4, np.uint32],  
                'int32': ['i', 4, np.int32], 
                'uint32': ['I', 4, np.uint32], 
                'int8': ['b', 1, np.int8], 
                'int16': ['h', 2, np.int16]}
    fmt = switcher[dataType]
    data = struct.unpack("%d%s" % (len(data)/fmt[1], fmt[0]), data)
    
    data = np.array(data, dtype=fmt[2])
    if dataShape:
        data = data.reshape(dataShape)
    return data

def load_submitted_data(folderpath):
    '''
    Load all .raw files from specified folder and sort them in a dictionary,
    and check if the description .pdf or .docx is there in the submission dir,
    along with required .txt files 'results_body1.txt' and 'results_head1.txt'.
    input:  folderpath  - path of the folder containing the data
    output: data_img    - dictionary containing the image data 
    '''   
    files = [file for file in os.listdir(folderpath) if file.find(".raw") != -1]  
    doc_or_pdf_files = [file for file in os.listdir(folderpath) if file.endswith(".docx") or file.endswith(".pdf")]
    required_text_files = ['results_body1.txt', 'results_head1.txt']
    missing_files = []

    # Check for description file '.docx' or '.pdf'
    if not doc_or_pdf_files:
        missing_files.append("description .docx or .pdf file")

    # Check for required .txt files 'results_body1.txt', 'results_head1.txt'
    for fname in required_text_files:
        if not os.path.exists(os.path.join(folderpath, fname)):
            missing_files.append(fname)
    
    # Check if there are exactly 29 .raw files
    if len(files) != 29:
        missing_files.append("29 .raw files (only found {})".format(len(files)))
    
    # if missing_files:
    #     print("Error! The following required items are missing in the submission: " + ", ".join(missing_files) + ".")
    #     sys.exit()
    
    data_img = {} 
    for i in files:
        imgsize = os.path.getsize(os.path.join(folderpath, i))
        if imgsize != 512*512*4:
            print("Error! Size of {} is not right.\n".format(i))
            sys.exit()
        name = i[:-4]  # cut off the .raw from the filenames for dict
        file = rawread(os.path.join(folderpath, i), [512, 512], 'float')  # assuming rawread is defined elsewhere
        data_img[name] = file    
    return data_img

def sanity_check(data_img, data_ref): 
    '''
    Check if the loaded data has some format issues. Will stop code if yes.
    input:  data_img - 2D CT image submitted by participants
            data_ref - 2D CT reference image 
    '''  
    # Check if the right number of cases is imported. 
    if len(data_img) != 29:
        print('Your data dictionary does not contain the required 29 cases.')
        sys.exit()
        
    # Check if the right keys are used 
    for i in data_img.keys():
        if i not in ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
                     '11', '12', '13', '14', '15', '16', '17', '18', '19' , '20',
                     '21', '22', '23', '24', '25', '26', '27', '28', '29']:
            print('Keys in the data dictionary are not correct.')
            sys.exit()
    
### UPDATED: METRIC_SSIM # 05_21_2024
def metric_ssim(data_ref, data_img, mask, metalmask):
    '''
    Calculates a global and a masked structural similarity index
    input:  data_ref - 2D CT reference image without artifacts
            data_img - 2D CT image with MAR applied
            mask - mask in which the SSIM should be averaged
            metalmask - mask of the metal area to be excluded from ssim
    output: ssim_map - global SSIM and 2D array map with local values
            ssim_masked - mean value of ssim within the mask
    '''    

    # replace pixels within gt metal
    data_ref_masked = np.copy(data_ref)
    data_ref_masked[metalmask] = 100
    data_img_masked = np.copy(data_img)
    data_img_masked[metalmask] = 100 
    
    # Calculate  SSIM
    ssim_map = ssim(data_ref_masked, data_img_masked, sigma = 1.5, 
                    gaussian_weights = True, use_sample_covariance = False, 
                    data_range= 5000, full = True, grad = False)
    # Calculate SSIM within the defined mask
    ssim_masked = ssim_map[1][mask == True].mean()
    return ssim_map, ssim_masked




      
def metric_mean_error(m1, m2):
    '''
    Calculates the mean CT number difference between two images
    input:  m1 - 2D CT reference image without artifacts
            m2 - 2D CT image with MAR applied
    output: mean  - mean CT number difference
    '''    
    a = np.sum(m1)
    b = np.sum(m2)
    nonz = np.count_nonzero(m1)
    mean = (a-b) / nonz
    return mean

def metric_sobel(img_1, img_2, mask_bone, mask_soft):
    '''
    Calculates the Sobel magnitude for two different tissue regions
    input:  img_1       - 2D reference (baseline) image
            img_2       - 2D MAR reduced image
            mask_bone   - 2D mask over a soft tissue / bone edge region
            mask soft   - 2D mask over a soft / adipose / water edge region
            sobel_rel_bone - change in Sobel max magnitude in the bone tissue
    output: sobel_rel_soft - change in Sobel max magnitude in the soft tissue
            sobel_rel_avg - average of sobel_rel_soft and sobel_rel_bone
    '''  
    # For baseline image
    img_1_sobel_h = ndimage.sobel(img_1, 0)
    img_1_sobel_v = ndimage.sobel(img_1, 1)
    img_1_sobel_magn = np.sqrt(img_1_sobel_h**2 + img_1_sobel_v**2)
    
    # for MAR image
    img_2_sobel_h = ndimage.sobel(img_2, 0)
    img_2_sobel_v = ndimage.sobel(img_2, 1)
    img_2_sobel_magn = np.sqrt(img_2_sobel_h**2 + img_2_sobel_v**2)        

    percentile = 90
    max_sobel_img_1_soft = np.percentile(img_1_sobel_magn[mask_soft == 1], percentile)
    max_sobel_img_1_bone = np.percentile(img_1_sobel_magn[mask_bone == 1], percentile)    
    max_sobel_img_2_soft = np.percentile(img_2_sobel_magn[mask_soft == 1], percentile)
    max_sobel_img_2_bone = np.percentile(img_2_sobel_magn[mask_bone == 1], percentile)

    # max_sobel_img_1_soft = np.max(img_1_sobel_magn[mask_soft == 1])
    # max_sobel_img_1_bone = np.max(img_1_sobel_magn[mask_bone == 1])    
    # # max_sobel_img_2_soft = np.max(img_2_sobel_magn[mask_soft == 1])
    # # max_sobel_img_2_bone = np.max(img_2_sobel_magn[mask_bone == 1])
    if debug:
        print('################')
        print(max_sobel_img_1_bone)
        print(max_sobel_img_2_bone)
    
    sobel_rel_bone = max_sobel_img_2_bone / max_sobel_img_1_bone
    sobel_rel_soft = max_sobel_img_2_soft / max_sobel_img_1_soft
    sobel_rel_avg =  np.mean([sobel_rel_bone, sobel_rel_soft])
    
    return sobel_rel_bone, sobel_rel_soft, sobel_rel_avg

def metric_metalgeometry(img, metal_mask_ref, metalregion_mask, voxelsize, 
                         bonetrs, bonetrsdiff):
    '''
    Calculates relative volume change, dice coefficent and Hausdorff for metal
    input:  img       - 2D MAR reduced image
            metal_mask_ref 2D reference mask for the metal geometry
            metalregion_mask - Mask in which metal in img is determined
            voxelsize   - Voxelsize of the CT image [x,y,z]
            bontrs - patient-specific max bone value within metalregion_mask
            bonetrsdiff - offset over which everything is defined as metal
    output: metal_vol_rel - Metal volume change relative to reference
            dice_metal - Dice coefficient for metal geometry
            hdd_metal - Hausdorff distance for metal geometry
    '''          
    # Calculate initial metal area, in mm^2 
    metal_vol_ref = np.sum(metal_mask_ref) * voxelsize**2         
    
    ## Calculate metal volume within ROI around implant
    metal_mask = np.zeros((512,512))
    metal_mask[img > bonetrs + bonetrsdiff] = 1
    metal_mask[metalregion_mask == 0] = 0
        
    metal_vol  = (np.sum(metal_mask * voxelsize**2))  
    
    # Calculate relative metal volume change 
    metal_vol_rel = abs(1 - metal_vol / metal_vol_ref)
        
    ## Calculate the dice for metal
    # Calculate the intersection of the two metal masks
    metal_overlap = np.zeros((512,512))
        
    for m in range(0,512):
        for n in range(0,512):
            if metal_mask[m,n] == 1 and metal_mask_ref[m,n] == 1:
                metal_overlap[m,n] = 1

    dice_metal = (2* np.sum(metal_overlap * voxelsize**2) / 
                  (metal_vol_ref + metal_vol ))
        
    # Calculate Hausdorff for metal
    # hdd_metal = hdd(metal_mask, metal_mask_ref, method = 'standard')
    hdd_metal = hdd(metal_mask, metal_mask_ref)

    return metal_vol_rel, dice_metal, hdd_metal

def mask_tissueregion(ct, lowCTN, highCTN, mask_ext):
    '''
    Creates binary masks based on CT numbers for tissue regions within external
    input:  CT - 2D CT array on which the mask is to be created
            lowCTN - lowest CT number of the tissue region
            highCTN - highest CT number of the tissue region
            mask_ext - mask of external without the metal
    output: mask - 2D array with mask
    '''    
    mask = np.zeros ((512,512))
    mask[ct < highCTN] = 1
    mask[ct < lowCTN] = 0
    mask[mask_ext == 0] = 0
    return mask

def metric_bone_geometry(img_ref, img_mar, mask, bonetrs, voxelsize):
    '''
    Calculates relative volume change, dice coefficent and Hausdorff for metal
    input:  img_ref - 2D reference image
            img_mar - 2D MAR image
            mask - mask in which metrics are calculated
                    (here: external excluding the ground truth metal geometry)
            bontrs - patient-specific max bone value within metalregion_mask
            voxelsize   - Voxelsize of the CT image [x,y,z]
    output: bone_vol_rel - Bone volume change relative to reference
            dice_bone - Dice coefficient for metal geometry
            hdd_bone - Hausdorff distance for metal geometry
    '''          
    # Create tissue-region-specific masks
    # mask_lung_ref = mask_tissueregion(img_ref, -900, -150, mask)
    # mask_lung = mask_tissueregion(img_mar, -900, -150, mask)
        
    # mask_soft_ref = mask_tissueregion(img_ref, -150, 150, mask)
    # mask_soft = mask_tissueregion(img_mar, -150, 150, mask)
            
    mask_bone_ref = mask_tissueregion(img_ref, 150, 2500,  mask)
    mask_bone = mask_tissueregion(img_mar, 150, 2500, mask)
                
    # Calculate initial bone area excluding original metal geometry, in mm^2 
    bone_vol_ref = np.sum(mask_bone_ref) * voxelsize**2         
    
    # Calculate bone area, excluding original metal geometry 
    bone_vol = np.sum(mask_bone) * voxelsize**2      

    # Calculate relative change in bone volume
    bone_vol_rel = abs(1 - bone_vol / bone_vol_ref)
        
    ## Calculate the dice for bones
    # Calculate the intersection of the two bone masks

    bone_overlap = np.zeros((512,512))
        
    for m in range(0,512):
        for n in range(0,512):
            if mask_bone[m,n] == 1 and mask_bone_ref[m,n] == 1:
                bone_overlap[m,n] = 1        
  
    dice_bone = (2* np.sum(bone_overlap * voxelsize**2) / 
                 (bone_vol_ref + bone_vol))
        
    # Calculate Hausdorff for bone
    # hdd_bone = hdd(mask_bone_ref, mask_bone, method = 'standard')        
    hdd_bone = hdd(mask_bone_ref, mask_bone)

    return bone_vol_rel, dice_bone, hdd_bone

def hlut_apply(img):
    '''
    Applies a clinical CTN-to-SPR translation to the 2D CT image
    input:  img - 2D CT array on which to apply the HLUT
    output: SPR - 2D array with SPR values
    '''    
    hlut_x = [-1000, -200, -120,-20,50,100,140,2990,2995,999960]
    hlut_y = [0,0.788,0.897,1.003,1.038,1.084,1.097,2.732,3.200,3.200]

    hlut = interpolate.interp1d (hlut_x, hlut_y, bounds_error = False, 
                                 fill_value = 0)
    spr = hlut(img)
    return spr

def metric_protonrange(beamangle, img_1, img_2, mask_tumor, mask_external, 
                       voxelsize):
    '''
    Calculates the water-equivalent distance along a simplified beam path
    input:  beamangle - incoming beam angle. Zero is a beam from the left
            img_1 - 2D reference image 
            img_2 - 2D MAR image
            mask_tumor - Mask of the target to which the beam is integrated
            mask_external - mask of the patient (here: including metal!) 
            voxelsize - 
    output: spr_lineintg_ref - List of all SPR line integrals in reference img
            spr_lineintg - list of all SPR line integrals in MAR image
            spr_maxdicc - largest difference in range, in mm
            spr_maxdiff_rel - largest differencein range relative to max range
    '''   
    # Check if there is a beam angle, rotate images necessary
    if beamangle != 0: 
        img_1_rot = rotate(img_1, angle = beamangle, axes = (0,1), 
                             reshape = False)
        img_2_rot = rotate(img_2, angle = beamangle, axes = (0,1), 
                             reshape = False)
        tumor_rot = rotate(mask_tumor, angle = beamangle, axes=(0,1), 
                             reshape = False)
    else:
        img_1_rot = img_1
        img_2_rot = img_2
        tumor_rot = mask_tumor
    
    # Determine coordinates of the tumor volume
    tumor_coords = np.where(tumor_rot == True)
    
    # Determine coordinates of distal voxels of tumor volume, as seen by beam
    border_coordinates = []
    for j in range(min(tumor_coords[0]), max(tumor_coords[0])):
        subarr = tumor_rot[j,:]
        if any(subarr) == True:
            sublist = np.where(subarr == True)
            sublist_max = np.max(sublist)
            border_coordinates.append([j, sublist_max])

    # Create test array to check if border was created correctly
    # border_test = np.copy(img_1_rot)
    # for k in border_coordinates:
    #     x, y = k
    #     border_test[x,y] = 5000
    #     im_border = Image.fromarray(border_test)
    #     im_border.save('ctv_border_{}.tiff'.format(beamangle))    
    
    # Apply clinical HLUT to scans 
    spr_ref = hlut_apply(img_1_rot)
    spr_img = hlut_apply(img_2_rot)
    
    # Apply external to images
    spr_ref[mask_external == 0] = 0
    spr_img[mask_external == 0] = 0

    # Initialize line integral lists
    spr_lineintg_ref = []
    spr_lineintg = []

    # Calculate all line integrals, 1 voxel apart
    for j in range(0, len(border_coordinates)):
        x, y = border_coordinates[j]
        spr_lineintg_ref.append(voxelsize * np.sum(spr_ref[x, :y]))

        spr_lineintg.append(voxelsize * np.sum(spr_img[x, :y]))        
    
    # Calculate difference between the individual proton ranges
    spr_lineintg_diff = np.subtract(spr_lineintg_ref, spr_lineintg)
    
    # Determine maximal difference, absolute and relative to maximum in ref
    spr_maxdiff = np.max(np.abs(spr_lineintg_diff))
    spr_maxdiff_rel = spr_maxdiff / (np.max(spr_lineintg_ref))
        
    return (spr_lineintg_ref, spr_lineintg, spr_lineintg_diff, spr_maxdiff, 
            spr_maxdiff_rel)

def metric_streak(img_mar, img_ref, roi_amplitude):
    '''
    Estimate the amplitude of the streaks within a given ROI, defined as the 
    distance between the average of the highest and lowest 5% of the data, 
    respectively. 
    input:  img_mar - 2D MAR image
            roi_amplitude - Mask of region with streak artifacts
    output: streak_distance - determined distance of the values, in HU
    ''' 
    perc = 5
    
    diff_image = img_mar - img_ref
    
    # Create list of all pixels within predefined streak region
    streak_values = diff_image[roi_amplitude == True]
    
    # Sort list to determine average within upper and lower 10% of values
    streak_values.sort()
    cutoff = int((len(streak_values)/(100/perc)))  
    
    if debug:
        print(len(streak_values))
        print(cutoff)
    
    avg_lowerten = np.mean(streak_values[:cutoff])
    avg_higherten = np.mean(streak_values[(len(streak_values)-cutoff):])
    
    # Calculate absolute distance between the two 
    streak_distance = np.abs(avg_lowerten - avg_higherten)

    return streak_distance

def metric_lesion(lesion_mask, img_ref, img_mar):
    # Define size of border around lesion for metric 
    border = 2
    
    # Define analysis area within the image
    xmin = np.min(np.where(lesion_mask == True)[0])
    xmax = np.max(np.where(lesion_mask == True)[0])
    ymin = np.min(np.where(lesion_mask == True)[1])
    ymax = np.max(np.where(lesion_mask == True)[1])        

    # Define and normalize lesion template
    T_1 = lesion_mask[xmin - border:xmax + border, ymin - border:ymax + border]
    T_2 = T_1 - np.mean(T_1)   
    
    # Calculate inner product of         
    I_1 = img_mar[xmin - border:xmax + border, ymin - border:ymax + border]
    I_1_ref = img_ref[xmin - border:xmax + border, ymin - border:ymax + border]

    # Calculate detectability metrics for MAR and ref
    detec = np.sum(T_2* I_1)
    detec_ref = np.sum(T_2* I_1_ref)        

    # Shift and normalize accordingly
    detec_shifted = (detec+detec_ref) / detec_ref
        
                
    return detec_shifted

def scoring_axes(results_submission):
    '''
    Manually define the fit parameters to assign a score to each result
    input:  results_submission - dictionary with results for all cases
    output: scory_y - score values for fit 
            fit_x_dict - dictionary containing the x values for the fit
    '''   
    score_y = [0, 0, 1, 2, 3, 4, 4]

    fit_x_dict = {}
    for i in results_submission.keys():
        fit_x_dict[i] = {}
    
    fit_x_dict['01']['ctn_acc']  = [0, 0,  15, 30, 45, 60, 999999] #
    fit_x_dict['02']['ctn_acc']  = [0, 0,  30, 60, 90, 120, 99999] #
    fit_x_dict['03']['ctn_acc']  = [0, 0, 40, 80, 120, 160, 99999]#
    fit_x_dict['04']['ctn_acc']  = [0, 0, 20, 40, 60, 80,  99999] #
    fit_x_dict['05']['ctn_acc']  = [0, 0, 20, 40, 60, 80,  99999] #
    fit_x_dict['06']['ctn_acc']  = [0, 0, 35, 70, 105, 140, 99999] #
    fit_x_dict['07']['ctn_acc']  = [0, 0,  50, 100, 150, 200, 99999] #
    fit_x_dict['08']['ctn_acc']  = [0, 0,  40, 80, 120, 160, 99999] #
    fit_x_dict['09']['ctn_acc']  = [0, 0,  50, 100, 150, 200, 99999] #
    fit_x_dict['10']['ctn_acc']  = [0, 0, 65, 130, 195, 260, 99999] # 
    fit_x_dict['11']['ctn_acc']  = [0, 0, 75, 150, 225, 300, 99999] # 
    fit_x_dict['12']['ctn_acc']  = [0, 0,  15, 30, 45, 60, 99999] #
    fit_x_dict['13']['ctn_acc']  = [0, 0, 25, 50, 75, 100, 99999] #
    fit_x_dict['14']['ctn_acc']  = [0, 0, 20, 40, 60, 80,  99999] #
    fit_x_dict['15']['ctn_acc']  = [0, 0,  50, 100, 150, 200, 99999] #
    fit_x_dict['16']['ctn_acc']  = [0, 0,  50, 100, 150, 200, 99999] #
    fit_x_dict['17']['ctn_acc']  = [0, 0, 85, 170, 255, 340, 99999] #
    fit_x_dict['18']['ctn_acc']  = [0, 0,  50, 100, 150, 200, 99999] #
    fit_x_dict['19']['ctn_acc']  = [0, 0,  50, 100, 150, 200, 99999] #
    fit_x_dict['20']['ctn_acc']  = [0, 0, 65, 130, 195, 260, 99999] # 
    fit_x_dict['21']['ctn_acc']  = [0, 0, 150, 300, 450, 600, 99999]#
    fit_x_dict['22']['ctn_acc']  = [0, 0, 150, 300, 450, 600, 99999]#
    fit_x_dict['23']['ctn_acc']  = [0, 0, 65, 130, 195, 260, 99999] # 
    fit_x_dict['24']['ctn_acc']  = [0, 0, 85, 170, 255, 340, 99999] #
    fit_x_dict['25']['ctn_acc']  = [0, 0, 150, 300, 450, 600, 99999]#
    fit_x_dict['26']['ctn_acc']  = [0, 0, 150, 300, 450, 600, 99999]#
    fit_x_dict['27']['ctn_acc']  = [0, 0, 25, 50, 75, 100, 99999] #
    fit_x_dict['28']['ctn_acc']  = [0, 0, 65, 130, 195, 260, 99999] # 
    fit_x_dict['29']['ctn_acc']  = [0, 0, 85, 170, 255, 340, 99999] #


    fit_x_dict['01']['noise']  = [0, 8, 10, 12, 14, 16, 9999]
    fit_x_dict['02']['noise']  = [0, 10, 12, 14, 16, 18, 9999]
    fit_x_dict['03']['noise']  = [0, 12, 15, 18, 21, 24, 9999]
    fit_x_dict['04']['noise']  = [0, 36, 42, 46, 50, 54, 9999]
    fit_x_dict['05']['noise']  = [0, 36, 42, 46, 50, 54, 9999]
    fit_x_dict['06']['noise']  = [0, 12, 15, 18, 21, 24, 9999]
    fit_x_dict['07']['noise']  = [0, 12, 15, 18, 21, 24, 9999]
    fit_x_dict['08']['noise']  = [0, 12, 15, 18, 21, 24, 9999]
    fit_x_dict['09']['noise']  = [0, 12, 15, 18, 21, 24, 9999]
    fit_x_dict['10']['noise']  = [0, 12, 15, 18, 21, 24, 9999]
    fit_x_dict['11']['noise']  = [0, 18, 22, 26, 30, 34, 9999]
    fit_x_dict['12']['noise']  = [0, 18, 22, 26, 30, 34, 9999]
    fit_x_dict['13']['noise']  = [0, 22, 28, 33, 38, 42, 9999]
    fit_x_dict['14']['noise']  = [0, 18, 22, 26, 30, 34, 9999]
    fit_x_dict['15']['noise']  = [0, 12, 15, 18, 21, 24, 9999]
    fit_x_dict['16']['noise']  = [0, 12, 15, 18, 21, 24, 9999]
    fit_x_dict['17']['noise']  = [0, 12, 15, 18, 21, 24, 9999]
    fit_x_dict['18']['noise']  = [0, 8, 10, 12, 14, 16, 9999]
    fit_x_dict['19']['noise']  = [0, 10, 12, 14, 16, 18, 9999]
    fit_x_dict['20']['noise']  = [0, 10, 12, 14, 16, 18, 9999]
    fit_x_dict['21']['noise']  = [0, 20, 25, 30, 35, 40, 9999]
    fit_x_dict['22']['noise']  = [0, 18, 22, 26, 30, 34, 9999]
    fit_x_dict['23']['noise']  = [0, 48, 45, 64, 72, 80, 9999]
    fit_x_dict['24']['noise']  = [0, 38, 45, 52, 59, 66, 9999]
    fit_x_dict['25']['noise']  = [0, 20, 25, 30, 35, 40, 9999]
    fit_x_dict['26']['noise']  = [0, 18, 22, 26, 30, 34, 9999]
    fit_x_dict['27']['noise']  = [0, 38, 45, 52, 59, 66, 9999]
    fit_x_dict['28']['noise']  = [0, 32, 38, 44, 50, 56, 9999]
    fit_x_dict['29']['noise']  = [0, 32, 38, 44, 50, 56, 9999]
    # fit_x_dict['30']['noise']  = [0, 26, 31, 36, 41, 46, 9999]

    

    fit_x_dict['01']['sobel_rel_avg']  = [10, 1, 0.94, 0.85, 0.67, 0.56, 0]
    fit_x_dict['02']['sobel_rel_avg']  = fit_x_dict['01']['sobel_rel_avg']
    fit_x_dict['03']['sobel_rel_avg']  = fit_x_dict['01']['sobel_rel_avg']
    fit_x_dict['04']['sobel_rel_avg']  = fit_x_dict['01']['sobel_rel_avg']
    fit_x_dict['05']['sobel_rel_avg']  = fit_x_dict['01']['sobel_rel_avg']
    fit_x_dict['06']['sobel_rel_avg']  = fit_x_dict['01']['sobel_rel_avg']
    fit_x_dict['07']['sobel_rel_avg']  = fit_x_dict['01']['sobel_rel_avg']
    fit_x_dict['08']['sobel_rel_avg']  = fit_x_dict['01']['sobel_rel_avg']
    fit_x_dict['09']['sobel_rel_avg']  = fit_x_dict['01']['sobel_rel_avg']
    fit_x_dict['10']['sobel_rel_avg']  = fit_x_dict['01']['sobel_rel_avg']
    fit_x_dict['11']['sobel_rel_avg']  = fit_x_dict['01']['sobel_rel_avg']
    fit_x_dict['12']['sobel_rel_avg']  = fit_x_dict['01']['sobel_rel_avg']
    fit_x_dict['13']['sobel_rel_avg']  = fit_x_dict['01']['sobel_rel_avg']
    fit_x_dict['14']['sobel_rel_avg']  = fit_x_dict['01']['sobel_rel_avg']
    fit_x_dict['15']['sobel_rel_avg']  = fit_x_dict['01']['sobel_rel_avg']
    fit_x_dict['16']['sobel_rel_avg']  = fit_x_dict['01']['sobel_rel_avg']
    fit_x_dict['17']['sobel_rel_avg']  = fit_x_dict['01']['sobel_rel_avg']
    fit_x_dict['18']['sobel_rel_avg']  = fit_x_dict['01']['sobel_rel_avg']
    fit_x_dict['19']['sobel_rel_avg']  = fit_x_dict['01']['sobel_rel_avg']
    fit_x_dict['20']['sobel_rel_avg']  = fit_x_dict['01']['sobel_rel_avg']
    fit_x_dict['21']['sobel_rel_avg']  = fit_x_dict['01']['sobel_rel_avg']
    fit_x_dict['22']['sobel_rel_avg']  = fit_x_dict['01']['sobel_rel_avg']
    fit_x_dict['23']['sobel_rel_avg']  = fit_x_dict['01']['sobel_rel_avg']
    fit_x_dict['24']['sobel_rel_avg']  = fit_x_dict['01']['sobel_rel_avg']
    fit_x_dict['25']['sobel_rel_avg']  = fit_x_dict['01']['sobel_rel_avg']
    fit_x_dict['26']['sobel_rel_avg']  = fit_x_dict['01']['sobel_rel_avg']
    fit_x_dict['27']['sobel_rel_avg']  = fit_x_dict['01']['sobel_rel_avg']
    fit_x_dict['28']['sobel_rel_avg']  = fit_x_dict['01']['sobel_rel_avg']
    fit_x_dict['29']['sobel_rel_avg']  = fit_x_dict['01']['sobel_rel_avg']
    # fit_x_dict['30']['sobel_rel_avg']  = fit_x_dict['01']['sobel_rel_avg']



    fit_x_dict['01']['streak']  = [0, 0, 20, 40, 60, 80, 99999] 
    fit_x_dict['02']['streak']  = [0, 0, 50, 100, 150, 200, 99999]
    fit_x_dict['03']['streak']  = [0, 0, 150, 300, 450, 600, 99999]
    fit_x_dict['04']['streak']  = [0, 0, 100, 200, 300, 400, 99999]
    fit_x_dict['05']['streak']  = [0, 0, 150, 300, 450, 600, 99999]
    fit_x_dict['06']['streak']  = [0, 0, 40, 80, 120, 160, 99999]
    fit_x_dict['07']['streak']  = [0, 0, 50, 100, 150, 200, 99999]
    fit_x_dict['08']['streak']  = [0, 0, 40, 80, 120, 160, 99999]
    fit_x_dict['09']['streak']  = [0, 0, 40, 80, 120, 160, 99999]
    fit_x_dict['10']['streak']  = [0, 0, 50, 100, 150, 200, 99999]
    fit_x_dict['11']['streak']  = [0, 0, 150, 300, 450, 600, 99999]
    fit_x_dict['12']['streak']  = [0, 0, 100, 200, 300, 400, 99999]
    fit_x_dict['13']['streak']  = [0, 0, 150, 300, 450, 600, 99999]
    fit_x_dict['14']['streak']  = [0, 0, 150, 300, 450, 600, 99999]
    fit_x_dict['15']['streak']  = [0, 0, 150, 300, 450, 600, 99999]
    fit_x_dict['16']['streak']  = [0, 0, 100, 200, 300, 400, 99999]
    fit_x_dict['17']['streak']  = [0, 0, 150, 300, 450, 600, 99999]
    fit_x_dict['18']['streak']  = [0, 0, 40, 80, 120, 160 , 99999]
    fit_x_dict['19']['streak']  = [0, 0, 35, 60, 105, 140, 99999]
    fit_x_dict['20']['streak']  = [0, 0, 100, 200, 300, 400, 99999]
    fit_x_dict['21']['streak']  = [0, 0, 150, 300, 450, 600, 99999]
    fit_x_dict['22']['streak']  = [0, 0, 150, 300, 450, 600, 99999]
    fit_x_dict['23']['streak']  = [0, 0, 150, 300, 450, 600, 99999]
    fit_x_dict['24']['streak']  = [0, 0, 150, 300, 450, 600, 99999]
    fit_x_dict['25']['streak']  = [0, 0, 150, 300, 450, 600, 99999]
    fit_x_dict['26']['streak']  = [0, 0, 150, 300, 450, 600, 99999]
    fit_x_dict['27']['streak']  = [0, 0, 150, 300, 450, 600, 99999]
    fit_x_dict['28']['streak']  = [0, 0, 150, 300, 450, 600, 99999]
    fit_x_dict['29']['streak']  = [0, 0, 150, 300, 450, 600, 99999]
    # fit_x_dict['30']['streak']  = [0, 0, 100, 200, 300, 400, 99999]
    


    fit_x_dict['01']['ssim']  = [20, 1,  0.99, 0.98, 0.97, 0.96, -20]
    fit_x_dict['02']['ssim']  = [20, 1, .98, .96, .94, .92, -20]
    fit_x_dict['03']['ssim']  = [20, 1, .92, .84, .76, .68 , -20]
    fit_x_dict['04']['ssim']  = [20, 1, .96, .93, .90, .87 , -20]
    fit_x_dict['05']['ssim']  = [20, 1, .96, .93, .90, .87 , -20]
    fit_x_dict['06']['ssim']  = [20, 1,  0.99, 0.98, 0.97, 0.96, -20]
    fit_x_dict['07']['ssim']  = [20, 1,  0.99, 0.98, 0.97, 0.96, -20]
    fit_x_dict['08']['ssim']  = [20, 1,  0.99, 0.98, 0.97, 0.96, -20]
    fit_x_dict['09']['ssim']  = [20, 1,  0.99, 0.98, 0.97, 0.96, -20]
    fit_x_dict['10']['ssim']  = [20, 1, .97, .95, .92, .89 , -20]
    fit_x_dict['11']['ssim']  = [20, 1, .94, .89, .84, .79 , -20]
    fit_x_dict['12']['ssim']  = [20, 1, .98, .96, .94, .92, -20]
    fit_x_dict['13']['ssim']  = [20, 1, .95, .90, .85, .80, -20]
    fit_x_dict['14']['ssim']  = [20, 1, .98, .96, .94, .92, -20]
    fit_x_dict['15']['ssim']  = [20, 1, .95, .90, .85, .80, -20]
    fit_x_dict['16']['ssim']  = [20, 1, .96, .92, .89, .86, -20]
    fit_x_dict['17']['ssim']  = [20, 1, .92, .84, .76, .68, -20]
    fit_x_dict['18']['ssim']  = [20, 1, .98, .96, .94, .92, -20]
    fit_x_dict['19']['ssim']  = [20, 1, .99, .98, .97, .96, -20]
    fit_x_dict['20']['ssim']  = [20, 1, .97, .94, .91, .88, -20]
    fit_x_dict['21']['ssim']  = [20, 1, .90, .80, .70, .60, -20] 
    fit_x_dict['22']['ssim']  = [20, 1, .91, .82, .73, .64, -20] 
    fit_x_dict['23']['ssim']  = [20, 1, .94, .89, .84, .79, -20]
    fit_x_dict['24']['ssim']  = [20, 1, .91, .82, .73, .64, -20] 
    fit_x_dict['25']['ssim']  = [20, 1, .90, .80, .70, .60, -20]  
    fit_x_dict['26']['ssim']  = [20, 1, .91, .82, .73, .64, -20] 
    fit_x_dict['27']['ssim']  = [20, 1, .94, .89, .84, .79, -20]
    fit_x_dict['28']['ssim']  = [20, 1, .94, .89, .84, .79, -20]
    fit_x_dict['29']['ssim']  = [20, 1, .91, .82, .73, .64, -20]



    fit_x_dict['01']['metal_vol_rel']  = [0, 0, 0.35, 0.70, 1.05, 1.40, 9999]
    fit_x_dict['02']['metal_vol_rel']  = [0, 0, 0.50, 1.00, 1.50, 2.00, 9999]
    fit_x_dict['03']['metal_vol_rel']  = [0, 0, 0.05, 0.10, 0.15, 0.20, 9999]
    fit_x_dict['04']['metal_vol_rel']  = [0, 0, 0.25, 0.50, 0.75, 1.00, 9999]
    fit_x_dict['05']['metal_vol_rel']  = [0, 0, 0.20, 0.40, 0.60, 0.80, 9999]
    fit_x_dict['06']['metal_vol_rel']  = [0, 0, 0.25, 0.50, 0.75, 1.00, 9999]
    fit_x_dict['07']['metal_vol_rel']  = [0, 0, 0.25, 0.50, 0.75, 1.00, 9999]
    fit_x_dict['08']['metal_vol_rel']  = [0, 0, 0.15, 0.30, 0.45, 0.60, 9999]
    fit_x_dict['09']['metal_vol_rel']  = [0, 0, 0.25, 0.50, 0.75, 1.00, 9999]
    fit_x_dict['10']['metal_vol_rel']  = [0, 0, 0.25, 0.50, 0.75, 1.00, 9999]
    fit_x_dict['11']['metal_vol_rel']  = [0, 0, 0.30, 0.60, 0.90, 1.20, 9999]
    fit_x_dict['12']['metal_vol_rel']  = [0, 0, 0.25, 0.50, 0.75, 1.00, 9999]
    fit_x_dict['13']['metal_vol_rel']  = [0, 0, 0.25, 0.50, 0.75, 1.00, 9999]
    fit_x_dict['14']['metal_vol_rel']  = [0, 0, 0.35, 0.70, 1.05, 1.40, 9999]
    fit_x_dict['15']['metal_vol_rel']  = [0, 0, 0.15, 0.30, 0.45, 0.60, 9999]
    fit_x_dict['16']['metal_vol_rel']  = [0, 0, 0.20, 0.40, 0.60, 0.80, 9999]
    fit_x_dict['17']['metal_vol_rel']  = [0, 0, 0.25, 0.50, 0.75, 1.00, 9999]
    fit_x_dict['18']['metal_vol_rel']  = [0, 0, 0.25, 0.50, 0.75, 1.00, 9999]
    fit_x_dict['19']['metal_vol_rel']  = [0, 0, 0.25, 0.50, 0.75, 1.00, 9999]
    fit_x_dict['20']['metal_vol_rel']  = [0, 0, 0.25, 0.50, 0.75, 1.00, 9999]
    fit_x_dict['21']['metal_vol_rel']  = [0, 0, 0.70, 1.40, 2.10, 2.80, 9999]
    fit_x_dict['22']['metal_vol_rel']  = [0, 0, 0.60, 1.20, 1.80, 2.40, 9999]
    fit_x_dict['23']['metal_vol_rel']  = [0, 0, 0.13, 0.25, 0.38, 0.50, 9999]
    fit_x_dict['24']['metal_vol_rel']  = [0, 0, 0.25, 0.50, 0.75, 1.00, 9999]
    fit_x_dict['25']['metal_vol_rel']  = [0, 0, 0.25, 0.50, 0.75, 1.00, 9999]
    fit_x_dict['26']['metal_vol_rel']  = [0, 0, 0.25, 0.50, 0.75, 1.00, 9999]
    fit_x_dict['27']['metal_vol_rel']  = [0, 0, 0.07, 0.14, 0.21, 0.28, 9999]
    fit_x_dict['28']['metal_vol_rel']  = [0, 0, 0.07, 0.14, 0.21, 0.28, 9999]
    fit_x_dict['29']['metal_vol_rel']  = [0, 0, 0.07, 0.14, 0.21, 0.28, 9999]
    # fit_x_dict['30']['metal_vol_rel']  = [0, 0, 0.25, 0.50, 0.75, 1.00, 9999]



    fit_x_dict['01']['dice_metal']  = [1, 1, 0.85, 0.70, 0.55, 0.40, 0]
    fit_x_dict['02']['dice_metal']  = [1, 1, 0.80, 0.60, 0.50, 0.40, 0]
    fit_x_dict['03']['dice_metal']  = [1, 1, 0.97, 0.95, 0.93, 0.90, 0]
    fit_x_dict['04']['dice_metal']  = [1, 1, 0.95, 0.90, 0.85, 0.80, 0]
    fit_x_dict['05']['dice_metal']  = [1, 1, 0.80, 0.60, 0.50, 0.40, 0]
    fit_x_dict['06']['dice_metal']  = [1, 1, 0.80, 0.60, 0.50, 0.40, 0]
    fit_x_dict['07']['dice_metal']  = [1, 1, 0.80, 0.60, 0.50, 0.40, 0]
    fit_x_dict['08']['dice_metal']  = [1, 1, 0.80, 0.60, 0.50, 0.40, 0]
    fit_x_dict['09']['dice_metal']  = [1, 1, 0.85, 0.70, 0.55, 0.40, 0]
    fit_x_dict['10']['dice_metal']  = [1, 1, 0.80, 0.60, 0.50, 0.40, 0]
    fit_x_dict['11']['dice_metal']  = [1, 1, 0.90, 0.80, 0.70, 0.60, 0]
    fit_x_dict['12']['dice_metal']  = [1, 1, 0.90, 0.80, 0.70, 0.60, 0]
    fit_x_dict['13']['dice_metal']  = [1, 1, 0.90, 0.80, 0.70, 0.60, 0]
    fit_x_dict['14']['dice_metal']  = [1, 1, 0.85, 0.70, 0.55, 0.40, 0]
    fit_x_dict['15']['dice_metal']  = [1, 1, 0.95, 0.90, 0.85, 0.80, 0]
    fit_x_dict['16']['dice_metal']  = [1, 1, 0.90, 0.80, 0.70, 0.60, 0]
    fit_x_dict['17']['dice_metal']  = [1, 1, 0.90, 0.80, 0.70, 0.60, 0]
    fit_x_dict['18']['dice_metal']  = [1, 1, 0.85, 0.70, 0.55, 0.40, 0]
    fit_x_dict['19']['dice_metal']  = [1, 1, 0.85, 0.70, 0.55, 0.40, 0]
    fit_x_dict['20']['dice_metal']  = [1, 1, 0.85, 0.70, 0.55, 0.40, 0]
    fit_x_dict['21']['dice_metal']  = [1, 1, 0.80, 0.60, 0.50, 0.40, 0]
    fit_x_dict['22']['dice_metal']  = [1, 1, 0.80, 0.60, 0.50, 0.40, 0]
    fit_x_dict['23']['dice_metal']  = [1, 1, 0.95, 0.90, 0.85, 0.80, 0]
    fit_x_dict['24']['dice_metal']  = [1, 1, 0.85, 0.70, 0.55, 0.40, 0]
    fit_x_dict['25']['dice_metal']  = [1, 1, 0.80, 0.60, 0.50, 0.40, 0]
    fit_x_dict['26']['dice_metal']  = [1, 1, 0.80, 0.60, 0.50, 0.40, 0]
    fit_x_dict['27']['dice_metal']  = [1, 1, 0.95, 0.90, 0.85, 0.80, 0]
    fit_x_dict['28']['dice_metal']  = [1, 1, 0.95, 0.90, 0.85, 0.80, 0]
    fit_x_dict['29']['dice_metal']  = [1, 1, 0.95, 0.90, 0.85, 0.80, 0]
    # fit_x_dict['30']['dice_metal']  = [1, 1, 0.97, 0.95, 0.93, 0.90, 0]


    fit_x_dict['01']['bone_vol_rel']  = [0, 0, .01, .02, .03, 0.04, 9999]
    fit_x_dict['02']['bone_vol_rel']  = [0, 0, .04, .08, .10, 0.12, 9999]
    fit_x_dict['03']['bone_vol_rel']  = [0, 0, .05, .10, .13, 0.15, 9999]
    fit_x_dict['04']['bone_vol_rel']  = [0, 0, .01, .02, .03, 0.04, 9999]
    fit_x_dict['05']['bone_vol_rel']  = [0, 0, .01, .02, .03, 0.04, 9999]
    fit_x_dict['06']['bone_vol_rel']  = [0, 0, .10, .20, .25, 0.30, 9999]
    fit_x_dict['07']['bone_vol_rel']  = [0, 0, .10, .20, .25, 0.30, 9999]
    fit_x_dict['08']['bone_vol_rel']  = [0, 0, .10, .20, .25, 0.30, 9999]
    fit_x_dict['09']['bone_vol_rel']  = [0, 0, .10, .20, .25, 0.30, 9999]
    fit_x_dict['10']['bone_vol_rel']  = [0, 0, .15, .30, .35, 0.40, 9999]
    fit_x_dict['11']['bone_vol_rel']  = [0, 0, .02, .04, .06, 0.08, 9999]
    fit_x_dict['12']['bone_vol_rel']  = [0, 0, .02, .04, .06, 0.08, 9999]
    fit_x_dict['13']['bone_vol_rel']  = [0, 0, .04, .08, .10, 0.12, 9999]
    fit_x_dict['14']['bone_vol_rel']  = [0, 0, .02, .04, .06, 0.08, 9999]
    fit_x_dict['15']['bone_vol_rel']  = [0, 0, .10, .20, .25, 0.30, 9999]
    fit_x_dict['16']['bone_vol_rel']  = [0, 0, .10, .20, .25, 0.30, 9999]
    fit_x_dict['17']['bone_vol_rel']  = [0, 0, .15, .30, .35, 0.40, 9999]
    fit_x_dict['18']['bone_vol_rel']  = [0, 0, .10, .20, .25, 0.30, 9999]
    fit_x_dict['19']['bone_vol_rel']  = [0, 0, .05, .10, .13, 0.15, 9999]
    fit_x_dict['20']['bone_vol_rel']  = [0, 0, .15, .30, .35, 0.40, 9999]
    fit_x_dict['21']['bone_vol_rel']  = [0, 0, .02, .04, .06, 0.08, 9999]
    fit_x_dict['22']['bone_vol_rel']  = [0, 0, .04, .08, .10, 0.12, 9999]
    fit_x_dict['23']['bone_vol_rel']  = [0, 0, .02, .04, .06, 0.08, 9999]
    fit_x_dict['24']['bone_vol_rel']  = [0, 0, .02, .04, .06, 0.08, 9999]
    fit_x_dict['25']['bone_vol_rel']  = [0, 0, .02, .04, .06, 0.08, 9999]
    fit_x_dict['26']['bone_vol_rel']  = [0, 0, .04, .08, .10, 0.12, 9999]
    fit_x_dict['27']['bone_vol_rel']  = [0, 0, .05, .10, .13, 0.15, 9999]
    fit_x_dict['28']['bone_vol_rel']  = [0, 0, .02, .04, .06, 0.08, 9999]
    fit_x_dict['29']['bone_vol_rel']  = [0, 0, .10, .20, .25, 0.30, 9999]
    # fit_x_dict['30']['bone_vol_rel']  = [0, 0, .05, .10, .13, 0.15, 9999]



    fit_x_dict['01']['dice_bone']  = [1, 1, 0.99, 0.98, 0.97, 0.96, 0]
    fit_x_dict['02']['dice_bone']  = [1, 1, 0.98, 0.96, 0.94, 0.92, 0]
    fit_x_dict['03']['dice_bone']  = [1, 1, 0.90, 0.80, 0.70, 0.60, 0]
    fit_x_dict['04']['dice_bone']  = [1, 1, 0.95, 0.90, 0.85, 0.80, 0]
    fit_x_dict['05']['dice_bone']  = [1, 1, 0.95, 0.90, 0.85, 0.80, 0]
    fit_x_dict['06']['dice_bone']  = [1, 1, 0.95, 0.90, 0.85, 0.80, 0]
    fit_x_dict['07']['dice_bone']  = [1, 1, 0.95, 0.90, 0.85, 0.80, 0]
    fit_x_dict['08']['dice_bone']  = [1, 1, 0.95, 0.90, 0.85, 0.80, 0]
    fit_x_dict['09']['dice_bone']  = [1, 1, 0.95, 0.90, 0.85, 0.80, 0]
    fit_x_dict['10']['dice_bone']  = [1, 1, 0.90, 0.80, 0.70, 0.60, 0]
    fit_x_dict['11']['dice_bone']  = [1, 1, 0.95, 0.90, 0.85, 0.80, 0]
    fit_x_dict['12']['dice_bone']  = [1, 1, 0.98, 0.96, 0.94, 0.92, 0]
    fit_x_dict['13']['dice_bone']  = [1, 1, 0.95, 0.90, 0.85, 0.80, 0]
    fit_x_dict['14']['dice_bone']  = [1, 1, 0.95, 0.90, 0.85, 0.80, 0]
    fit_x_dict['15']['dice_bone']  = [1, 1, 0.95, 0.90, 0.85, 0.80, 0]
    fit_x_dict['16']['dice_bone']  = [1, 1, 0.95, 0.90, 0.85, 0.80, 0]
    fit_x_dict['17']['dice_bone']  = [1, 1, 0.90, 0.80, 0.70, 0.60, 0]
    fit_x_dict['18']['dice_bone']  = [1, 1, 0.95, 0.90, 0.85, 0.80, 0]
    fit_x_dict['19']['dice_bone']  = [1, 1, 0.95, 0.90, 0.85, 0.80, 0]
    fit_x_dict['20']['dice_bone']  = [1, 1, 0.95, 0.90, 0.85, 0.80, 0]
    fit_x_dict['21']['dice_bone']  = [1, 1, 0.85, 0.70, 0.60, 0.50, 0]
    fit_x_dict['22']['dice_bone']  = [1, 1, 0.85, 0.70, 0.60, 0.50, 0]
    fit_x_dict['23']['dice_bone']  = [1, 1, 0.90, 0.80, 0.70, 0.60, 0]
    fit_x_dict['24']['dice_bone']  = [1, 1, 0.85, 0.70, 0.60, 0.50, 0]
    fit_x_dict['25']['dice_bone']  = [1, 1, 0.85, 0.70, 0.60, 0.50, 0]
    fit_x_dict['26']['dice_bone']  = [1, 1, 0.85, 0.70, 0.60, 0.50, 0]
    fit_x_dict['27']['dice_bone']  = [1, 1, 0.85, 0.70, 0.60, 0.50, 0]
    fit_x_dict['28']['dice_bone']  = [1, 1, 0.90, 0.80, 0.70, 0.60, 0]
    fit_x_dict['29']['dice_bone']  = [1, 1, 0.85, 0.70, 0.60, 0.50, 0]
    # fit_x_dict['30']['dice_bone']  = [1, 1, 0.95, 0.90, 0.85, 0.80, 0]




    fit_x_dict['01']['spr_maxdiff_rel']  = [0, 0, 0.005, 0.01, 0.015, 0.020, 9999]
    fit_x_dict['02']['spr_maxdiff_rel']  = fit_x_dict['01']['spr_maxdiff_rel']
    fit_x_dict['03']['spr_maxdiff_rel']  = fit_x_dict['01']['spr_maxdiff_rel']
    fit_x_dict['04']['spr_maxdiff_rel']  = fit_x_dict['01']['spr_maxdiff_rel']
    fit_x_dict['05']['spr_maxdiff_rel']  = fit_x_dict['01']['spr_maxdiff_rel']
    fit_x_dict['06']['spr_maxdiff_rel']  = fit_x_dict['01']['spr_maxdiff_rel']
    fit_x_dict['07']['spr_maxdiff_rel']  = fit_x_dict['01']['spr_maxdiff_rel']
    fit_x_dict['08']['spr_maxdiff_rel']  = fit_x_dict['01']['spr_maxdiff_rel']
    fit_x_dict['09']['spr_maxdiff_rel']  = fit_x_dict['01']['spr_maxdiff_rel']
    fit_x_dict['10']['spr_maxdiff_rel']  = fit_x_dict['01']['spr_maxdiff_rel']
    fit_x_dict['11']['spr_maxdiff_rel']  = fit_x_dict['01']['spr_maxdiff_rel']
    fit_x_dict['12']['spr_maxdiff_rel']  = fit_x_dict['01']['spr_maxdiff_rel']
    fit_x_dict['13']['spr_maxdiff_rel']  = fit_x_dict['01']['spr_maxdiff_rel']
    fit_x_dict['14']['spr_maxdiff_rel']  = fit_x_dict['01']['spr_maxdiff_rel']
    fit_x_dict['15']['spr_maxdiff_rel']  = fit_x_dict['01']['spr_maxdiff_rel']
    fit_x_dict['16']['spr_maxdiff_rel']  = fit_x_dict['01']['spr_maxdiff_rel']
    fit_x_dict['17']['spr_maxdiff_rel']  = fit_x_dict['01']['spr_maxdiff_rel']
    fit_x_dict['18']['spr_maxdiff_rel']  = fit_x_dict['01']['spr_maxdiff_rel']
    fit_x_dict['19']['spr_maxdiff_rel']  = fit_x_dict['01']['spr_maxdiff_rel']
    fit_x_dict['20']['spr_maxdiff_rel']  = fit_x_dict['01']['spr_maxdiff_rel']
    fit_x_dict['21']['spr_maxdiff_rel']  = fit_x_dict['01']['spr_maxdiff_rel']
    fit_x_dict['22']['spr_maxdiff_rel']  = fit_x_dict['01']['spr_maxdiff_rel']
    fit_x_dict['23']['spr_maxdiff_rel']  = fit_x_dict['01']['spr_maxdiff_rel']
    fit_x_dict['24']['spr_maxdiff_rel']  = fit_x_dict['01']['spr_maxdiff_rel']
    fit_x_dict['25']['spr_maxdiff_rel']  = fit_x_dict['01']['spr_maxdiff_rel']
    fit_x_dict['26']['spr_maxdiff_rel']  = fit_x_dict['01']['spr_maxdiff_rel']
    fit_x_dict['27']['spr_maxdiff_rel']  = fit_x_dict['01']['spr_maxdiff_rel']
    fit_x_dict['28']['spr_maxdiff_rel']  = fit_x_dict['01']['spr_maxdiff_rel']
    fit_x_dict['29']['spr_maxdiff_rel']  = fit_x_dict['01']['spr_maxdiff_rel']
    # fit_x_dict['30']['spr_maxdiff_rel']  = fit_x_dict['01']['spr_maxdiff_rel']

    return score_y, fit_x_dict

