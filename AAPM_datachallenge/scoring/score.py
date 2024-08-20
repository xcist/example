#!/usr/bin/env python
import pdb
import json
import os # to create file lists
import sys # for sanity check quit
import pickle # for dictionary import
import numpy as np
from skimage.metrics import structural_similarity as ssim # for ssim calc
from sklearn.metrics import mean_absolute_error as mae # for mae calc
from sklearn.metrics import mean_squared_error as mse # for mse calc
from skimage.metrics import hausdorff_distance as hdd # for Hausdorff calc
from scipy.ndimage import rotate # for image rotation
from scipy import interpolate # for HLUT fit
from scipy import ndimage # for Sobel filter
import utils


debug = False # for print statements

if __name__ == "__main__":
    ### metric_v1
    # Define file paths
    # folder_allresults = os.path.join(submit_dir)
    path_submitted_data = 'submission'
    
    path_reference_data = '1_data_dicts/base.pkl'
    path_structures = '1_data_dicts/structures.pkl'
    path_parameters = '1_data_dicts/parameters.pkl'
    
    
    # Load submitted and reference data
    print('Load initial data for analysis...')

    # Load submitted data
    data_img = utils.load_submitted_data(path_submitted_data)
    
    # Load reference image data
    with open(path_reference_data, 'rb') as f:
        data_ref = pickle.load(f)    
        
    # Load patient-specific structureset references
    with open(path_structures, 'rb') as f:
        data_str = pickle.load(f)    
        # print("Data_str: ", data_str['29'].keys())

    # Fill up the mask for the hollow structure of case 3 (adjustment for fairness)
    shell = data_str['03']['metal'] 
    data_str['03']['metal'] = ndimage.binary_fill_holes(shell)
        
    # Load parameter array
    with open(path_parameters, 'rb') as f:
        parameters = pickle.load(f)     


    # Check if the imported data makes sense
    utils.sanity_check(data_img, data_ref)
    
    
    # Initiate result array
    results = {}
    
    ## Iterate analyiss through the defined cases
    cases = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
             '11', '12', '13', '14', '15', '16', '17', '18', '19' , '20',
             '21', '22', '23', '24', '25', '26', '27', '28', '29']
    
    
    for i in cases:
        print('########')
        print('Start analysis of case {}...'.format(i))
        results[i] = {}

        # Assign shorter variable to image data
        img_1 = np.array(data_ref[i]) # Reference image without metal
        img_2 = np.array(data_img[i]) # Submitted MAR image
        # pdb.set_trace()
        
        # Load external mask for evaluation, exclude metal ground truth 
        mask_ext = data_str[i]['external'] 
        mask_extwom = data_str[i]['external'] # 
        mask_extwom[data_str[i]['metal']== True] = False
        

        ### UPDATED PART BEGINNING ########
 
        # Create a diluted metal mask for bigger exclusion of metal artifacts
        metalmask_diluted = ndimage.binary_dilation(data_str[i]['metal'], 
                                                    iterations=2) 
        mask_extwom_dil = np.copy(data_str[i]['external']) 
        mask_extwom_dil[metalmask_diluted == True] = False

        
        ## Calculate SSIM metrics       
        print('Calculate SSIM metrics...')
        results[i]['ssim'] = {}
        results[i]['ssim']['ssim_map'], results[i]['ssim']['ssim_masked'] = ( 
            utils.metric_ssim(img_1, img_2, mask_extwom_dil, metalmask_diluted))

        ## Calculate CTN error metrics within external, ground truth excluded
        print('Calculate CTN error metrics...')
        results[i]['mae'] = mae(img_1[mask_extwom_dil == 1], 
                                img_2[mask_extwom_dil == 1])
        results[i]['mse'] = mse(img_1[mask_extwom_dil == 1], 
                                img_2[mask_extwom_dil == 1])
        
        results[i]['rmse'] = np.sqrt(results[i]['mse'])
        results[i]['me'] = utils.metric_mean_error(img_1[mask_extwom_dil == 1], 
                                             img_2[mask_extwom_dil == 1])

        ### UPDATED PART END ######## 
        
               
        ## Calculate noise as STD within homogenous regions
        print('Calculate noise metric...')
        results[i]['noise_ref'] = np.std(img_1[data_str[i]['noise'] == 1])
        results[i]['noise'] = np.std(img_2[data_str[i]['noise'] == 1])

        
        ## Calculate average Sobel maximum within defined masks
        print('Calculate Sobel metric...')
        (results[i]['sobel_rel_bone'], results[i]['sobel_rel_soft'],
         results[i]['sobel_rel_avg']) = utils.metric_sobel(img_1, img_2, 
                        data_str[i]['sobel_soft'], data_str[i]['sobel_bone'])
        

        ## Evaluate metal geometry
        print('Calculate metal geometry metrics...')
        (results[i]['metal_vol_rel'], results[i]['dice_metal'], 
         results[i]['hdd_metal']) = utils.metric_metalgeometry(
             img_2, data_str[i]['metal'], data_str[i]['metalregion'], 
             parameters[i]['voxelsize'][0], parameters[i]['bonetrs'], 250)         

             
        ## Evaluate bone geometry
        print('Calculate bone geometry metrics...')
        (results[i]['bone_vol_rel'], results[i]['dice_bone'], 
         results[i]['hdd_bone']) = utils.metric_bone_geometry(img_1, img_2, 
            mask_extwom, parameters[i]['bonetrs'], 
            parameters[i]['voxelsize'][0])        

        
        ## Calculate the proton range difference 
        print('Calculate proton beam ranges...')
        (results[i]['spr_lineintg_ref'], results[i]['spr_lineintg'], 
         results[i]['spr_lineintg_diff'],results[i]['spr_maxdiff'],
         results[i]['spr_maxdiff_rel']) = (
             utils.metric_protonrange(parameters[i]['beamangle'], img_1, img_2, 
                data_str[i]['tumor'], mask_ext, parameters[i]['voxelsize'][0]))
        

        ## Evaluate streak amplitudes
        print('Calculate streak amplitude metric...')
        results[i]['streakamplitude'] = utils.metric_streak(img_2, img_1,
                                            data_str[i]['roi_amplitude'] )

        
    # Create output folder and export data for analysis in 
    # metric_scoring.py
        
    if not os.path.exists('2_results/'):
        os.makedirs('2_results/')   
        
    with open('2_results/results.pkl', 'wb') as f:
        pickle.dump(results, f)

    print('########')
    print('All calculations concluded. Good job! Always wear your seatbelt. ')


    ### metric_scoring_v1.py
    # Define which metrics to be includedin scoring
    scoring_metrics = ['rmse', 'noise', 'sobel_rel_avg', 'streakamplitude', 
                       'ssim', 'metal_geometry_comb', 'bone_geometry_comb', 
                       'spr_maxdiff_rel'] # lesion_detec is out
    
    # Load the results dictionary created by metric.py
    folder = '2_results'
    path_dict = folder + '/results.pkl'
    with open(path_dict, 'rb') as f:
        results_submission = pickle.load(f)

   # Define scoring dictioary
    scores = {}
    for i in results_submission.keys():
        scores[i] = {}

    # Define fit parameters to assign scores
    score_y, fit_x_dict = utils.scoring_axes(results_submission)

    # Fill dictionary with scores
    for i in results_submission.keys():
        # Metric 1: CTN accuracy
        x_ctn_acc = fit_x_dict[i]['ctn_acc']
        fit_ctn_acc = interpolate.interp1d(x_ctn_acc, score_y)
        scores[i]['rmse'] = fit_ctn_acc(results_submission[i]['rmse'])


        # Metric 2: Noise
        x_ctn_noise = fit_x_dict[i]['noise']
        fit_ctn_noise = interpolate.interp1d(x_ctn_noise, score_y)
        scores[i]['noise'] = fit_ctn_noise(results_submission[i]['noise'])

        # Metric 3: Spatial resolution
        x_ctn_sobel = fit_x_dict[i]['sobel_rel_avg']
        fit_ctn_sobel = interpolate.interp1d(x_ctn_sobel, score_y)
        scores[i]['sobel_rel_avg'] = fit_ctn_sobel(
            results_submission[i]['sobel_rel_avg'])
        
        # Metric 4: Streak amplitude
        x_ctn_sreak = fit_x_dict[i]['streak']
        fit_ctn_streak = interpolate.interp1d(x_ctn_sreak, score_y)
        scores[i]['streakamplitude'] = fit_ctn_streak(
            results_submission[i]['streakamplitude'])


        # Metric 5: SSIM
        x_ctn_ssim = fit_x_dict[i]['ssim']
        fit_ctn_ssim = interpolate.interp1d(x_ctn_ssim, score_y)
        scores[i]['ssim'] = fit_ctn_ssim(
            results_submission[i]['ssim']['ssim_masked'])

        
        # Metric 6a: Relative metal volume change
        x_ctn_metal_vol_rel = fit_x_dict[i]['metal_vol_rel']
        fit_ctn_metal_vol_rel = interpolate.interp1d(x_ctn_metal_vol_rel,
                                                      score_y)
        scores[i]['metal_vol_rel'] = fit_ctn_metal_vol_rel(
            results_submission[i]['metal_vol_rel'])


        # Metric 6b: Dice coefficient metal
        x_ctn_metal_dice = fit_x_dict[i]['dice_metal']
        fit_ctn_metal_dice = interpolate.interp1d(x_ctn_metal_dice, score_y)
        scores[i]['dice_metal'] = fit_ctn_metal_dice(
            results_submission[i]['dice_metal'])

        # Combine Metric 6a and 6b
        scores[i]['metal_geometry_comb'] = np.mean(
            [scores[i]['metal_vol_rel'], scores[i]['dice_metal'] ])

        # Set the score of metal geometry for hollow case to 2, as participants  
        # were told no hollow objects
        scores['03']['metal_geometry_comb'] = 2.0

        # Metric 7a: Relative bone volume change
        x_ctn_bone_vol_rel = fit_x_dict[i]['bone_vol_rel']
        fit_ctn_bone_vol_rel = interpolate.interp1d(x_ctn_bone_vol_rel, 
                                                    score_y)
        scores[i]['bone_vol_rel'] = fit_ctn_bone_vol_rel(
            results_submission[i]['bone_vol_rel'])

        # Metric 7b: Dice coefficient bone
        x_ctn_bone_dice = fit_x_dict[i]['dice_bone']
        fit_ctn_bone_dice = interpolate.interp1d(x_ctn_bone_dice, score_y)
        scores[i]['dice_bone'] = fit_ctn_bone_dice(
            results_submission[i]['dice_bone'])

        # Combine Metric 7a and 7b
        scores[i]['bone_geometry_comb'] = np.mean(
            [scores[i]['bone_vol_rel'], scores[i]['dice_bone'] ])


        # Metric 8: Range shift
        x_ctn_rangeshift = fit_x_dict[i]['spr_maxdiff_rel']
        fit_ctn_rangeshift = interpolate.interp1d(x_ctn_rangeshift, score_y)
        scores[i]['spr_maxdiff_rel'] = fit_ctn_rangeshift(
            results_submission[i]['spr_maxdiff_rel'])


    # Export results as .txt  files(
    if not os.path.exists('3_scoring'):
        os.makedirs('3_scoring')   
    
    for i in results_submission.keys():
        # results_list = []
        with open('3_scoring/results_{}.txt'.format(i), 'w')as f:
            for j in scoring_metrics:
                item = scores[i][j]
                f.write(str(j) + '\t' + str(item) + '\n')
    

    ### Write Leaderboard
    output_filename = 'scores.txt'
    output_file = open(output_filename, 'w')
    
    html_filename = 'scores.html'
    html_file = open(html_filename, 'w')

    # Participant Data 
    scoring_metrics = [i for i in scoring_metrics]
    all_metrics_scores = {metric:[] for metric in scoring_metrics}
    for score in scores.keys():
        for metric in scoring_metrics:
            all_metrics_scores[metric].append(scores[score][metric])

    all_cases_scores = {case:[] for case in cases}
    for score in scores.keys():
        for metric in scoring_metrics:
            all_cases_scores[score].append(scores[score][metric])

    final_score = np.average(
        [
        np.mean(all_metrics_scores['rmse']),
        np.mean(all_metrics_scores['noise']),
        np.mean(all_metrics_scores['sobel_rel_avg']),
        np.mean(all_metrics_scores['streakamplitude']),
        np.mean(all_metrics_scores['ssim']),
        np.mean(all_metrics_scores['metal_geometry_comb']),
        np.mean(all_metrics_scores['bone_geometry_comb']),
        np.mean(all_metrics_scores['spr_maxdiff_rel'])
        ],
        weights = [1,1,1,1,1,1,1,1]
    )

    
    for key in scores.keys():
        scores[key]
        for metric_key in scores[key].keys():
            scores[key][metric_key] = float(scores[key][metric_key])

    # Leaderboard
    ## By case
    output_file.write(f"Score: {final_score}\n")
    output_file.write(f"Avg_01: {np.mean(all_cases_scores['01'])}\n")
    output_file.write(f"Avg_02: {np.mean(all_cases_scores['02'])}\n")
    output_file.write(f"Avg_03: {np.mean(all_cases_scores['03'])}\n")
    output_file.write(f"Avg_04: {np.mean(all_cases_scores['04'])}\n")
    output_file.write(f"Avg_05: {np.mean(all_cases_scores['05'])}\n")
    output_file.write(f"Avg_06: {np.mean(all_cases_scores['06'])}\n")
    output_file.write(f"Avg_07: {np.mean(all_cases_scores['07'])}\n")
    output_file.write(f"Avg_08: {np.mean(all_cases_scores['08'])}\n")
    output_file.write(f"Avg_09: {np.mean(all_cases_scores['09'])}\n")
    output_file.write(f"Avg_10: {np.mean(all_cases_scores['10'])}\n")
    output_file.write(f"Avg_11: {np.mean(all_cases_scores['11'])}\n")
    output_file.write(f"Avg_12: {np.mean(all_cases_scores['12'])}\n")
    output_file.write(f"Avg_13: {np.mean(all_cases_scores['13'])}\n")
    output_file.write(f"Avg_14: {np.mean(all_cases_scores['14'])}\n")
    output_file.write(f"Avg_15: {np.mean(all_cases_scores['15'])}\n")
    output_file.write(f"Avg_16: {np.mean(all_cases_scores['16'])}\n")
    output_file.write(f"Avg_17: {np.mean(all_cases_scores['17'])}\n")
    output_file.write(f"Avg_18: {np.mean(all_cases_scores['18'])}\n")
    output_file.write(f"Avg_19: {np.mean(all_cases_scores['19'])}\n")
    output_file.write(f"Avg_20: {np.mean(all_cases_scores['20'])}\n")
    output_file.write(f"Avg_21: {np.mean(all_cases_scores['21'])}\n")
    output_file.write(f"Avg_22: {np.mean(all_cases_scores['22'])}\n")
    output_file.write(f"Avg_23: {np.mean(all_cases_scores['23'])}\n")
    output_file.write(f"Avg_24: {np.mean(all_cases_scores['24'])}\n")
    output_file.write(f"Avg_25: {np.mean(all_cases_scores['25'])}\n")
    output_file.write(f"Avg_26: {np.mean(all_cases_scores['26'])}\n")
    output_file.write(f"Avg_27: {np.mean(all_cases_scores['27'])}\n")
    output_file.write(f"Avg_28: {np.mean(all_cases_scores['28'])}\n")
    output_file.write(f"Avg_29: {np.mean(all_cases_scores['29'])}\n")
    ## By metric
    # output_file.write(f"Score: {final_score}\n")
    # output_file.write(f"Avg_rmse: {np.mean(all_metrics_scores['rmse'])}\n")
    # output_file.write(f"Avg_noise: {np.mean(all_metrics_scores['noise'])}\n")
    # output_file.write(f"Avg_sobel_rel_avg: {np.mean(all_metrics_scores['sobel_rel_avg'])}\n")
    # output_file.write(f"Avg_streakamplitude: {np.mean(all_metrics_scores['streakamplitude'])}\n")
    # output_file.write(f"Avg_ssim: {np.mean(all_metrics_scores['ssim'])}\n")
    # output_file.write(f"Avg_metal_geometry_comb: {np.mean(all_metrics_scores['metal_geometry_comb'])}\n")
    # output_file.write(f"Avg_bone_geometry_comb: {np.mean(all_metrics_scores['bone_geometry_comb'])}\n")
    # output_file.write(f"Avg_spr_maxdiff_rel: {np.mean(all_metrics_scores['spr_maxdiff_rel'])}")

    # Detailed Results
    # remove 'metal_vol_rel', 'dice_metal', 'bone_vol_rel', 'dice_bone', 'lesion_detec'
    keys_to_remove = ['metal_vol_rel', 'dice_metal', 'bone_vol_rel', 'dice_bone', 'lesion_detec']
    for case in scores.values():
        for key in keys_to_remove:
            case.pop(key, None)

    html_file.write('<pre>' + json.dumps(scores, indent=2) + '</pre>')
       
    output_file.close()
    html_file.close()
    if debug: print("Scores: ", scores)

    # pdb.set_trace()
