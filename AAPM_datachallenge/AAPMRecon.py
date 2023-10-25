#!/usr/bin/env python
'''
This is the recon code for the AAPM data challenge. Run `AAPMRecon.py` to see usage.
'''

import sys
import platform
import gecatsim as xc
from gecatsim.pyfiles.CommonTools import *
from gecatsim.reconstruction.pyfiles import recon

#=======================================
# help function
#=======================================
def AAPMRecon_help():
    print("""
AAPM data challenge reconstruction code, version 230919.
See https://github.com/xcist/example/blob/main/AAPM_datachallenge for additional details.
Usage:
  python AAPMRecon.py {anatomy} {input_file_name}
Input:
  {anatomy}: head or not, could be either h[head] or o[other, non-head]
  {input_file_name}: input sinogram
Output:
  {input_file_name}_512x512x4.raw: reconstructed images in binary format, can be opened via ImageJ, etc.
  32 bit floating, little endianess.
Requirements:
  xcist(github.com/xcist/main)
Bug report:
  Jiayong Zhang (jiayong.zhang@ge.com)
""")
    sys.exit(1)

#=======================================
# setting vendor neutral cfgs
#=======================================
def AAPMRecon_init(inp_file, FOV):
    cfg = CFG()

    # Phantom
    cfg.phantom.callback = "Phantom_Voxelized"      # name of function that reads and models phantom
    cfg.phantom.projectorCallback = "C_Projector_Voxelized" # name of function that performs projection through phantom
    cfg.phantom.filename = 'CatSim_logo_1024.json'  # phantom filename, not actually used in AAPM Recon
    cfg.phantom.centerOffset = [0.0, 0.0, 0.0]      # offset of phantom center relative to origin (in mm)
    cfg.phantom.scale = 1                           # re-scale the size of phantom
    if platform.system() == "Linux":
        cfg.phantom.projectorNumThreads = 4
    elif platform.system() == "Windows":
        cfg.phantom.projectorNumThreads = 1
    else:
        cfg.phantom.projectorNumThreads = 1
    
    # physics
    cfg.physics.energyCount = 12                    # number of energy bins
    cfg.physics.monochromatic = -1                  # -1 for polychromatic (see protocol.cfg);
    cfg.physics.colSampleCount = 1                  # number of samples of detector cells in lateral direction
    cfg.physics.rowSampleCount = 1                  # number of samples of detector cells in longitudinal direction
    cfg.physics.srcXSampleCount = 2                 # number of samples of focal spot in lateral direction
    cfg.physics.srcYSampleCount = 2                 # number of samples of focal spot cells in longitudinal direction
    cfg.physics.viewSampleCount = 2                 # number of samples of each view angle range in rotational direction
    cfg.physics.recalcDet = 0                       # recalculate detector geometry
    cfg.physics.recalcSrc = 0                       # recalculate source geometry and relative intensity
    cfg.physics.recalcRayAngle = 0                  # recalculate source-to-detector-cell ray angles
    cfg.physics.recalcSpec = 0                      # recalculate spectrum
    cfg.physics.recalcFilt = 0                      # recalculate filters
    cfg.physics.recalcFlux = 0                      # recalculate flux
    cfg.physics.recalcPht = 0                       # recalculate phantom
    cfg.physics.enableQuantumNoise = 1              # enable quantum noise
    cfg.physics.enableElectronicNoise = 1           # enable electronic noise
    cfg.physics.rayAngleCallback = "Detector_RayAngles_2D" # name of function to calculate source-to-detector-cell ray angles
    cfg.physics.fluxCallback = "Detection_Flux"     # name of function to calculate flux
    cfg.physics.scatterCallback = "Scatter_ConvolutionModel"                # name of function to calculate scatter
    cfg.physics.scatterKernelCallback = ""          # name of function to calculate scatter kernel ("" for default kernel)
    cfg.physics.scatterScaleFactor = 1              # scale factor, 1 appropriate for 64-mm detector and 20-cm water
    cfg.physics.callback_pre_log = "Scatter_Correction"
    cfg.physics.prefilterCallback = "Detection_prefilter" # name of function to calculate detection pre-filter
    cfg.physics.crosstalkCallback = "CalcCrossTalk" # name of function to calculate X-ray crosstalk in the detector
    cfg.physics.col_crosstalk = 0.025
    cfg.physics.row_crosstalk = 0.02
    cfg.physics.opticalCrosstalkCallback = "CalcOptCrossTalk" # name of function to calculate X-ray crosstalk in the detector
    cfg.physics.col_crosstalk_opt = 0.04
    cfg.physics.row_crosstalk_opt = 0.045
    cfg.physics.lagCallback = ""                    # name of function to calculate detector lag
    cfg.physics.opticalCrosstalkCallback = ""       # name of function to calculate optical crosstalk in the detector
    cfg.physics.DASCallback = "Detection_DAS"       # name of function to calculate the detection process
    cfg.physics.outputCallback = "WriteRawView"     # name of function to produce the simulation output
    cfg.physics.callback_post_log = 'Prep_BHC_Accurate'
    cfg.physics.EffectiveMu = 0.2
    cfg.physics.BHC_poly_order = 5
    cfg.physics.BHC_max_length_mm = 300
    cfg.physics.BHC_length_step_mm = 10
    
    # protocol
    cfg.protocol.scanTypes = [1, 1, 1]              # flags for airscan, offset scan, phantom scan
    cfg.protocol.scanTrajectory = "Gantry_Helical"  # name of the function that defines the scanning trajectory and model
    cfg.protocol.viewsPerRotation = 1000            # total numbers of view per rotation
    cfg.protocol.viewCount = 1000                   # total number of views in scan
    cfg.protocol.startViewId = 0                    # index of the first view in the scan
    cfg.protocol.stopViewId = cfg.protocol.startViewId + cfg.protocol.viewCount - 1 # index of the last view in the scan
    cfg.protocol.airViewCount = 1                   # number of views averaged for air scan
    cfg.protocol.offsetViewCount = 1                # number of views averaged for offset scan
    cfg.protocol.rotationTime = 1.0                 # gantry rotation period (in seconds)
    cfg.protocol.rotationDirection = 1              # gantry rotation direction (1=CW, -1 CCW, seen from table foot-end)
    cfg.protocol.startAngle = 0                     # relative to vertical y-axis (n degrees)
    cfg.protocol.tableSpeed = 0                     # speed of table translation along positive z-axis (in mm/sec)
    cfg.protocol.startZ = 0                         # start z-position of table
    cfg.protocol.tiltAngle = 0                      # gantry tilt angle towards negative z-axis (in degrees)
    cfg.protocol.wobbleDistance = 0.0               # focalspot wobble distance
    cfg.protocol.focalspotOffset = [0, 0, 0]        # focalspot position offset
    cfg.protocol.mA = 500                           # tube current (in mA)
    cfg.protocol.spectrumCallback = "Spectrum"      # name of function that reads and models the X-ray spectrum
    cfg.protocol.spectrumFilename = "xcist_kVp120_tar7_bin1.dat" # name of the spectrum file
    cfg.protocol.spectrumUnit_mm = 1;               # Is the spectrum file in units of photons/sec/mm^2/<current>?
    cfg.protocol.spectrumUnit_mA = 1;               # Is the spectrum file in units of photons/sec/<area>/mA?
    cfg.protocol.spectrumScaling = 1                # scaling factor, works for both mono- and poly-chromatic spectra
    cfg.protocol.bowtie = "large.txt"               # name of the bowtie file (or [] for no bowtie)
    cfg.protocol.filterCallback = "Xray_Filter"     # name of function to compute additional filtration
    cfg.protocol.flatFilter = ['air', 0.001]        # additional filtration - materials and thicknesses (in mm)
    cfg.protocol.dutyRatio = 1.0                    # tube ON time fraction (for pulsed tubes)
    cfg.protocol.maxPrep = -1                       # set the upper limit of prep, non-positive will disable this feature
    
    # Scanner
    cfg.scanner.detectorCallback = "Detector_ThirdgenCurved" # name of function that defines the detector shape and model
    cfg.scanner.sid = 550.0                         # source-to-iso distance (in mm)
    cfg.scanner.sdd = 950.0                         # source-to-detector distance (in mm)
    cfg.scanner.detectorColsPerMod = 1              # number of detector columns per module
    cfg.scanner.detectorRowsPerMod = 1              # number of detector rows per module
    cfg.scanner.detectorColOffset = -1.25             # detector column offset relative to centered position (in detector columns)
    cfg.scanner.detectorRowOffset = 0.0             # detector row offset relative to centered position (in detector rows)
    cfg.scanner.detectorColSize = 1.0               # detector column pitch or size (in mm)
    cfg.scanner.detectorRowSize = 1.0               # detector row pitch or size (in mm)
    cfg.scanner.detectorColCount = 900              # total number of detector columns
    cfg.scanner.detectorRowCount = cfg.scanner.detectorRowsPerMod     # total number of detector rows
    cfg.scanner.detectorPrefilter = []              # detector filter 
    cfg.scanner.focalspotCallback = "SetFocalspot"  # name of function that defines the focal spot shape and model
    cfg.scanner.focalspotData = "vct_large_fs.npz"  # Parameterize the model
    cfg.scanner.targetAngle = 7.0                   # target angle relative to scanner XY-plane (in degrees)
    cfg.scanner.focalspotWidth = 1.0
    cfg.scanner.focalspotLength = 1.0
    cfg.scanner.focalspotWidthThreshold =0.5
    cfg.scanner.focalspotLengthThreshold =0.5
    
    # Detector
    cfg.scanner.detectorMaterial = "GOS"            # detector sensor material
    cfg.scanner.detectorDepth = 3.0                 # detector sensor depth (in mm)
    cfg.scanner.detectionCallback = "Detection_EI"  # name of function that defines the detection process (conversion from X-rays to detector signal)
    cfg.scanner.detectionGain = 0.1                 # factor to convert energy to electrons (electrons / keV)
    cfg.scanner.detectorColFillFraction = 0.9       # active fraction of each detector cell in the column direction
    cfg.scanner.detectorRowFillFraction = 0.9       # active fraction of each detector cell in the row direction
    cfg.scanner.eNoise = 25                         # standard deviation of Gaussian electronic noise (in electrons)
    
    # recon
    cfg.recon.fov = FOV                           # diameter of the reconstruction field-of-view (in mm)
    cfg.recon.imageSize = 512                       # number of columns and rows to be reconstructed (square)
    cfg.recon.sliceCount = 1                        # number of slices to reconstruct
    cfg.recon.sliceThickness = 0.579                 # reconstruction slice thickness AND inter-slice interval (in mm)
    cfg.recon.centerOffset = [0.0, 0.0, 0.0]        # reconstruction offset relative to center of rotation (in mm)
    cfg.recon.reconType = 'fdk_equiAngle'           # Name of the recon function to call
    cfg.recon.kernelType = 'standard'               # 'R-L' for the Ramachandran-Lakshminarayanan (R-L) filter, rectangular window function
    cfg.recon.startAngle = 0                        # in degrees; 0 is with the X-ray source at the top
    cfg.recon.unit = 'HU'                           # '/mm', '/cm', or 'HU'
    cfg.recon.mu = 0.02                             # in /mm; typically around 0.02/mm
    cfg.recon.huOffset = -1000                      # unit is HU, -1000 HU by definition but sometimes something else is preferable
    cfg.recon.printReconParameters = False          # Flag to print the recon parameters
    cfg.recon.saveImageVolume = True                # Flag to save recon results as one big file
    cfg.recon.saveSingleImages = False              # Flag to save recon results as individual imagesrecon.printReconParameters = False      # Flag to print the recon parameters
    cfg.recon.displayImagePictures = False          # Flag to display the recon results as .png images
    cfg.recon.saveImagePictureFiles = False         # Flag to save the recon results as .png images
    cfg.recon.displayImagePictureAxes = False       # Flag to display the axes on the .png images
    cfg.recon.displayImagePictureTitles = False     # Flag to display the titles on the .png images

    cfg.resultsName = inp_file.split('.')[0]

    if cfg.physics.monochromatic>0:
        cfg.recon.mu = xc.GetMu('water', cfg.physics.monochromatic)[0]/10

    cfg.do_Recon = 1
    cfg.waitForKeypress = 0

    return cfg

def AAPMRecon_main(cfg):
    recon.recon(cfg)

if __name__=="__main__":
    # not enough arguments
    if len(sys.argv) < 3: AAPMRecon_help()
    anatomy = sys.argv[1]
    if anatomy.lower() == 'h':
        FOV = 220.16
    else:
        FOV = 400
    inp_file = sys.argv[2]
    if inp_file.split('.')[-1] == 'raw':
        inp_data = rawread(inp_file, [1000, 1, 900], 'float')
        rawwrite(inp_file.replace("raw", "prep"), inp_data)
    cfg = AAPMRecon_init(inp_file, FOV)
    AAPMRecon_main(cfg)
