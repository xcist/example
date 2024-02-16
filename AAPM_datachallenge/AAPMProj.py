#!/usr/bin/env python
'''
This is the forward projection code for the AAPM data challenge.
'''

import sys
import platform
import gecatsim as xc
from gecatsim.pyfiles.CommonTools import *
from gecatsim.reconstruction.pyfiles import recon
from ctypes import *
from numpy.ctypeslib import ndpointer
from gecatsim.pyfiles.CommonTools import *
import matplotlib.pyplot as plt

#=======================================
# help function
#=======================================
def AAPMProj_help():
    print("""
AAPM data challenge forward projection code, version 240216.
See https://github.com/xcist/example/blob/main/AAPM_datachallenge for additional details.
Usage:
  python AAPMProj.py {anatomy} {input_file_name}
Input:
  {anatomy}: could be h[head], o[other, non-head], or a value of FOV in mm
  {input_file_name}: input images
Output:
  {input_file_name}_DD2Proj_900x1000.raw: line integrals in binary format, can be opened via ImageJ, etc.
  32 bit floating, little endianess.
Requirements:
  xcist(github.com/xcist/main)
Bug report:
  Jiayong Zhang (jiayong.zhang@gehealthcare.com)
""")
    sys.exit(1)

#=======================================
# setting wrapper
#=======================================
def DD2FanProj(nrdet, x0, y0, xds, yds, xCor, yCor, viewangles, nrviews, sinogram, nrcols, nrrows, originalImgPtr):
    clib = load_C_lib()

    func = clib.DD2FanProj
    func.argtypes = [c_int, c_float, c_float, ndpointer(c_float), ndpointer(c_float), c_float, c_float, ndpointer(c_float), c_int, ndpointer(c_float), c_int, c_int, ndpointer(c_float)]
    func.restype = None

    func(nrdet, x0, y0, xds, yds, xCor, yCor, viewangles, nrviews, sinogram, nrcols, nrrows, originalImgPtr)

    return sinogram

if __name__=="__main__":
    # not enough arguments
    if len(sys.argv) < 3:
        print("Error! Not enough arguments, please read Usage information below.\n")
        AAPMProj_help()
    anatomy = sys.argv[1]
    if anatomy.lower() == 'h':
        FOV = 220.16
    elif anatomy.lower() == 'o':
        FOV = 400
    else:
        try:
            FOV = float(anatomy)
        except:
            print("Error! Please check the input of anatomy.\n")
    inp_file = sys.argv[2]

    sid = 550.
    sdd = 950.
    
    nrdetcols = 900
    nrcols = 512
    nrrows = 512
    pixsize = FOV/512
    nrviews = 1000
    
    x0 = 0.0/pixsize
    y0 = sid/pixsize
    xCor = 0.0/pixsize
    yCor = 0.0/pixsize
    
    dalpha = 2.*np.arctan2(1.0/2, sdd)
    alphas = (np.arange(nrdetcols)-(nrdetcols-1)/2-1.25)*dalpha
    xds = np.single(sdd*np.sin(alphas)/pixsize)
    yds = np.single((sid - sdd*np.cos(alphas))/pixsize)
    
    viewangles = np.single(1*(0+np.arange(nrviews)/(nrviews-1)*2*np.pi))
    if inp_file.split('.')[-1] == 'raw':
        raw_img = rawread(inp_file, [512, 512, 1], 'float')
    else:
        print("\nError! Input images should be in raw format.\n")
        AAPMProj_help()
        sys.exit(1)
    raw_img = raw_img/1000.*0.02+0.02 # now in the unit of mm^-1
    originalImgPtr = np.single(raw_img)
    sinogram = np.zeros([nrviews, nrdetcols, 1], dtype=np.single) 
        
    sinogram = DD2FanProj(nrdetcols, x0, y0, xds, yds, xCor, yCor, viewangles, nrviews, sinogram, nrcols, nrrows, originalImgPtr)
    sinogram = sinogram*pixsize
    rawwrite(os.path.splitext(inp_file)[0]+"_DD2FanProj_900x1000.raw", sinogram)
