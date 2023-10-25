This guide contains how to install and use the `AAPMRecon.py` (which is to reconstruct images from sinograms) and `AAPMProj.py` (which is to do forward projections from images) code for the AAPM data challenge.

# `AAPMRecon.py` installation guide

## How to install
Simply run `pip install gecatsim` either in a Windows Anaconda prompt or Linux terminal (if you have done this before, please also run this command as we have been updating the package continuously). This will install the backbone XCIST package used in following reconstructions.

After XCIST is installed, download `AAPMRecon.py` from https://github.com/xcist/example/blob/main/AAPM_datachallenge/AAPMRecon.py to the local directory that you want to do reconstructions (usually a directory containing sinogram). No installation is needed for this utility.
The current code has been validated to work on Windows, Linux OS. However, we recommend doing the simulation on Linux as it has threaded version of reconstruction, which should be faster.

## Usage
To run the `AAPMRecon.py`, just run `python AAPMRecon.py {anatomy} {input}`, in which `{anatomy}` specifies if the anatomy is head or not, and `{input}` is the input filename of a sinogram.

`{anatomy}` can be either `h` meaning the phantom is head or neck, or `o` means other anatomies. The only difference between `h` and `o` is FOV: `220.16mm` is used for head or neck, and `400mm` is used for others. The sinogram should be a binary file of 32-bit floating point (float32) of dimension [1000, 1, 900] (corresponding to [number of view, number of detector rows, number of detector columns]).

After running this command, you should be able to see the output in the name of `{input}_512x512x1.raw`, which is a binary file of float32 in the dimension of 512x512x1, i.e., it has 1 slice of reconstructed images of size 512x512 pixels. [Just a side note, the output can be directly opened in ImageJ if you want to have a look, but remember to make sure data type, size, endianness, etc. are correct.]

 ![Picture1](https://github.com/xcist/example/assets/100655819/103e9dc5-3940-4cca-974f-a499923697c8)

 # `AAPMProj.py` installation guide

## How to install
[no need to do this step if you have already installed `AAPMRecon.py`] Simply run `pip install gecatsim` either in a Windows Anaconda prompt or Linux terminal (if you have done this before, please also run this command as we have been updating the package continuously). This will install the backbone XCIST package used in following projections.

After XCIST is installed, download `AAPMProj.py` from https://github.com/xcist/example/blob/main/AAPM_datachallenge/AAPMProj.py to the local directory that you want to do forward projections (usually a directory containing images). No installation is needed for this utility.

## Usage
To run the `AAPMProj.py`, just run `python AAPMProj.py {anatomy} {input}`, in which `{anatomy}` specifies if the anatomy is head or not, and `{input}` is the input filename of an image.

`{anatomy}` can be either `h` meaning the phantom is head or neck, or `o` means other anatomies. The only difference between `h` and `o` is FOV: `220.16mm` is used for head or neck, and `400mm` is used for others. The input images should be HU values of size 512x512 pixels in the `raw` format.

After running this command, you should be able to see the output in the name of `{input}_900x1000.raw`, which is a binary file of float32 in the dimension of 900x1000, and it be read using `imageJ`.

Please note that the forward projection here is using 2D fan-beam Distance Driven method [1,2]. This is different from how the training dataset is generated, which employs 3D cone-beam Distance Driven method.

# Others
For any problems encountered in the reconstruction or projection, please send an email to `Jiayong.Zhang@ge.com`

# References
1. De Man, Bruno, and Samit Basu. "Distance-driven projection and backprojection in three dimensions." Physics in Medicine & Biology 49.11 (2004): 2463.
2. De Man, Bruno, and Samit Basu. "Distance-driven projection and backprojection." 2002 IEEE Nuclear Science Symposium Conference Record. Vol. 3. IEEE, 2002.
