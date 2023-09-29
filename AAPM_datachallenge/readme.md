# `AAPMRecon.py` installation guide

## How to install
Simply run `pip install gecatsim` either in a Windows Anaconda prompt or Linux terminal (if you have done this before, please also run this command as we have been updating the package continuously). This will install the backbone XCIST package used in following reconstructions .
After XCIST is installed, download AAPMRecon.py from [https://github.com/xcist/example/AAPM_datachallenge/AAPMRecon.py](https://github.com/xcist/example/blob/main/AAPM_datachallenge/AAPMRecon.py) to the local directory that you want to do reconstructions (usually a directory containing sinogram). No installation is needed for this utility.
The current code has been validated to work on Windows, Linux OS. However, we recommend doing the simulation on Linux as it has threaded version of reconstruction, which should be faster.

## Usage
To run the AAPMRecon.py, just run `python AAPMRecon.py {input}`, in which the {input} is the input filename of a sinogram. The sinogram should be a binary file of 32-bit floating point (float32) of dimension [1000, 1, 900] (corresponding to [number of view, number of detector rows, number of detector columns]).
After running this command, you should be able to see the output in the name of {input}_512x512x1.raw, which is a binary file of float32 in the dimension of 512x512x1, i.e., it has 1 slice of reconstructed images of size 512x512 pixels. [Just a side note, the output can be directly opened in ImageJ if you want to have a look, but remember to make sure data type, size, endianness, etc. are correct.]

 ![Picture1](https://github.com/xcist/example/assets/100655819/103e9dc5-3940-4cca-974f-a499923697c8)

## Others
For any problems encountered in the reconstruction, please send an email to Jiayong.Zhang@ge.com
