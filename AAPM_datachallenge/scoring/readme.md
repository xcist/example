This is the scoring program for the [AAPM MAR grand challenge](https://www.aapm.org/GrandChallenge/CT-MAR/)

# Prerequisites
* python 3.10+
* numpy
* scikit-image

# How to run the scoring program
1. put your MAR results of `01.raw`, ..., `29.raw` to the `submission` folder. The input images should be HU values of size 512x512 pixels in the float32 raw format
2. run `python score.py`

# output
* `scores.txt`: final score and average scores for each image
* `scores.html`: detailed scores for each image (in html format)
* `3_scoring/results_xx.txt`: detailed scores for each image
