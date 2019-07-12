# Traffic4cast benchmarks and submission utilities

## Introduction

The aim of our core competition is to predict the next 3 images in our traffic movies, which encode
the volume, speed and direction of observed traffic in each 100m x 100m grid within a 5min interval
into an RGB pixel colour as described for Berlin, Istanbul and Moscow.

The submission format for the 5 sequences of 3 images in each day of the test set is a multi-dimensional
array (tensor) of shape (5,3,495,436,3) and the objective function of all submitted tensors $x^c_1,\cdot x^c_72$
for each city $c$ is
$$
\frac{1}{3\cdot 72\cdot 5\cdot 3\cdot 495\cdot 436\cdot 3}\sum_c \sum_j=1^{72} \sum_{\phi} \left( \frac{x^c_{j,\phi}}{255} - \frac{g^c_{j,\phi}}{255} \right)^2
$$
where $g$ is the true collection of tensors observed (golden set).

## Benchmarks

The attached code provides the following benchmarks.

### Submission of only zeros.
Given that many pixel values are (0,0,0) (given the absence of drivable road network in the underlying 100mx100m area) just submitting
all zero values will result in many of the terms in the above MSE calculation to be zero.

### Simple average of 3 previous images.
Our second benchmark is the prediction obtained by averaging the 3 previous image pixel values in all colour channels to obtain an estimate
of the next image's corresponding value.

### Seq2Seq network taking 3 input images and predicting the next 3 trained on entire training data.

### Seq2Seq network taking 3 input images and predicting the next 3 trained only on the corresponding time slots for the submission.

## Utility scripts


