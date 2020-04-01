# computer-vision-backend
[Computer Vision Project Collection](https://github.com/seanpavlak/computer-vision-backend/blob/master/mnist%20classification/E1_keras.ipynb)

# Image Filter Project
* Plot the basis functions of a 16x16 discrete cosine transform (DCT)
* Implement low- and high-pass image filters by zeroing different ranges of the DCT coefficients
* Convolving a 2D convolution kernel with an image
* Multiply the transforms of the kernel with the image and apply the inverse transform
* PCA fit, isolate the required components, and PCA inverse transform to generate the original image

## Abstract
The first task was to identify the basis functions of a discrete cosine transform, this was done by setting the values an nxn matrix to adhere to a sinusoid. This plot of a DCT allows us to directly see that the top left of the plot renders broad strokes while the bottom right renders fine detail. Using this information we can apply a low- and high-pass image filters by applying different ranges of DCT coefficients. 

The next task was to convolve a 2D convolution kernel with an image, and multiply the transforms of the kernel with the image and apply the inverse transform. This is to show that both of these actions are in fact equivalent. In doing so I was able to generate an infinitesimal mean square error, to the order of 10^(-31).

Lastly, I performed PCA on an MNIST image to generate the 784 principle components. Once these were identified I selected a set of components and inverse transformed them to generate an image similar to the original image. 

## Tech Stack
* [Numpy](http://www.numpy.org/)
* [Matplotlib](https://matplotlib.org/)
* [Scipy](https://www.scipy.org/)
* [PIL](https://pillow.readthedocs.io/en/stable/)
* [Imageio](https://imageio.github.io/)
* [Glob](https://docs.python.org/2/library/glob.html)
* [Ipywidgets](https://ipywidgets.readthedocs.io/en/stable/)
* [IPython](https://ipython.org/)
* [Sklearn](https://scikit-learn.org/stable/)

# MNIST Classification Project
[Classify MNIST Data Set using Keras](https://github.com/seanpavlak/computer-vision-backend/blob/master/semantic%20segmentation/E1_%20semantic_segmentation.ipynb)

## Abstract
In this exercise I utilized the Keras library. Keras is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano. It was developed with a focus on enabling fast experimentation. My approach was to take this library and condense it down into a single executable function and make small incremental changes to my layers in order to develop the fastest and most accurate CNN possible. This CNN was trained on the MNIST dataset. In this experiment I was able to generate many CNNs, some of which had an accuracy up to 98.8%.

## Tech Stack
* [Matplotlib](https://matplotlib.org/)
* [Numpy](http://www.numpy.org/)
* [Keras](https://keras.io/)

# Semantic Segmentation Project
Using KITTI Road Data produce a model that is able to discern what is, and what is not, road
> [Download Data](https://www.dropbox.com/sh/flf5dnhuq6b9z6h/AAADsRNWn7lwb2wAZIhy3iNPa?dl=0)


## Abstract
I set out to better understand and implement semantic segmentation. At this point in time deep learning is the chosen mechanism to perform semantic segmentation, but people used approaches like TextonForest and Random Forest based classifiers in the past. As with image classification, convolutional neural networks have had enormous success on segmentation problems.

One newer method of segmentation is Fully Convolutional Neural Networks. Fully convolutional indicates that the neural network is composed of convolutional layers without any fully-connected layers usually found at the end of the network. To semantic segment a image, a "fully convolutional" neural networks can take input of arbitrary size and produce correspondingly-sized output with efficient inference and learning, both learning and inference are performed whole-image-at-a-time by dense feedforward computation and back-propagation. In-network upsampling layers enable pixelwise prediction and learning in nets with subsampled pooling.

The dataset I am using is the KITTI road data. KITTI is a project of Karlsruhe Institute of Technology and Toyota Technological Institute at Chicago. The dataset includes three different categories of road; urban unmarked, urban marked, and urban multiple marked lanes. This data is also broken up into preset training and testing sets.

The best model I was able to generate was able to differentiate between road and non-road area efficiently. This testing set still appears to give this model some difficulties with rough edges and with shadows on the road. However, in most of the cases this model was able to differentiate between road and non-road very well.

## Tech Stack
* [Distutils](https://docs.python.org/3/library/distutils.html)
* [Glob](https://docs.python.org/2/library/glob.html)
* [Numpy](http://www.numpy.org/)
* [Os](https://docs.python.org/3/library/os.html)
* [Re](https://docs.python.org/3/library/re.html)
* [Random](https://docs.python.org/2/library/random.html)
* [Scipy](https://www.scipy.org/)
* [Shutil](https://docs.python.org/3/library/shutil.html)
* [Tensorflow](https://www.tensorflow.org/)
* [Tqdm](https://tqdm.github.io/)
* [Time](https://docs.python.org/3/library/time.html)
* [Urllib](https://docs.python.org/3/library/urllib.html)
* [Warnings](https://docs.python.org/3/library/warnings.html)
* [Zipfile](https://docs.python.org/3/library/zipfile.html)
