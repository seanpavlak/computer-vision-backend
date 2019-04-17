# computer-vision-backend
Computer Vision Project Collection

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
Classify MNIST Data Set using Keras

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
