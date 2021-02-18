# Reposition Image Illumination

## Introduction

This project attempts to shift the lighting of an image from north to east, while adding a bit of warmth.
Given an image of a scene, lit from the north side, our trained neural network outputs an image of the
same scene, lit from the east, with a slightly warmer light. We eventually achieved our goal using a
[pytorch pix2pix network implementation](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) based on the works of [Jun-Yan Zhu](https://github.com/junyanz) and [Taesung Park](https://github.com/taesungp), and supported by [Tongzhou Wang](https://github.com/SsnL).


## Runtime Performance

Using Google Colab’s GPUs (exact model unknown) training 256x256 images, an epoch of 432 images
took between 31-33 seconds.
Prediction of a similarly sized image takes ~1.5 seconds per image.

## Running Instructions

**Example dataset:**

[Dataset](https://drive.google.com/file/d/1QD_r4pGGZlAethSVmdS2Y5bvZslesPdJ/view?usp=sharing)


**Data preparation:**
Obtain a dataset of at least a few hundred images (the more, the better) of images of size 256X256 and
RGB format. This network may be trained for different lighting angles (such as south to west, etc.).

We provide a python script to generate pix2pix training data in the form of pairs of images {A,B}, where
A and B are two different depictions of the same underlying scene. For example, these might be pairs
{label map, photo} or {BW image, color image}. Then we can learn to translate A to B or B to A:

Create folder /path/to/data with subfolders A and B. A and B should each have their own
subfolders train, test, etc. In /path/to/data/A/train, put training images in style A. In
/path/to/data/B/train, put the corresponding images in style B. Repeat same for other data
splits (test, etc.).

Corresponding images in a pair {A,B} must be the same size and have the same filename, e.g.,
/path/to/data/A/train/1.jpg is considered to correspond to
/path/to/data/B/train/1.jpg.

Once the data is formatted this way, call:
python datasets/combine_A_and_B.py --fold_A /path/to/data/A --fold_B
/path/to/data/B --fold_AB /path/to/data
This will combine each pair of images (A,B) into a single image file, ready for training.

**Training:**
Change the --dataroot and --name to your own dataset's path and model's name. Use --
gpu_ids 0,1,.. to train on multiple GPUs and --batch_size to change the batch size. Add --
direction BtoA if you want to train a model to transform from class B to A.

!python train.py --dataroot ./datasets/path/to/data --name day2night --
model pix2pix --direction AtoB

**Testing:**
Change the --dataroot, --name, and --direction to be consistent with your trained model's
configuration and how you want to transform images.

!python test.py --dataroot ./datasets/relighting_dataset/ --direction
AtoB --model pix2pix --name relighting

The output images will be in the directory results/--name/test_latest/images.
