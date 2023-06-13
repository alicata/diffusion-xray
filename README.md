# Diffusion-xray
Experiments and utilities to reverse engineer diffusion model mechanics and their neural networks structure. 

## Overview
all experiments are scripts that start with test* or text* filename. They each test building blocks or simpler internal modules of the diffusion model.
* test noise corruptions in the image encoder
* test the image decoder
* test the text encoder 
* attemps visualizing semantic similarity in the text encodings' latent space
* latent level interpolations

## DebugTensor
Module xray/tensor_debug contains a DebugTensor class to support reverse engineer an encoded vector at various (de)noising stages.

## Setup Notes
* download stable diffusion weights from Keras-CV https://github.com/keras-team/keras-io/blob/master/guides/keras_cv/generate_images_with_stable_diffusion.py
* pip install opencv-python numpy keras keras-cv

The backend image encoder, decoder are based on Keras, but could in theory be swapped with a Pytorch or similar other DL frameowrk. But the Keras implementation makes it possible to run on CPU accessible host memory (not possible with NVDIA GPUs with 4GB VRAM or less)
