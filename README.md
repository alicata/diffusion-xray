# Diffusion-xray
Experiments and utilities to reverse engineer diffusion model mechanics and their neural networks structure on a CPU laptop for ease of inspecting.  

* CPU instead of NVIDIA GPU kernels
* 8GB or less host memory instead of 8GB GPU VRAM

![xray](https://github.com/alicata/diffusion-xray/blob/main/model_architecture.png)

1. Autoencoder: VAE architecture reduces input noise and generates denoised samples by compressing and decompressing them.
2. U-Net: ResNet-based block compresses noisy samples, reconstructs them with reduced noise using estimated residuals for denoised representation.
3. Text Encoder: CLIP ViT-L/14 Text Encoder processes prompt, produces embeddings for text analysis in Stable Diffusio

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
* `pip install -r requirements` (warning: uninstall GPU keras as would conflict with CPU keras version)
* [download weights](https://github.com/keras-team/keras-io/blob/master/guides/keras_cv/generate_images_with_stable_diffusion.py)


The backend image encoder, decoder are based on Keras, but could in theory be swapped with a Pytorch or similar other DL frameowrk. But the Keras implementation makes it possible to run on CPU accessible host memory (not possible with NVDIA GPUs with 4GB VRAM or less)
