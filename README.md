# Diffusion-xray
Experiments and utilities to reverse engineer stable diffusion model mechanics and their neural networks structure on a CPU laptop for ease of inspecting.  

## Alternatives to running SD on GPU
* Run Tensorflow on CPU  
* 8GB or less host memory instead of 8GB GPU VRAM

![xray](https://github.com/alicata/diffusion-xray/blob/main/model_architecture.png)



1. Autoencoder: VAE architecture encodes the input image into a compressed latent vector, and decodes the denoised latent vector back into a image.
2. [UNet](https://github.com/alicata/diffusion-xray/blob/main/stable_diffusion_tf/diffusion_model.py#L138): ResNet-based blocks denoise noisy latent vector, reconstructs them with estimated residuals, and the cross-attention layers integrate the conditional text embedding vector.
3. Text Encoder: CLIP ViT-L/14 Text Encoder processes prompt, produces embeddings for text conditioning (input to the attention layers in the UNet).
4. Scheduler runs noising of latent vector forward for several steps, and also runs the UNet denoiser backwards the needed time steps.

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
