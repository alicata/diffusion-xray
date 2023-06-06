# diffusion-xray
Experiments and utilities to reverse engineer diffusion model mechanics and their neural networks structure. 

## Experiments
all experiments are scripts that start with test* or text* filename. They each decompose the building blocks of a diffuction model and test its characterictics in isolation.

## Tools
Module xray/tensor_debug contains a DebugTensor class to support reverse engineer an encoded vector at various (de)noising stages.

The backend image encoder, decoder are based on Keras, but could in theory be swapped with a Pytorch or similar other DL frameowrk. But the Keras implementation makes it possible to run on CPU memory (not possible with NVDIA GPUs with 4GB VRAM or less)
