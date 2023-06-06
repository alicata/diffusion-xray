# diffusion-xray
experiments and utilities to reverse engineer diffusion model mechanics and its neural networks.

## Experiments
all experiments are scripts that start with test* or text* filename. They each decompose the building blocks of a diffuction model and test its characterictics in isolation.

## Tools
Module xray/tensor_debug contains a DebugTensor class to support reverse engineer an encoded vector at various (de)noising stages.

The backend image encoder, decoder are based on Keras, but could in theory be swapped with a Pytorch or similar other DL frameowrk. 


