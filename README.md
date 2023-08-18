# Diffusion-xray
Experiments and utilities to reverse engineer stable diffusion model mechanics and their neural networks structure on a CPU laptop for ease of inspecting.  

## Alternatives to running SD on GPU
* Run Tensorflow on CPU  
* 8GB or less host memory instead of 8GB GPU VRAM

![xray](https://github.com/alicata/diffusion-xray/blob/main/model_architecture.png)


### Image Generator
The diffusion model generator orchestrates image generation through the AE, UNet, Text Encoder, Noise Scheduler:
1. Autoencoder
   - VAE encodes the input image into a compressed latent vector
   - VAE decodes the denoised latent vector back into a image
2. [UNet](https://github.com/alicata/diffusion-xray/blob/main/stable_diffusion_tf/diffusion_model.py#L138)
   - ResNet-based blocks predict noise in latent vector (estimated residuals)
   - Cross-attention layers steer the residuals using the conditional text embedding vector
3. Text Encoder CLIP ViT-L/14
   - Text Encoder tokenizes prompt
   - Text Encoder produces embeddings for text conditioning (input to the attention layers in the UNet).
4. Noise Scheduler (DDPM / DDIM)
   - runs noising of latent vector forward for several steps (train)
   - runs the UNet denoiser backwards the needed time steps (train, inference)

### Sampling: Image Corrupting Noise vs Distribution Stabilizing Noise
An interesting sampling algorithm idea: **predicted** "corrupting" noise vs **extra** "stabilizing" noise.

At each step,
* the sampling algorithm removes the predicted corrupting noise (image pixel sample not normally distribution anymore)
* the sampling algorithm adds back a smaller scaled stabilizing noise to the currently denoised image (image pixel sample normally distributed again)
* the normally distributed image sample becomes compatible again with the distribution expected by the UNet residual predictor.  

```
# remove the predicted noise from corrupted image x
# z noise back in to avoid denoising collapse due to changed noise distribution
def denoise_add_noise(x, t, pred_noise):
    z_noise = b_t.sqrt()[t] * tf.randn_like(x) # stabilizing noise
    mean = (x - pred_noise * ((1 - a_t[t]) / (1 - ab_t[t]).sqrt())) / a_t[t].sqrt()
    return mean + z_noise
```

### Debugging Noise
If the **extra** noise is not adding back the UNet denoiser predicts wrong noise levels that collapse the image mean values. 

### Context & Time Embedding ~ Scaling & Offset the Noise Decoder Level 
For each time step,
- Time Embedding added to UNet so the UNet decoder offsets the image noise with the right noise step.
- Context Embedding added to UNet controls UNet decoder image noise level by text embedding

# Model Training: Random Timestep & Loss
* Sample a random image, and sample a random timestep (noise level)
* compare predicted noise at the time step with actual injected noise
* compute MSE(noise_true, noise_pred) loss from actual and predicted noise

# Control Sampling
* Context embedding cemb is added to the random noise during training

# Fast Sampling: DDIM
```
def sample_ddim(n_sample, n):
    samples = randn(n_sample, 3, height, height).to(device)  
    step_size = timesteps // n

    for i in range(timesteps, 0, -step_size):
        t = tensor([i / timesteps])[:, None, None, None]
        pred_noise = nn_model(samples, t)   
        samples = denoise_ddim(samples, i, i - step_size, pred_noise)
```

## Experiments Overview
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
