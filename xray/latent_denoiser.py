import tensorflow as tf
import numpy as np
from numpy.linalg import norm
from tqdm import tqdm
import cv2 as cv; 
import math

from tensorflow import keras

from stable_diffusion_tf.constants import _ALPHAS_CUMPROD
from stable_diffusion_tf.diffusion_model import UNetModel
from xray.tensor_debug import TensorDebug


class DDIMScheduler:
    def __init__(self, timesteps):
        self.timesteps = timesteps
        self.alphas = [_ALPHAS_CUMPROD[t] for t in timesteps]
        self.alphas_prev = [1.0] + self.alphas[:-1]

    def step(self):
        pass

    #def reverse_step(self, noise_pred, t, decoded_latent):
    def reverse_step(self, e_t, timestep, x):
        a_t, a_prev = self.alphas[timestep], self.alphas_prev[timestep]
        sigma_t = 0 # 0=DDIM, 1=DDPM

        sqrt_one_minus_at = math.sqrt(1 - a_t)
        pred_x0 = (x - sqrt_one_minus_at * e_t) / math.sqrt(a_t)

        # Direction pointing to x_t
        dir_xt = math.sqrt(1.0 - a_prev - sigma_t**2) * e_t
        x_prev = math.sqrt(a_prev) * pred_x0 + dir_xt
        return x_prev, pred_x0

    def add_noise(self):
        pass

class ClassifierFreeGuidance:
    def __init_(self):
        pass

    def interpolate(self):
        pass


class Diffusion():
    def __init__(self):
        self.img_width, self.img_height = 512, 512
        self.INPUT_LATENT_DIM = (64, 64, 4)
        self.INPUT_CONTEXT_DIM = (77, 768)
        self.INPUT_POS_EMB_DIM = (320,)

        fpath = self.get_model_uri() 
        self.denoiser = self.create()
        self.denoiser.load_weights(fpath)

    def create(self):
        # Creation diffusion UNet
        context = keras.layers.Input(self.INPUT_CONTEXT_DIM)
        t_emb = keras.layers.Input(self.INPUT_POS_EMB_DIM)
        latent = keras.layers.Input(self.INPUT_LATENT_DIM)
        unet = UNetModel()
        diffusion_model = keras.models.Model(
            [latent, t_emb, context], unet([latent, t_emb, context])
        )
        return diffusion_model

    def o(self):
        return {
            'input_size' : [[77,768],[320],[64, 64, 4]],
            'output_size' : [64, 64, 4],
        }

    def as_keras_model(self):
        return self.denoiser

    def get_model_uri(self):
        weights_fpath = keras.utils.get_file(
            origin="https://huggingface.co/fchollet/stable-diffusion/resolve/main/diffusion_model.h5",
            file_hash="a5b2eea58365b18b40caee689a2e5d00f4c31dbcb4e1d58a9cf1071f55bbbd3a",
            cache_dir="d:/.keras/"
        )        
        return weights_fpath

    def timestep_embedding(self, timesteps, dim=320, max_period=10000):
        half = dim // 2
        freqs = np.exp(
            -math.log(max_period) * np.arange(0, half, dtype="float32") / half
        )
        args = np.array(timesteps) * freqs
        embedding = np.concatenate([np.cos(args), np.sin(args)])
        return tf.convert_to_tensor(embedding.reshape(1, -1))

    def reverse_step(self, x, e_t, index, a_t, a_prev, temperature, seed):
        sigma_t = 0
        sqrt_one_minus_at = math.sqrt(1 - a_t)
        pred_x0 = (x - sqrt_one_minus_at * e_t) / math.sqrt(a_t)

        # Direction pointing to x_t
        dir_xt = math.sqrt(1.0 - a_prev - sigma_t**2) * e_t
        noise = sigma_t * tf.random.normal(x.shape, seed=seed) * temperature
        x_prev = math.sqrt(a_prev) * pred_x0 + dir_xt
        return x_prev, pred_x0

    def get_starting_parameters(self, timesteps, batch_size, seed, latent):
        n_h = self.img_height // 8
        n_w = self.img_width // 8
        alphas = [_ALPHAS_CUMPROD[t] for t in timesteps]
        alphas_prev = [1.0] + alphas[:-1]
        if latent is None:
            latent = tf.random.normal((batch_size, n_h, n_w, 4), seed=seed)
        return latent, alphas, alphas_prev

    # predict latent error e_t from conditional and unconditional error
    def diffuse_noise(
        self,
        latent,
        t,
        context,
        unconditional_context,
        unconditional_guidance_scale,
        batch_size,
    ):
        timesteps = np.array([t])
        t_emb = self.timestep_embedding(timesteps)
        t_emb = np.repeat(t_emb, batch_size, axis=0)

        unconditional_latent = self.denoiser.predict_on_batch(
            [latent, t_emb, unconditional_context]
        )
    
        # latent 32x64,64,4
        # t_emb  32,320
        # contex 1,77,768
        latent = self.denoiser.predict_on_batch([latent, t_emb, context])

        return unconditional_latent + unconditional_guidance_scale * (
            latent - unconditional_latent
        )

    def in_cache(self, latent):
        return latent is not None
        
    def generate(self, context, unconditional_context, s):

        #@cached sample latent
        if s['cache_enabled']:
            n = s['sample_n']
            latent = TensorDebug().n(n).load(home=None, index=0).dis().latent
            s['latent'] = latent

            if self.in_cache(latent): 
                iter_steps = s['num_steps'] * 2 - 1
                # 49 = last iteration
                # 00 = first iteration
                if s['start_index'] == iter_steps:
                    return latent
            else:
                latent = None

        # [1, 40, 80, 120, ... , 960, 1000]
        timesteps = np.arange(1, 1000, 1000 // s['num_steps'])
        latent, alphas, alphas_prev = self.get_starting_parameters(
            timesteps, s['batch_size'], s['seed'], s['latent']
        )
        scheduler = DDIMScheduler(timesteps)

        if s['latent_mask'] is not None:
            print("24. apply mask to starting latent ...")
            latent = TensorDebug().apply_mask_to_latent(s['latent_mask'], latent)

        # start from cached embeddinges based on start index (index=48 no cache)
        start_index = len(timesteps) - s['start_index']
        # Diffusion stageo
        p = list(enumerate(timesteps))[::-1]
        p = p[start_index::]
        progbar = tqdm(p)

        for index, timestep in progbar:
            print("step {0}".format(index))
            progbar.set_description(f"{index:3d} {timestep:3d}")

            noise_pred = self.diffuse_noise(latent, timestep, context, unconditional_context, 
                s['u_guidance_scale'],
                s['batch_size'],
            )

            #a_t, a_prev = alphas[index], alphas_prev[index]
            #latent, pred_x0 = self.reverse_step( latent, e_t, index, a_t, a_prev, s['temperature'], s['seed'])
            latent, pred_x0 = scheduler.reverse_step(noise_pred, index, latent)

            # @cached: only once if cache is enabled
            if s['cache_enabled']:
                n = s['sample_n']
                print("caching sasmple ", (n, index))
                TensorDebug().n(n).save_as_latent(latent, index)

        return latent

    def equal(self, A, B):
        if A is None or B is None:
            return 1.0
        return np.dot(A,B)/(norm(A)*norm(B)+0.0000000001)

def test():
    latents = [None, np.random.rand(64*64*4).reshape(64, 64, 4)]
    dm = Diffusion()

    t = None
    u = None
    for latent in latents:
        latent = dm.denoise(latent)
        print(latent[0:5])
