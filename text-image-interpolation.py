import tensorflow as tf
import numpy as np
from numpy.linalg import norm
import os
import cv2 as cv
from tensorflow import keras

from xray.text_encoder import TextContext
from xray.latent_denoiser import Diffusion 
from xray.image_auto_encoder import LatentDecoder
from xray.tensor_debug import TensorDebug
from hashlib import *

TEST_HOME = 'test/tsi2i'
TEST_HOME = 'test'
# TODO: TensorDebug().set_config({'home' : 'test/tsi2i'})

def sample_home(n):
    return os.path.join(TEST_HOME, str(n))

def debug(prompts, latents):
    #for i, l in enumerate(latents):
    #    TensorDebug().save_as_image("image{0:03d}.png".format(i), l, prompt=prompts[i])
    pass

def o(n):
    num_steps = 25 
    opt = {
        'u_guidance_scale' : 7.5, 
        'batch_size'  : 1, 
        'num_steps'   : num_steps, 
        'temperature' : 1,
        'seed'        : 12345678, 
        # cache: new format
        # 'tensor' : TensorDebug().load("test/0", index=49).tile((0, 0), (64, 32)).dis().latent,
        # 'mask' : TensorDebug().load_as_mask('test/0/mask64x64.00.png'),
        'cache'       : { 
            'enabled' : True, 
            'latent' : None, 'start_index' : 0, 'latent_mask' : None 
        },
        # cache: old format: 
        'sample_n'      : n,
        'cache_enabled' : True,
        'start_index'   : (num_steps*2 -1), 
        'latent_mask'   : None, 
    }
    return opt

def interpolate_encodings2(va, vb, n_steps=5):
    v0 = va[0, :, :]
    v1 = vb[0, :, :]
    maxdiff = np.argmax(np.abs(v0 - v1))
    a = 1e-5
    d = 44
    vi = [ v0[d] + a*(n/n_steps) for n in range(n_steps)]
    for i in range(4):
        res = va[0,d,:] = vi[i]
    return vi

def interpolate_encodings(v0, v1, n_steps=5):
    a = 1e-5
    vi = [ v0 + a*(n/n_steps) * (v1 - v0) for n in range(n_steps)]
    return vi

def main():
    tc = TextContext()
    dm = Diffusion()

    # load S* prompts  
    prompts = [
        'cat on a sofa',
        'cute cat on a couch' 
    ]

    print("1. generate text encodings ...")
    #. load S* prompts and generate their v* encodings
    vs = [tc.encode(p) for p in prompts]
    v_unc = tc.encode(None)
    tc.debug([v_unc])

    if True:
        inter_steps = 4
        vs = interpolate_encodings2(vs[0], vs[1], inter_steps) 
        # replicate prompts to match interpolation steps
        a = [prompts[0]]*(inter_steps//2)
        a.extend([prompts[1]]*(inter_steps//2))
        inter_prompts = a
    else:
        inter_prompts = prompts

    tc.debug(vs)

    if True:
        # inference loop to generate prompts encoding and image encodings
        print("2. generate latents")
        latents = [dm.generate(v, v_unc, o(n)) for n, v in enumerate(vs)]
        debug(prompts, latents)
    
    else:

        # 1. TODO
        # load user target images and force latents to match *v encodings
        target_images = [
            np.random.rand(512*512*3).reshape(512, 512, 3),
            np.random.rand(512*512*3).reshape(512, 512, 3),
    ]

    if True:
        print("3. decoder ....")
        decoder = LatentDecoder()
        for n, latent in enumerate(latents):
            image = decoder.decode(latent)
            prompt = inter_prompts[n]
            title = TensorDebug().prompt_to_title(prompt)
            filepath = os.path.join(TEST_HOME, "image_decoded.{0:03d}.{1}.png".format(n, title))
            cv.imwrite(filepath, image)
        


main()