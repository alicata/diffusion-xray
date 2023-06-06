import numpy as np
from numpy.linalg import norm

from xray.text_encoder import TextContext
from xray.tensor_debug import TensorDebug
from hashlib import *

TEST_HOME = 'test/tsi2i'
TEST_HOME = 'test'
# TODO: TensorDebug().set_config({'home' : 'test/tsi2i'})

from xray.encoder_visualizer import plot_embedding


    

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
            'latent' : None, 'start_index' : 0, 'latent_mask' : None ,
            'encoding' : 'text-encoding-*-.npy',
        },
        # cache: old format: 
        'sample_n'      : n,
        'cache_enabled' : True,
        'start_index'   : (num_steps*2 -1), 
        'latent_mask'   : None, 
        # text
        'distance' : [],
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

    exp1 = [
        'this is a photo of a cat on a sofa',
        'cute cat on a couch', 
        'a couch with a cat on it',
        'a photo of cat on the couch',
        'a cat on the chair',
        # distant
        'eagle flying in the sky among clouds',
        'this car engine is broken', 
        # unconditional
        None,
    ]
    
    exp2 = [
        None,
        "catt",
        'dog',
        'dog dog',
        'dog dog dog',
        'dog dog dog dog',
        'dog dog dog dog dog',
        'dog dog dog dog dog dog dog',
        'dog dog dog dog dog dog dog dog dog',

        "cat",
        "cat cat",
        "cat cat cat",
        "cat cat cat cat cat",
        "cat cat cat cat cat cat cat",
        "cat cat cat cat cat cat cat cat cat",
        "cat cat cat cat cat cat cat cat cat cat cat cat cat",
        "cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat",
    ]

    exp3 = [
        None,
        "cat",
        "cat cat",
        "a cat",
        "a cat with",
        "a cat on",
        "a cat in",
        "one cat",
        "the cat",
        "this cat",
        "that cat",
        "that is a",
        "that is a cat",

        "cat",
        "cat cat",
        "cats",
        "two cats",
        "2 cats",
        "three cats",
        "3 cats",
        "four cats",
        "many cats",

        "cat",
        "a cat",
        "a cat next",
        "a cat next to",
        "a cat next to another",
        "a cat next to anoter cat",        
        "a cat next to anoter cat.",        
        "a dog next to anoter dog.",        
    ]

    exp4 = [
        None,
        "cat", "dog", "budgie", "eagle",
        "cat", "dog", "budgie", "eagle",
        "cat", "dog", "budgie", "eagle",

        'eagle',
        'eagle eagle',
        'eagle eagle eagle ',
        'eagle eagle eagle eagle ',
        'cat eagle',

        'elephant',
        'zebra',
        'deer',
        'mustang',
        'goat',
        'horse',
        'donkey',
    ]

    exp5 = [
        None,
        "sofa",
        "couch",
        'armchair',
        "chair",
        "stool",
        "seat",
        "car seat",

        "car",
        "truck",
        "bike",
        "bicycle",
    ]

    exp6 = [
        None,
        "a",
        "any",
        "the",
        "this",
        "that",
        "those",
        "these",
        "for",
        "on",
        "in",
        "out",
        "far",

        "a cat on a sofa",
        "a cat on a couch",
        "a cat on a chair",
    ]

    exp7 = [
        None,
        "the beach was occupied by people",
        "the beach was occupied by dogs",
        "the beach was occupied by seagalls",
        "the beach was occupied by seals",
        "the beach was occupied by crabs",
        "the beach was occupied by bags",
        "the beach was occupied by rockets",
    ]

    exp8 = [None,
        "in",
        "Paris",
        "in Paris",
        "London",
        "in London",
        "Rome",
        "in Rome",
        "not in Rome",

        "The Big Ben is in London",
        "The Eiffel tower is in Paris",
        "The Colosseum is in Rome",
        "The Vatican is in Rome",

        "The Eiffel tower is in Rome",
    ]


    prompts = exp3 # cat position
    prompts = exp4 # animal types
    prompts = exp5 # furniture 
    prompts = exp6 # cat and furniture 

    prompts = exp7 # place with different things
    prompts = exp8 # place locations:w


    print("1. generate text encodings ...")
    tc = TextContext()
    vs = [tc.encode(p) for p in prompts]
    tc.debug(vs)

    plot_embedding(vs, prompts)

main()

