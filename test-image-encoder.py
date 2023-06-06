import numpy as np
import cv2 as cv

from xray.tensor_debug import TensorDebug
from xray.image_auto_encoder import ImageEncoder
from xray.image_auto_encoder import LatentDecoder 
from xray.encoder_visualizer import plot_latent
from xray.encoder_visualizer import plot_image
from xray.encoder_visualizer import pause 


def opts(n=0):
    opt = {
        'test_file2'  : 'd:/data/img/test.png',
        'test_file'   : 'd:/data/img/00001.jpg',
        'batch_size'  : 1, 
        'seed'        : 12345678, 
        'cache'       : { 
            'enabled' : False, 
        },
    }
    return opt


def main():
    o = opts()

    if o['test_file'] is None:
        images = [(128 + 127*np.random.rand(512*512*3)).reshape(o['batch_size'], 512, 512, 3).astype(np.uint8)]
        prompts = ['random512x512x3']
    else:
        images = [cv.imread(o['test_file']).reshape(1, 512, 512, 3)]
        prompts = [o['test_file']]

    ie = ImageEncoder()
    de = LatentDecoder()

    for image in images:
        plot_image(image[0, :, :, :], 'input image')

        print("encoding image ...")
        latent = ie.encode(image)
        plot_latent(latent, 'encoded latent')

        print("decode latent ...")
        decoded = de.decode(latent) 
        plot_image(decoded, 'decoded image')

    pause() 

main()

