import numpy as np
import cv2 as cv
import copy

from xray.tensor_debug import TensorDebug
from xray.image_auto_encoder import ImageEncoder
from xray.image_auto_encoder import LatentDecoder 
from xray.encoder_visualizer import plot_latent
from xray.encoder_visualizer import plot_image
from xray.encoder_visualizer import pause 


def opts(n=0):
    opt = {
        'test_file'   : 'd:/data/img/test.png',
        'batch_size'  : 1, 
        'seed'        : 12345678, 
        'num_samples' : 64*64,
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
        latent_original = copy.deepcopy(latent)

        # noise min/max value range (variance)
        m1, m2 = np.min(latent.flatten()), np.max(latent.flatten())
        lmax = min(np.abs(m1), m2)
        num_samples = o['num_samples'] 

        # interval scales (widths) from +/- 0 to +/- lmax
        # N=2 : [0,0] -> [-lmax/2 +lmax/2] -> [-lmax, +lmax] 
        N =  5 # scale of noise min/max  0, 0.3, 0.6, 0.9, 1.3, 1.6, 1.9, ... 3.2
        SL = 3 # sample levels 0, 512, 1024, 2048, 4096

        for i in range(N+1):
            alpha = np.float(i) / np.float(N)
            print("random noise in range -{0} to +{1}".format(alpha*lmax, alpha*lmax))

            # create noise range interval to apply to latent
            for j in range(SL):
                # noise density increases from 0 to max_noise_samples
                beta = np.float(j) / np.float(SL)
                n = int(num_samples  * beta)

                # restart corruption from scratch, do not acculate noise
                latent = copy.deepcopy(latent_original)

                # corrupot 64x64 tensor with noise in +/- lmax range
                ii = (np.random.rand(n, 2) * 63).astype(np.int16)
                vv = (np.random.rand(n, 4) * 2 - 1)*lmax*alpha
                latent[0, ii[:,0], ii[:,1], :] = vv 
            
                plot_latent(latent, 'encoded latent')
                print("decode latent ... {0} {1}".format(n, lmax*alpha))
                decoded = de.decode(latent) 
                plot_image(decoded, 'decoded image', waittime_ms=0)

    pause() 

main()

