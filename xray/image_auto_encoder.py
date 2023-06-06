import numpy as np
import cv2 as cv
import os
from hashlib import md5

from tensorflow import keras
from stable_diffusion_tf.autoencoder_kl import Decoder, Encoder 
from xray.tensor_debug import TensorDebug

MODEL_INTERNALLY_USES_BGR = True

class LatentDecoder():
    def __init__(self):
        self.IMAGE_DIM = (512, 512)
        self.LATENT_DIM = (64, 64, 4)

        fpath = self.get_model_uri() 
        self.model = self.create()
        self.model.load_weights(fpath)
        os.makedirs('d:/cache/latent', exist_ok=True)

    def create(self):
        n_h = self.IMAGE_DIM[0] // 8
        n_w = self.IMAGE_DIM[1] // 8
        latent = keras.layers.Input((n_h, n_w, 4))
        decoder = Decoder()
        decoder = keras.models.Model(latent, decoder(latent))
        return decoder 

    def o(self):
        # TODO: is this correct? or input / output reversed? 
        return {
            'input_size' : self.IMAGE_DIM,
            'output_size' : self.LATENT_DIM,
        }

    def as_keras_model(self):
        return self.model

    def get_model_uri(self):
        weights_fpath = keras.utils.get_file(
            origin="https://huggingface.co/fchollet/stable-diffusion/resolve/main/decoder.h5",
            file_hash="6d3c5ba91d5cc2b134da881aaa157b2d2adc648e5625560e3ed199561d0e39d5",
            cache_dir="d:/.keras/"
        )
        return weights_fpath

    def decode(self, latent):
        # Decoding stage
        print("ImageDecoder latent shape = ", latent.shape)
        decoded = self.model.predict_on_batch(latent)
        decoded = ((decoded + 1) / 2) * 255
        image =  np.clip(decoded[0,:,:,:], 0, 255).astype("uint8")
        print("ImageDecoder decoded = ", decoded.shape)
        print("ImageDecoder image shape = ", image.shape)
        if MODEL_INTERNALLY_USES_BGR:
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        return image 

class ImageEncoder():
    def __init__(self):
        self.IMAGE_DIM = (512, 512, 3)
        self.LATENT_DIM = (64, 64, 4)

        fpath = self.get_model_uri() 
        self.model = self.create()

    def create(self):
        n_h = self.IMAGE_DIM[0] // 8
        n_w = self.IMAGE_DIM[1] // 8
        rgb = keras.layers.Input((self.IMAGE_DIM[0], self.IMAGE_DIM[1], 3)) 

        encoder = Encoder(512, 512)
        fpath = self.get_model_uri()
        encoder.load_weights(fpath)
        print("---------DEBUG-START-Encoder-------")
        encoder.summary()
        encoder.layers
        encoder.layers[1].weights
        print("---------DEBUG-END--------")

        model = keras.models.Model(rgb, encoder(rgb))

        print("---------DEBUG-START-KERAS-MODEL-------")
        model.summary()
        model.layers
        model.layers[1].weights
        print("---------DEBUG-END--------")

        return model 

    def o(self):
        # TODO: input and output reversed?
        return {
            'input_size' : self.IMAGE_DIM,
            'output_size' : self.LATENT_DIM,
        }

    def as_keras_model(self):
        return self.model

    def get_model_uri(self):
        weights_fpath = keras.utils.get_file(
            origin="https://huggingface.co/fchollet/stable-diffusion/resolve/main/vae_encoder.h5",
            file_hash="c60fb220a40d090e0f86a6ab4c312d113e115c87c40ff75d11ffcf380aab7ebb",
            cache_dir="d:/.keras/"
        )
        return weights_fpath

    def encode(self, image):
        # TODO: work with batches
        cached = self.load_cache(image)
        if cached is not None:
            return cached

        # check for [1, h, w, d] shape
        if len(image.shape) == 3:
            image = np.expand_dims(image, 0) 
        original_image = image

        if MODEL_INTERNALLY_USES_BGR:
            image = cv.cvtColor(image[0,:,:,:], cv.COLOR_RGB2BGR)
            image = np.expand_dims(image, 0) 
        image = ((image.astype(np.float32) / 255.0) - 0.5) * 2.0

        latent = self.model.predict_on_batch(image)
        self.save_cache(original_image, latent) 
        return latent

    def hash_image(self, rgb):
        return md5(rgb).hexdigest()

    def save_cache(self, image, latent):
        if len(image.shape) == 4:
            rgb = image[0,:,:,:]
        else:
            rgb = image
        hpath = "d:/cache/latent/latent_emb_{0}.npy".format(self.hash_image(rgb))
        np.save(hpath, latent)

    def load_cache(self, image):
        if len(image.shape) == 4:
            rgb = image[0,:,:,:]
        else:
            rgb = image
        hpath = "d:/cache/latent/latent_emb_{0}.npy".format(self.hash_image(rgb))
        try:
            v = np.load(hpath)
            print("loaded ", hpath)
        except:
            v = None
        return v

def test():

    images = [np.random.rand(512*512*3).reshape(512, 512, 3)]
    # TODO: images = [np.random.rand(512*512*3).reshape(1, 512, 512, 3)]
    ie = ImageEncoder()
    for image in images:
        latent = ie.encode(image) 
        TensorDebug().dis(image)
        TensorDebug().dis(latent)
        


    # random input latent into image
    latents = [np.random.rand(64*64*4).reshape(1, 64, 64, 4)]
    ld = LatentDecoder()

    for latent in latents:
        image = ld.decode(latent)
        print(image[0:5])

if __name__ == "__main__":
    test()