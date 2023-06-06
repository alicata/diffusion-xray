import numpy as np

from xray.tensor_debug import TensorDebug
from xray.tensor_debug import mem_info
from xray.text_encoder import TextContext
from xray.latent_denoiser import Diffusion
from xray.image_auto_encoder import LatentDecoder 


class Text2Image:
    def __init__(self, img_height=1000, img_width=1000, jit_compile=False):
        print(".")
        print(".")
        print(".")
        print(".")
        print("initialize Text2Image")
        self.img_height = img_height
        self.img_width = img_width

        print("00. initialize contructor to load models ...")
        text_encoder, diffusion_model, decoder = get_models(img_height, img_width)
        self.text_encoder = text_encoder
        self.diffusion_model = diffusion_model
        self.decoder = decoder

        self.o = {}
        self.o['cache'] = { 
            'context' : None, 
            # latent tensor chache behavior: no cache, continue from scheduler, continue from decoder
            'latent' :  { 
                'enable_scheduler' : True,
                'tensor' : None, #TensorDebug().load("test/0", index=49).tile((0, 0), (64, 32)).dis().latent,
                'mask' : None, #TensorDebug().load_as_mask('test/0/mask64x64.00.png'),
                'index' : 49 
            }
        }

        if jit_compile:
            self.text_encoder.as_keras_model().compile(jit_compile=True)
            self.diffusion_model.as_keras_model().compile(jit_compile=True)
            self.decoder.as_keras_model().compile(jit_compile=True)
        print("10. initialization completed. ")

    # sd.sample()
    def generate(
        self,
        prompt,
        batch_size=1,
        num_steps=25,
        unconditional_guidance_scale=7.5,
        temperature=1,
        seed=None,
    ):
        print("20. start generation process ...")
        latent = self.o['cache']['latent']['tensor']

        if self.o['cache']['latent']['enable_scheduler']:
            print("21. generate context .. ")
            context, unconditional_context = self.generate_context(prompt, batch_size)
            s = {'u_guidance_scale' : unconditional_guidance_scale, 
                'batch_size' : batch_size, 
                'num_steps' : num_steps, 
                'temperature' : temperature,
                'cache_enabled' : False,
                #'sample_n'      : n,
                'latent' : latent,
                'start_index' : self.o['cache']['latent']['index'],
                'latent_mask' : self.o['cache']['latent']['mask'],
                'seed' : seed}
            
            print("22. generate latent .. ")
            latent = self.generate_latent(context, unconditional_context, s)

        print("23. generate image .. ")
        image = self.decoder.decode(latent)
        # return a list of 1 image only
        return [image] 

    def generate_latent(self, context, unconditional_context, s):
        latent = self.diffusion_model.generate(context, unconditional_context, s)
        return latent

    def generate_context(self, prompt, batch_size):
        cond_context = self.text_encoder.encode(prompt)
        unco_context = self.text_encoder.encode(None)
        return cond_context, unco_context 



def get_models(img_height, img_width, download_weights=True):

    print("01. diffusion create UNet")
    dm = Diffusion()
    print("01. loaded. mem usage {0}".format(mem_info()))

    print("02. text encoder create")
    tc = TextContext()
    print("02. loaded. mem usage {0}".format(mem_info()))

    print("03. creation of Decoder")
    decoder = LatentDecoder()
    print("03. loaded. mem usage {0}".format(mem_info()))

    print("04. all model loaded. mem usage {0}".format(mem_info()))
    return tc, dm, decoder
