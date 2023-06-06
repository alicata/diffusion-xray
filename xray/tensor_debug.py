from copy import deepcopy
import cv2 as cv
import numpy as np
import os
import psutil

def mem_info():
    avail = psutil.virtual_memory().available
    perc  = psutil.virtual_memory().available * 100 / psutil.virtual_memory().total
    info = "\nhost[...    avail:{0:03f}       perc:{1:03f}  ]".format(avail/(1024**2), perc)
    return info

class Util:
    def __init__(self):
        pass

class TensorDebug:
    def __init__(self, home='test'):
        self.o = {
            'img' : {'reso' : (640, 640)},
            'emb' : {}
        }
        self.dbg_level = 1
        self.home = home
        self.latent = None
        self._n = 0

        os.makedirs(self.home, exist_ok=True)

    def n(self, _n):
        self._n=_n
        return self
            
    def prompt_to_title(self, prompt):
        # <S*> -> _S~_
        s = prompt
        s = s.replace('<', '-')
        s = s.replace('>', '-')
        s = s.replace('*', '~')
        return s
    
    def latent_emb_path(self, index=0):
        return os.path.join(self.home, "latent-{1}-emb-{0:03d}.npy".format(index, self._n))

    def latent_img_path(self, index=0):
        return os.path.join(self.home, "latent-{1}-img-{0:03d}.png".format(index, self._n))

    def to_image(self, latent, channels=[0,1,2]):
        # from [0, 64, 64, 4] to [64, 64, 3]
        np_array = np.array((latent[0,:,:,0:3])*31 + 128).astype(np.uint8) 
        return np_array

    def to_numpy_tensor(self, latent):
        np_array = np.array(latent) 
        return np_array

    # Save latent tf tensor as latent tensor numpy array and rgb image
    # 1. latent as tf/pt sensor
    # 2. latent as numpy array
    # 3. latent as bgr image 
    def save_as_latent(self, latent, index):
        print("save ", self.latent_img_path())
        self.save(self.latent_emb_path(index), latent)

        im = self.to_image(latent)
        self.save_as_image(self.latent_img_path(index), latent)
        
        if self.dbg_level > 0:
            print("latent #{0} shape:{1} mem={2}".format(index, latent.shape, mem_info()))
        self.dis(im)
        return self

    def get_latent(self):
        if self.latent == (None,):
            self.latent = None
        return self.latent

    def dis(self, im=None, index=0):
        if im is None:
            if self.latent is None:
                return self
            im = self.to_image(self.latent)
        disp = cv.resize(im, self.o['img']['reso'], interpolation=cv.INTER_NEAREST) 
        cv.putText(disp, "t " + str(index), (10, 30), fontScale=2.0, fontFace=cv.FONT_HERSHEY_PLAIN, color=(255, 0, 0))
        cv.imshow("latent", disp)
        cv.waitKey(10)
        return self

    def save_as_image(self, filepath, latent, prompt=None, channels=[0,1,2]):
        im = self.to_image(latent, channels)
        cv.putText(im, prompt, (5, 30), fontScale=0.5, fontFace=cv.FONT_HERSHEY_PLAIN, color=(0, 0, 0))
        cv.putText(im, prompt, (6, 31), fontScale=0.5, fontFace=cv.FONT_HERSHEY_PLAIN, color=(0, 255, 0))
        cv.imwrite( filepath, im)

    def load_as_image(self, filepath, channels=None):
        im = cv.imread(filepath)  
        return im 

    def load_as_mask(self, filepath):
        return self.load_as_image(filepath)
    
    def merge_image_into_latent(self, image, latent, channels=None):
        # TODO
        return None
    
    def apply_mask_to_latent(self, mask, latent):
        """
        combine mask with latent tensor, on each dimension 

        latent tensor is [0, 64, 64, 4] float32
        latent image  is [64, 64, 3]    uint8
        mask          is [64, 64]       float32

        mask will be combined with latent tensor dimensions 

        """
        if isinstance(mask, str):
            mask = cv.imread(mask)
        mask = mask[:,:,0]
        b, w, h, d = latent.shape 

        maskf = mask.astype(np.float32) / 255.0
        # todo: float' object cannot be interpreted as an integer
        # threshold and not max that returns a single float max
        a = 1.000000 - 1e-8 
        r = (np.random.rand(64, 64).astype(np.float32) * a + (1.0 - a))

        # NOTE. clip to make sure mask is within bounds
        # BUG. markf * r was oversaturating the latent and causing gray images. 
        maskf = np.clip(maskf + r, 0.0, 1.0)

        for i in range(d):
            latent[0,:,:,i] = latent[0,:,:,i] * maskf.astype(type(latent.flatten()[0]))
        return latent 

    def tile(self, origin=(0,0), size=(8, 8)):
        x, y = origin
        w, h = size
        latent = self.latent
        if latent is not None:
            n = latent.shape[1]//h
            m = latent.shape[2]//w
            # copy tile over entire latent d-dimensional tensor 
            for i in range(4):
                t = np.tile(latent[0,y:(h+y),x:(x+w),i], (n, m))
                latent[0,:,:,i] = t

        self.latent = latent
        return self
        

    def save(self, filepath, latent):
        emb = self.to_numpy_tensor(latent)
        np.save(filepath, emb) 
    
    def load_as_latent(self, home, index=0):
        self.home=home
        latent_path = self.latent_emb_path(index)
        print('load cached latent embedding ', latent_path)
        try:
            latent = np.load(latent_path)
        except:
            latent = None
        return latent

    def load(self, home, index=0):
        # default home
        if home is None:
            home = self.home
        self.latent = self.load_as_latent(home, index)
        return self

    def random_sample(self):
        # sample used for testing and debuggin
        l = np.random.rand(64*64*4).reshape(1, 64, 64, 4)
        l = (l - 0.5) * 8
        return l

def test():
    dump = TensorDebug()   
    s = dump.random_sample()
    print("sample ", s)
    dump.latent(s, 0)
