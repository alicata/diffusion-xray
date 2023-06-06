import tensorflow as tf
import numpy as np
from numpy.linalg import norm

from tensorflow import keras

#from text_encoder import TextEncoder
from stable_diffusion_tf.clip_encoder import CLIPTextTransformer as TextEncoder
from stable_diffusion_tf.clip_tokenizer import SimpleTokenizer
from stable_diffusion_tf.constants import _UNCONDITIONAL_TOKENS
from hashlib import md5 
import os


class TextContext:
    def __init__(self):
        self.MAX_PROMPT_LENGTH = 77
        self.MAX_EMBEDDING_LENGTH = 768

        self.input_pos_ids = None
        fpath = self.get_model_uri() 
        # tokenization
        self.tokenizer = SimpleTokenizer() 
        self.tokens = list()
        # encoding
        self.text_encoder = self.create_text_encoder()
        self.text_encoder.load_weights(fpath)

        os.makedirs('d:/cache/text', exist_ok=True)

    def o(self):
        return {
            'input_size' : self.MAX_PROMPT_LENGTH,
            'output_size' : self.MAX_EMBEDDING_LENGTH,
        }

    def as_keras_model(self):
        return self.text_encoder

    def get_model_uri(self):
        o = "https://huggingface.co/fchollet/stable-diffusion/resolve/main/text_encoder.h5"
        weights_fpath = keras.utils.get_file(
            origin=o,
            file_hash="d7805118aeb156fc1d39e38a9a082b05501e2af8c8fbdc1753c9cb85212d6619",
            cache_dir="d:/.keras/"
        )
        return weights_fpath

    def pos_ids_as_input(self):
        #return tf.convert_to_tensor([list(range(MAX_PROMPT_LENGTH))], dtype=tf.int32)
        if self.input_pos_ids is None:
            self.input_pos_ids = keras.layers.Input(shape=(self.MAX_PROMPT_LENGTH,), dtype="int32")
        return self.input_pos_ids 

    def create_text_encoder(self):
        input_word_ids = keras.layers.Input(shape=(self.MAX_PROMPT_LENGTH,), dtype="int32")
        embeds = TextEncoder()([input_word_ids, self.pos_ids_as_input()])
        text_encoder = keras.models.Model([input_word_ids, self.pos_ids_as_input()], embeds)
        return text_encoder

    def generate_pos_ids(self, batch_size=1):
        # Encode prompt tokens (and their positions) into a "context vector"
        pos_ids = np.array(list(range(self.MAX_PROMPT_LENGTH)))[None].astype("int32")
        pos_ids = np.repeat(pos_ids, batch_size, axis=0)
        return pos_ids

    def encode_unconditioned(self, batch_size=1):
        # Encode unconditional tokens (and their positions into an
        # "unconditional context vector"
        unconditional_tokens = np.array(_UNCONDITIONAL_TOKENS)[None].astype("int32")
        unconditional_tokens = np.repeat(unconditional_tokens, batch_size, axis=0)
        self.unconditional_tokens = tf.convert_to_tensor(unconditional_tokens)

        pos_ids = self.generate_pos_ids()
        unconditional_context = self.text_encoder.predict_on_batch(
            [self.unconditional_tokens, pos_ids]
        )        
        return unconditional_context

    def hash_prompt(self, prompt):
        return md5(str(prompt).encode('utf-8')).hexdigest()
        
    def load_cache(self, prompt):
        hpath = "d:/cache/text/text_emb_{0}.npy".format(self.hash_prompt(prompt))
        try:
            v = np.load(hpath)
        except:
            v = None
        return v 

    def save_cache(self, prompt, emb):
        hpath = "d:/cache/text/text_emb_{0}.npy".format(self.hash_prompt(prompt))
        np.save(hpath, emb)

    def encode(self, prompt=None, batch_size=1):
        # TODO: work with batches
        cached = self.load_cache(prompt)
        if cached is not None:
            return cached

        if prompt is None:
            print("unconditioned encoding.")
            self.tokens = []
            return self.encode_unconditioned(batch_size)

        inputs = self.tokenizer.encode(prompt)
        self.tokens = inputs

        words = inputs + [49407] * (self.MAX_PROMPT_LENGTH- len(inputs))
        words = np.array(words)[None].astype("int32")
        words = np.repeat(words, batch_size, axis=0)
        # words = tf.convert_to_tensor([words], dtype=int32)

        # Encode prompt tokens (and their positions) into a "context vector"
        pos_ids = self.generate_pos_ids()
        contex = self.text_encoder.predict_on_batch([words, pos_ids])
        print("context sum : "+str(prompt)+" ", np.sum(contex.flatten()))

        self.save_cache(prompt, contex) 
        return contex

    def equal(self, A, B):
        if A is None or B is None:
            return 1.0
        return np.dot(A,B)/(norm(A)*norm(B)+0.0000000001)

    def hash(self, m):
        h = [md5(r).hexdigest() for r in m[0,:,:]]
        h = md5(np.array(h)).hexdigest()
        return h

    def debug(self, vs):
        print("debug text encodings: ", len(vs))
        u = None
        t = None
        for i, v in enumerate(vs):
            # each matrix v = 1x77x768
            print("------------------v[{0}]------------- {1}".format(i, v.shape))
            print("                  {0}                    ".format(self.hash(v)))
            print("---------------------")
            print("knowledge column:  0    1  ...     ...    ...      ...    ...     ...   768")
            print("{0:03d}-tok1: {1} {2}".format(i, md5(v[0, 1, :]).hexdigest(), v[0, 1, 0:9])) #*A* photo of"
            print("{0:03d}-tok2: {1} {2}".format(i, md5(v[0, 2, :]).hexdigest(), v[0, 2, 0:9])) # A *photo*

            # similiarity with previous vector
            word_count = len(v) - 2
            s = np.ceil(100*self.equal(t, u))
            t = u
            for w in v[0,1:(word_count + 2),:]:
                w *= 10
                w = w[0:16].astype(np.int16)
                print("s={0}:  {1}".format(s, w))

def test():
    prompts = ["coyote in woods",
            "coyote in forest",
            "wolf in forest",
            "wolf in woods",
            "wolf along freeway"]
    tc = TextContext()

    t = None
    u = None
    for prompt in prompts:
        context_matrix = tc.encode(prompt)
        word_count = len(tc.tokens) - 2
        tokens = tc.tokens[1:(word_count+1)]
        print("prompt   :", prompt)
        print("tokens   :", tokens )

        s = np.ceil(100*TextContext().equal(t, u))
        t = u

        for w in context_matrix[0,1:(word_count + 2),:]:
            u = w
            w *= 10
            w = w[0:16].astype(np.int16)
            print("s={0}:  {1}".format(s, w))
