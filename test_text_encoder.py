import numpy as np
from xray.text_encoder import TextContext
from xray.text_encoder import simil 

def test():
    prompts = [None, "coyote in woods",
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

        print("prompt   :", prompt)
        if prompt is not None:
            tokens = tc.tokens[1:(word_count+1)]
            print("tokens   :", tokens )
        else:
            print("No tokens [unconditional]")

        s = np.ceil(100*simil(t, u))
        t = u

        for w in context_matrix[0,1:(word_count + 2),:]:
            #s = np.floor(np.sum(w))
            u = w

            w *= 10
            w = w[0:16].astype(np.int16)
            print("s={0}:  {1}".format(s, w))

test()