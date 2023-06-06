
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm


import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
import cv2 as cv

def selection(XY, limitXY=[[-2,+2],[-2,+2]]):
        XY_select = []
        for elt in XY:
            if elt[0] > limitXY[0][0] and elt[0] < limitXY[0][1] and elt[1] > limitXY[1][0] and elt[1] < limitXY[1][1]:
                XY_select.append(elt)

        return np.array(XY_select)

def plot_3d_histom():
    x = np.random.randn(10)
    y = np.random.randn(10)
    XY = np.stack((x,y),axis=-1)

    n = 77 
    m = 768 

    XY_select = selection(XY, limitXY=[[-m,+m],[-n,+n]])
    xAmplitudes = np.array(XY_select)[:,0]#your data here
    yAmplitudes = np.array(XY_select)[:,1]#your other data here

    fig = plt.figure() #create a canvas, tell matplotlib it's 3d
    ax = fig.add_subplot(111, projection='3d')

    hist, xedges, yedges = np.histogram2d(x, y, bins=(m,n), range = [[-m, m],[-n,+n]]) # you can change your bins, and the range on which to take data
    # hist is a 7X7 matrix, with the populations for each of the subspace parts.
    xpos, ypos = np.meshgrid(xedges[:-1]+xedges[1:], yedges[:-1]+yedges[1:]) -(xedges[1]-xedges[0])

    xpos = xpos.flatten()*1./2
    ypos = ypos.flatten()*1./2
    zpos = np.zeros_like (xpos)

    dx = xedges [1] - xedges [0]
    dy = yedges [1] - yedges [0]
    dz = hist.flatten()

    cmap = cm.get_cmap('jet') # Get desired colormap - you can change this!
    max_height = np.max(dz)   # get range of colorbars so we can normalize
    min_height = np.min(dz)
    # scale each z to [0,1], and get their rgb values
    rgba = [cmap((k-min_height)/max_height) for k in dz] 

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=rgba, zsort='average')
    plt.title("X vs. Y Amplitudes for ____ Data")
    plt.xlabel("My X data source")
    plt.ylabel("My Y data source")
    plt.savefig("Your_title_goes_here")
    plt.show()

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.view_init(-90, 0)
ax.zaxis.set_major_formatter('{x:.02f}')

def plot_embedding_surf(embedding=np.random.rand(77,768)*20-10, text=None, limits=None):
    # TODO; adapt to [1,N,M] shape
    N = embedding.shape[0]
    M = embedding.shape[1]
    mind1 = np.argmin(embedding[1, :])
    mind2 = np.argmin(embedding[2, :])

    print(text) 

    # Make data.
    X = np.arange(0, M, 1.0)
    Y = np.arange(0, N, 1.0)
    X, Y = np.meshgrid(X, Y)

    Z = embedding
    if limits:
        minZ, maxZ = limits
    else:
        minZ, maxZ = -10, 10

    # Plot the surface.
    ax.clear()
    surf = ax.plot_surface(X, Y, Z, 
            #cmap=cm.Reds,
            cmap=cm.plasma,
            linewidth=0, 
            antialiased=False)

    #fig.colorbar(surf, shrink=0.5, aspect=5)

    ax.set(title=str(text) + " max-token-ix=" + str(mind1) + " " + str(mind2)) #, yscale='log')
    ax.set_zlim(minZ, maxZ)
    plt.pause(0.1)
    input()
    

diff = {
    'ad0'  : lambda x, y : np.abs(x - y),
    'sd2'  : lambda x, y : -(x - y)**2,
    'sd3'  : lambda x, y : -(x - y)**3,
    'sd4'  : lambda x, y : -((x - y)**4),
    'ld4'  : lambda x, y : -np.log((2.71 + np.abs(x - y))**4),
    'ad1'  : lambda x, y : np.clip(np.abs(x - y), 0, 1),
    'h16' : lambda x, y : np.clip(np.abs(x - y), 0, 8)[1:9, 0:16],
}

def plot_embedding(embs,  prompts=None):
    # transform to conventional shape
    embs = [e[0,:,:] for e in embs]
    ref = embs[0]

    # set z limits based on range of the diff function
    limits = np.min(np.array(embs).flatten()), np.max(np.array(embs).flatten())
    limits = { 
        'ld4' : (-10, -2),
        'sd4' : (-1000, 0), 
        'sd3' : (-300, 300), 
        'sd2' : (-100, 0), 
    }

    # choose a diff function
    fun = 'sd4'
    limits = limits[fun] 

    for n, emb in enumerate(embs):
        d = diff[fun](emb, ref)
        p = prompts[n] if prompts else None
        plot_embedding_surf(d, p, limits)
    #plt.close()

    print("done")

def plot_image(image, prompt=None, waittime_ms=30):
    title = prompt if prompt else 'image'
    cv.imshow(title, image)
    cv.waitKey(waittime_ms)

def plot_latent(latent, prompt=None):
    disp = cv.resize(latent[0,:,:,0:4], (640, 640), interpolation=cv.INTER_NEAREST) 
    return plot_image(disp, prompt, waittime_ms=100)

def pause():
    cv.waitKey(0)

if __name__ == "__main__":
    plot_embedding_surf()
    print("done")