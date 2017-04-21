import numpy as np
from skimage import io

def to_rgb(im):
    w, h = im.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = im
    ret[:, :, 1] = im
    ret[:, :, 2] = im
    return ret

def load_image(tmdb_id):
    image_name = str(tmdb_id) + '.jpg'
    try:
        im = io.imread('b:/posters224/' + image_name)
        if(len(im.shape) == 2):
            im = to_rgb(im)
        return (tmdb_id, (im/255.).astype(np.float32, copy = False))
    except:
        return (tmdb_id, None)