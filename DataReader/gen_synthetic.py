import numpy as np
np.random.seed(1)

def rotate_images(imgs):
    ret = []
    for img in imgs:
        tmp = img
        for x in xrange(3):
            tmp = np.rot90(tmp)
            ret.append(tmp)
    return ret

def translate_images(imgs):
    ret = []
    for img in imgs:
        zs = [np.zeros(img.shape) for x in xrange(4)]
        zs[0][1:,1:] = img[:-1,:-1]
        zs[1][1:,:-1] = img[:-1,1:]
        zs[2][:-1,1:] = img[1:,:-1]
        zs[3][:-1,:-1] = img[1:,1:]
        ret.extend(zs)
    return ret

def flip_images(imgs):
    ret = []
    for img in imgs:
        ret.append(np.flipud(img))
    return ret

# Since this is non-deterministic, do this last
def add_noise(imgs, mean, sd):
    ret = []
    for img in imgs:
        ret.append(img+np.random.randn(img.shape[0],img.shape[1])*sd+mean)
    return ret

# 80x data multiplier
# time cost is roughly 1.5min / epoch
def transform_images(imgs):
    ret = []
    for img in imgs:
        tmp = []
        tmp.append(img)
        tmp.extend(flip_images(tmp))
        tmp.extend(rotate_images(tmp))
        tmp.extend(translate_images(tmp))
        tmp.extend(add_noise(tmp,0,0.01))
        ret.extend(tmp)
    return ret