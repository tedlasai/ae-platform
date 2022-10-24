import numpy as np
import rawpy

im_path = "E:\data/black.dng"

raw_im = rawpy.imread(im_path)
raw_bayer = raw_im.raw_image.copy()
black = np.mean(raw_bayer)
print(black)









