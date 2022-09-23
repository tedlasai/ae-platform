import cv2
import numpy as np
import rawpy
from simple_camera_pipeline_aug.python.pipeline import run_pipeline_v2,get_metadata

import exposure_class
read_path = 'E:/s1f1/1P0A5067.dng'
raw_im = rawpy.imread(read_path)
raw_bayer = raw_im.raw_image.copy()
shape0 = raw_bayer.shape[0]
shape1 = raw_bayer.shape[1]
params = {
    'input_stage': 'raw',
    #             # options: 'raw', 'normal', 'white_balance', 'demosaic', 'xyz', 'srgb', 'gamma', 'tone'
    'output_stage': 'tone',
    #             # options: 'normal', 'white_balance', 'demosaic', 'xyz', 'srgb', 'gamma', 'tone'
    'demosaic_type': 'menon2007'
}
# raw_bayer_downscaled = np.empty((int(shape0 * (1 / 2)), int(shape1 * (1 / 2))))
# #
# # raw_bayer_downscaled_shape = (raw_bayer_downscaled.shape[1] // 2, raw_bayer_downscaled.shape[0] // 2)
# raw_bayer_downscaled[::2, ::2] = c1
# raw_bayer_downscaled[1::2, ::2] = c2
# raw_bayer_downscaled[::2, 1::2] = c3
# raw_bayer_downscaled[1::2, 1::2] = c4

c1 = raw_bayer[::4, ::4]
c2 = raw_bayer[1::4, ::4]
c3 = raw_bayer[::4, 1::4]
c4 = raw_bayer[1::4, 1::4]
raw_bayer_downscaled = np.empty((int(raw_bayer.shape[0] * 0.5), int(raw_bayer.shape[1] * 0.5)))
raw_bayer_downscaled[::2, ::2] = c1
raw_bayer_downscaled[1::2, ::2] = c2
raw_bayer_downscaled[::2, 1::2] = c3
raw_bayer_downscaled[1::2, 1::2] = c4


metadata = get_metadata(read_path)
#
output_image = run_pipeline_v2(raw_bayer_downscaled, params, metadata)
output_image = np.clip(output_image, 0, 1)

output_image_shape = (int(shape1 * (1 / 8)), int(shape0 * (1 / 8)))
output_image = cv2.resize(output_image, output_image_shape)

# output_image = output_image ** (1 / 2.2)
output_image = output_image * 255
output_image = output_image.astype(np.uint8)

output_image = cv2.cvtColor(output_image,cv2.COLOR_RGB2BGR)
cv2.imwrite('testing1.jpg',output_image)

