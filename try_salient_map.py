import cv2
import argparse
import numpy as np
import pyimgsaliency as psal
#cv2.saliency

# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True,
# 	help="path to input image")
# args = vars(ap.parse_args())
# load the input image
#image = cv2.imread(args["images/out.jpg"])

# path to the image
# filename = 'pyimgsaliency/bird.jpg'
# filename = 'images/out.jpg'
# image = cv2.imread("images/out.jpg")



def one_img_rbd(image):
	#rbd = psal.get_saliency_rbd(image).astype('uint8')
	mbd = psal.get_saliency_mbd(image).astype('uint8')
	#cv2.imwrite("rbdshow"+'.jpg',rbd)
	#saliencyMap = rbd / 255
	saliencyMap = mbd / 255
	#saliencyMap = cv2.resize(saliencyMap,(168,112))
	#cv2.imwrite("rbdshow__" + '.jpg', (saliencyMap*255).astype(np.uint8))
	return saliencyMap

def one_img(image):
	saliency1 = cv2.saliency.StaticSaliencySpectralResidual_create()
	#
	(success, saliencyMap) = saliency1.computeSaliency(image)

	return saliencyMap




#image_out = np.zeros(gray.shape)


#map_try = one_img_rbd(gray[1,18])

images = np.load("Image_Arrays_from_dng/Scene25_show_dng_imgs.npy")
images = images[:,:,::4,::4]

gray = images
x,y,z,l,c = gray.shape

image_out = np.zeros((100,40,112,168))
for i in range(x):
	for j in range(y):
		# if (i*40 + j == 119) or (i*40 + j == 158) or (i*40 + j == 159):
		# 	image_out[i,j] = np.array(image_out[i,j-1])
		# 	continue
		try:
			map_ = one_img_rbd(gray[i, j])
			image_out[i, j] = cv2.resize(map_, (168, 112))
		except:
			image_out[i, j] = np.array(image_out[i, j - 1])

		print(i*40+j)


		#image_out[i, j] = np.resize(one_img(gray[i, j]),(112,168))
np.save('saliency_maps/Scene25_salient_maps_mbd', np.asarray(image_out))
#np.save('Scene22_salient_maps', np.asarray(image_out))
# initialize OpenCV's static saliency spectral residual detector and
# compute the saliency map
# saliency1 = cv2.saliency.StaticSaliencySpectralResidual_create()
#
# (success, saliencyMap) = saliency1.computeSaliency(image)
# saliencyMap = (saliencyMap * 255).astype("uint8")
# cv2.imshow("Image", image)
# cv2.imshow("Output1", saliencyMap)
#
# cv2.waitKey(0)

# initialize OpenCV's static fine grained saliency detector and
# compute the saliency map
# saliency = cv2.saliency.StaticSaliencyFineGrained_create()
# (success, saliencyMap) = saliency.computeSaliency(image)
# # if we would like a *binary* map that we could process for contours,
# # compute convex hull's, extract bounding boxes, etc., we can
# # additionally threshold the saliency map
# threshMap = cv2.threshold(saliencyMap.astype("uint8"), 0, 255,
# 	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
# # show the images
# #cv2.imshow("Image", image)
# cv2.imshow("Output2", saliencyMap)
# cv2.imshow("Thresh", threshMap)
# cv2.waitKey(0)