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
images = np.load("image_Arrays_from_dng/Scene17_show_dng_imgs.npy")
images = images[:,:,::4,::4]

#images_raw = np.load("image_Arrays_exposure_new/Scene22_ds_raw_imgs.npy")
#images = images[15][14]

#gray = np.mean(images, axis = 4)
gray = images
#images[images>245] = 0
#image = images ** (0.45)
#
# images = images.astype("uint8")
# image = images[13][20] ** (1/2.2)
# images1 = images1.astype("uint8")
# image1 = images1[13][20] ** (1/2.2)
# filename = image
# # image = image.astype("uint8")
# # get the saliency maps using the 3 implemented methods
#rbd = psal.get_saliency_rbd(images).astype('uint8')
# rbd1 = psal.get_saliency_rbd(image1).astype('uint8')


# ft = psal.get_saliency_ft(filename).astype('uint8')

# mbd = psal.get_saliency_mbd(filename).astype('uint8')

# often, it is desirable to have a binary saliency map
# binary_sal = psal.binarise_saliency_map(mbd,method='adaptive')

#img = cv2.imread(filename)
# backgroundforeground = np.empty(rbd.shape)
# #foreground
# backgroundforeground[rbd>100] = 255
# #bacgkround
# backgroundforeground[rbd<=100] = 0
# rbd[gray>245]= 0
#
# cv2.imshow('background_foreground',backgroundforeground)
# cv2.imshow('img',images)
# cv2.imshow('rbd',rbd)
# cv2.imshow('rbd',rbd1)
# # cv2.imshow('ft',ft)
# cv2.imshow('mbd',mbd)

#openCV cannot display numpy type 0, so convert to uint8 and scale
# cv2.imshow('binary',255 * binary_sal.astype('uint8'))


#cv2.waitKey(0)


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



x,y,z,l,c = gray.shape
#image_out = np.zeros(gray.shape)


#map_try = one_img_rbd(gray[1,18])

i = 99
j = 16
m = one_img_rbd(gray[i, j])
saliency = cv2.resize(m, (168, 112))
l =  np.where(saliency < 0.1, 0, 1)
cv2.imwrite("saliencymap" '.jpg', (l * 255).astype(np.uint8))
print("WRote")

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