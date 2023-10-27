import os
import random

import numpy as np
import pyrtools as pt
from PIL import Image
import matplotlib.pyplot as plt

import hyperparameters as hp

def probs1000_to_cat16(prediction, class_id_to_key=hp.CLASS_ID_TO_KEY, mappings=hp.CAT2SNET):
	"""Each 16-way category includes several 1000-way categories.
	This method computes mean prob. of 1000-way cats for each 16-way cat and picks the 16-way cat with max mean prob."""
	catprobs = dict([(k, []) for k in mappings.keys()])

	for key, vals in mappings.items():
		for val in vals:
			# find class id for that key
			if val not in class_id_to_key:
				continue
			class_id = class_id_to_key.index(val)

			# get probability for that class id
			catprob = prediction[class_id].item()

			# append to catprobs
			catprobs[key].append(catprob)

	# get key for max mean values
	catpred = max(catprobs, key=lambda k: np.mean(catprobs[k]))

	return catpred

def rmspower(im):
	"""Computes RMS power of an image."""
	return np.sqrt(np.mean(np.square(im)))

def contrast_normalize_np(im):
	return (im - np.min(im)) / (np.max(im) - np.min(im))

def solomon_filter(image, noise_sd, freq, contrast=hp.CONTRAST, imagenet_mean=hp.IMAGENET_MEAN, epsilon=hp.EPSILON, n_freqs=hp.N_FREQS):
	"""Applies noise sampled from a Gaussian with mean 0 and given 
	standard deviation to an image at a given spatial frequency."""

	def add_noise_at_sf(image, noise, freq):

		if noise_sd != 0:
			# create a laplacian pyramid of noise with floor(log_2(224)) levels.
			# Levels are octave spaced because their resolutions differ by 
			# a factor of 2
			pyr = pt.pyramids.LaplacianPyramid(noise)

			# reconstruct noise in pyramid for required sf band
			bandi = n_freqs - (freq+1)
			recon = pyr.recon_pyr(levels=bandi)
			first_recon = pyr.recon_pyr(levels=0)
			sf_noise = recon * rmspower(first_recon) / rmspower(recon) # equate power with first noise	
			noisyim = image + sf_noise

			return noisyim, sf_noise

		else:
			# if noise SD is 0, return image
			noisyim = image
			return noisyim, np.zeros_like(image)

	# normalize image to 0-1, decrease histogram width (contrast) and shift to imagenet mean
	image = (image / 255.0)
	image = (image - image.mean()) * contrast + imagenet_mean

	# generate Gaussian noise with mean 0 and given SD
	if noise_sd != 0:
		noise = np.random.randn(*image.shape) * noise_sd
	else:
		noise = None

	# add noise at required sf to image
	noisyim, sf_noise = add_noise_at_sf(image, noise, freq)

	# check for out of bounds values
	if noisyim.min() < 0 or noisyim.max() > 1:
		# find out of bound pixels and distort image appropriately
		for i in range(noisyim.shape[0]):
			for j in range(noisyim.shape[1]):
				OOB = noisyim[i, j] < 0 or noisyim[i, j] > 1

				if OOB:
					if noisyim[i, j] < 0:
						newp = epsilon - sf_noise[i, j]
						otherOOB = newp > 1
					elif noisyim[i, j] > 1:
						newp = 1 - epsilon - sf_noise[i, j]
						otherOOB = newp < 0

					if otherOOB:
						# means that shifting image pixel causes out of bounds in other direction
						# then only soln is to resample noise until no OOB
						# this will might cause noisy image to be outside desired SF band
						# but assuming only few noise pixels have to be replaced this way
						# that should be ok
						# TODO: check how many pixels we're doing this for
						newn = sf_noise[i, j]
						while newn + image[i,j] < 0 or newn + image[i,j] > 1:
							newn = np.random.randn() * noise_sd
						sf_noise[i, j] = newn

					else:
						# we can just set image pixel to shift value
						image[i, j] = newp

		# compute new noisy image
		noisyim = image + sf_noise 

		assert noisyim.max() <= 1 and noisyim.min() >= 0, "Image out of bounds"

	return noisyim.astype(np.float32)

def snetid2category(snet_id, mappings=hp.CAT2SNET):
	"""Converts a Synset ID to a category name."""
	for key, val in mappings.items():
		if snet_id in val:
			return key
	return None

def plot_image(image, vmin=0, vmax=255):
	plt.figure(figsize=(4,4))
	saveable = plt.imshow(image, cmap='gray', vmin = vmin, vmax = vmax)
	plt.axis('off')
	plt.show()

	return saveable

def plot_hist(image, bins=50):
	plt.figure(figsize=(4,2))
	plt.hist(image.flatten(), bins=bins)
	plt.xlabel('Pixel value')
	plt.ylabel('Frequency')
	plt.show()
