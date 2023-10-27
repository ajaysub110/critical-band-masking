"""
Loads grayscale images from GraySplit dataset and generates stimuli for noise masking experiment
"""

# built in packages
import os
import random
import sys

# external packages
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from hyperparameters import *
from methods import solomon_filter

# set random seeds
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# global vars
SAVE = True

if not os.path.exists(STIMULI_ROOT) and SAVE:
	os.makedirs(STIMULI_ROOT, exist_ok=True)

# function defs
def stack_and_transform(im):
	"""
	stack 1-channel image to form 3-channel image and then pass to torchvision transforms
	"""
	im3 = np.stack([im, im, im], axis=2)
	transformed_im = IMAGE_TRANSFORMS(im3).unsqueeze(0)

	return transformed_im[0]

def contrast_normalize_torch(im):
	# normalize contrast in torch tensor of image
	return (im - torch.min(im)) / (torch.max(im) - torch.min(im))

def create_cbm_dataset():
	# here, ROOT expects location of GraySplit (for now, download GraySplit from drive link)
	for subdir in os.listdir(GRAY_ROOT):
		if subdir != '.DS_Store':
			filenames = os.listdir(os.path.join(GRAY_ROOT, subdir))
		else:
			continue
		
		if SAVE:
			os.makedirs(os.path.join(STIMULI_ROOT, subdir))

		for filename in filenames:
			# select spatial frequency
			f_i = random.choice(list(range(N_FREQS)))

			# apply noise of selected intensity and frequency to it
			image = np.array(Image.open(os.path.join(GRAY_ROOT, subdir, filename)).convert('L'), dtype=np.uint8)

			fnsplit = filename.split('_')

			# Select noise/contrast, and filter image
			noise_sd = random.choice(NOISE_SDS)
			maskedim = solomon_filter(image, noise_sd, freq=f_i)
			saveable = stack_and_transform(maskedim)
			fnsplit[1] = f"cbm_noise{noise_sd}_freq{f_i}"

			savepath = os.path.join(STIMULI_ROOT, subdir, '_'.join(fnsplit))

			if SAVE:
				save_image(saveable, savepath)

if __name__ == '__main__':
	create_cbm_dataset()