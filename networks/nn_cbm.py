"""
Loads images from CBM dataset and tests neural network of choice on them
"""

# built-in
import os
import random
import re
import json
import sys
import datetime
import time
import argparse

# third party
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import pyrtools as pt
import torch

# local
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from hyperparameters import *
from model import Model

def nn_cbm(network_name=NETWORK_NAME, pretrained=True, output_size=1000):
	model = Model(network_name, pretrained=pretrained, output_size=output_size)

	image_root = STIMULI_ROOT
	blocks = os.listdir(image_root)
	modeaccs = {}

	for block in blocks:
		if block == 'train' or block == '.DS_Store':
			continue
		block_index = int(block)

		# loop through images
		filenames = os.listdir(os.path.join(image_root, block))

		for file_index, filename in enumerate(filenames):
			abs_file_index = len(filenames) * block_index + file_index

			# load noisy grayscale image, stack to make 3-channel image, and preprocess using transforms.
			image = np.array(Image.open(os.path.join(image_root, block, filename)).convert('L'), dtype=np.uint8)
			image3 = np.stack([image, image, image], axis=2)
			target = filename.split('_')[-1].split('.')[0]

			catpred, correct = model.categorize16(image3, target)

			# get mode (noise sf combination) and append correctness value to modeaccs
			mode = filename.split('_')[2] + filename.split('_')[3]

			if mode not in modeaccs:
				modeaccs[mode] = [correct]
			else:
				modeaccs[mode].append(correct)

	# save to json file
	os.makedirs(NETWORK_DATA_ROOT, exist_ok=True)
	with open(f'{NETWORK_DATA_ROOT}/cbm_{network_name}_accuracies_{datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")}.json', 'w') as f:
		json.dump(modeaccs, f)

if __name__ == "__main__":
	nn_cbm('resnet50', pretrained=True, output_size=16)
