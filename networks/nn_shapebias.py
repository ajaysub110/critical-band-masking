import os
import sys
import json

import numpy as np
from PIL import Image

from modelvshuman.datasets import cue_conflict
from modelvshuman.models.pytorch import model_zoo
from modelvshuman import models
import torchvision

sys.path.append('../')
from hyperparameters import *
from model import Model
from methods import snetid2category, catpred_16way, predict_category

dataset = cue_conflict(batch_size=16, num_workers=4)

def compute_shape_bias(network_name=NETWORK_NAME):
	model = Model(network_name)

	categories = os.listdir(dataset.path)

	sb_scores = []
	for shapecat in categories:
		imagefs = os.listdir(os.path.join(dataset.path, shapecat))
		cat_corrects = []

		for imagef in imagefs:
			# don't proceed if shape and texture are from same cat
			texcat = imagef.split('-')[1].split('.')[0][:-1]
			if texcat == shapecat:
				continue

			# load image and pass it through model
			image3 = np.array(Image.open(os.path.join(dataset.path, shapecat, imagef)), dtype=np.uint8)

			catpred, correct = model.categorize16(image3, shapecat)

			cat_corrects.append(correct) 

		sb_scores.append(np.mean(cat_corrects))

	return sb_scores

if __name__ == "__main__":
	compute_shape_bias('resnet50')
