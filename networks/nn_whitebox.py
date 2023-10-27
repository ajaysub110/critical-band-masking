import sys
import json
import os

import torch
import torchvision
import torchvision.models as tvmodels
from torchvision import transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from art.attacks.evasion import ProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier
from art.utils import load_dataset

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from hyperparameters import *
from modelvshuman import models
from model import Model

def load_images():
	# Evaluate the white box accuracy of the model on the ImageNet validation dataset
	snets = os.listdir(IMAGENET_ROOT)

	class_id_to_key = dict(zip(CLASS_ID_TO_KEY, list(range(len(CLASS_ID_TO_KEY)))))

	# prepare dataset in required format
	images = []
	labels = []
	for snetid in snets[:]:
		true_label = class_id_to_key[snetid]
		for filename in os.listdir(os.path.join(IMAGENET_ROOT, snetid))[:1]:
			image = Image.open(os.path.join(IMAGENET_ROOT, snetid, filename)).convert('RGB')

			# Resize the image to 256x256
			image = image.resize((256, 256))

			# Crop the center 224x224 region of the image
			left = (256 - 224) / 2
			top = (256 - 224) / 2
			right = (256 + 224) / 2
			bottom = (256 + 224) / 2
			image = np.array(image.crop((left, top, right, bottom)), dtype=np.uint8)

			images.append(image)
			labels.append(true_label)


	images = np.array(images)
	labels = np.array(labels)

	# convert to required dimension
	images = np.transpose(images, (0, 3, 1, 2)).astype(np.float32)

	# normalize
	images = images/255
	mean = np.array([0.485, 0.456, 0.406]).reshape(1,3,1,1)
	std = np.array([0.229, 0.224, 0.225]).reshape(1,3,1,1)
	images = np.float32((images - mean) / std)

	return images, labels

# Load the pre-trained model
def nn_whitebox(images, labels, network_name = NETWORK_NAME):
	model = Model(network_name)

	# Create a PyTorchClassifier instance from the pre-trained model
	classifier = PyTorchClassifier(model=model.net, loss=torch.nn.CrossEntropyLoss(), input_shape=(3, 224, 224), nb_classes=1000, clip_values=(0, 1)) # THIS RUNS OK

	predictions = classifier.predict(images)
	benign_accuracy = np.sum(np.argmax(predictions, axis=1) == labels) / len(labels)
	print(network_name)
	print("Accuracy on benign test examples: {}%".format(benign_accuracy * 100))

	# Define the attack
	attack = ProjectedGradientDescent(estimator=classifier, norm=2, eps=eps, max_iter=32)
	images_adv = attack.generate(x=images)

	predictions = classifier.predict(images_adv)
	whitebox_accuracy = np.sum(np.argmax(predictions, axis=1) == labels) / len(labels)
	print("Accuracy on adversarial test examples: {}%".format(whitebox_accuracy * 100))

	return benign_accuracy, whitebox_accuracy

if __name__ == "__main__":
	images, labels = load_images()
	# nn_whitebox(images, labels)

	network_list = ['resnet50']
	waccs = {}
	eps = 1.0
	baccs = {}
	for network_name in network_list:
		try:
			print(network_name)
			baccs[network_name], waccs[network_name] = nn_whitebox(images, labels, network_name)
			print('done')
			with open(os.path.join(NETWORK_DATA_ROOT, f'baccs_eps{str(eps)}.json'), 'w') as f:
				json.dump(baccs, f)
			with open(os.path.join(NETWORK_DATA_ROOT, f'waccs_eps{str(eps)}.json'), 'w') as f:
				json.dump(waccs, f)
		except Exception as e:
			print('failed')
			print(e)
		print()
		continue