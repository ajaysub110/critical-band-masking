# built-in
import sys
import os

# third-party
import torchvision.models as tvmodels
from torchvision import transforms as T
import torch.nn as nn
from modelvshuman.models.pytorch import model_zoo

# local
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from hyperparameters import *
from methods import (
	probs1000_to_cat16
)

class Model:
	def __init__(self, network_name, pretrained=True, output_size=1000):
		self.network_name = network_name
		self.pretrained = pretrained
		self.output_size = output_size
		self.net = self.load(network_name, pretrained=pretrained)

		self._categories1000 = None
		self._transforms = None

	def load(self, network_name, pretrained=True):
		"""
		load network with pretrained weights
		"""
		def change_output_size(model, new_output_size):
			num_features = model.fc.in_features
			model.fc = nn.Linear(num_features, new_output_size)
			return model

		# torchvision models
		if network_name in tvmodels.list_models():
			if pretrained == True:
				print('loading from torchvision')
				network = getattr(tvmodels, network_name)(weights='DEFAULT')
			else:
				print('loading from {}'.format(pretrained))
				# network = getattr(tvmodels, network_name)()
				network = tvmodels.__dict__[network_name]()
				if self.output_size == 16:
					network = change_output_size(network, new_output_size=16)
				network = torch.nn.DataParallel(network)
				checkpoint = torch.load(pretrained)
				network.load_state_dict(checkpoint['state_dict'])
		
		# geirhos modelvshuman models
		elif network_name in dir(model_zoo):
			print('loading from modelvshuman')
			network = getattr(model_zoo, network_name)(model_name=network_name).model

		# outliers
		elif network_name == 'voneresnet50':
			from vonenet import get_model
			network = get_model(model_arch='resnet50', pretrained=True)

		else:
			raise ValueError(f"Invalid network name {network_name}")
		network.eval()

		return network.to(DEVICE)

	def categorize16(self, image3, target):
		"""
		transform image and predict 16-way category of image.
		return category prediction and correctness
		"""

		# transform image
		transformed_image3 = self.transforms(image3).unsqueeze(0)
		transformed_image3 = transformed_image3.to(DEVICE)

		# feed image to network and obtain predictions. eval if correct.
		probs = self.net(transformed_image3).squeeze(0).softmax(0)

		# get 16-way category prediction
		if self.output_size == 1000:
			catpred = probs1000_to_cat16(probs)
		elif self.output_size == 16:
			catpred = CATEGORIES16[np.argmax(probs.detach().cpu())]

		# check if correct
		correct = int(catpred == target)

		return catpred, correct

	def categorize16_batch(self, image_batch, target):
		"""
		predict 16-way category for each image in transformed batch.
		return category prediction and correctness
		"""

		# feed image to network and obtain predictions. eval if correct.
		predictions = self.net(image_batch).squeeze(0).softmax(1)

		# get 16-way category prediction
		catpreds, corrects = [], []
		for p in range(len(predictions)):
			catpred = probs1000_to_cat16(predictions[p])
			catpreds.append(catpred)
			# check if correct
			corrects.append(int(catpred == target))

		return catpreds, corrects

	@property
	def categories1000(self):
		"""return imagenet1000 category labels"""
		if self._categories1000 is None:
			self._categories1000 = tvmodels.AlexNet_Weights.DEFAULT.meta['categories']
		return self._categories1000

	@property
	def transforms(self):
		"""return network transforms to input"""
		if self._transforms is None:
			self._transforms = T.Compose([
				T.ToTensor(),
				T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
			])
		return self._transforms

if __name__ == "__main__":
	model = Model('alexnet')
	print(model.categories1000)