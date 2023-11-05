import re

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# -------------------------
# HUMAN DATA ANALYSIS
# -------------------------
def human_training_accuracy(blocked_datas, dfs, vb=True):
	"""
	Computes accuracy of each observer on training block
	"""
	cat_accs = []
	for blocked_data, df in zip(blocked_datas, dfs):
		accuracy_training = blocked_data['training']['correct'].sum() / len(blocked_data['training']['correct'])
		cat_accs.append(accuracy_training * 100)
		if vb:
			print("{}: Accuracy on training block: {} %".format(df['url'][0], accuracy_training * 100))

	return cat_accs

def human_accuracy_matrix(blocked_datas):
	"""
	Computes averaged-human accuracy matrix and optionally plots heatmap
	"""
	final_mat = np.zeros((5, 7))
	all_mats = []
	# iterate through observer data
	for blocked_data in blocked_datas:
		mode_accuracies = {}
		for block in list(map(str, range(5))):
			block = blocked_data[block]

			for _, trial in block.iterrows():
				if trial['mode'] not in mode_accuracies:
					mode_accuracies[trial['mode']] = [trial['correct']]
				else:
					mode_accuracies[trial['mode']].append(trial['correct'])

		# get matrix for the observer
		noise_list = list(map(str, [0, 0.02, 0.04, 0.08, 0.16]))
		freq_list = list(map(str, range(7)))
		accuracy_matrix = np.zeros((len(noise_list), len(freq_list)))

		for mode in mode_accuracies:
			mode_split = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", mode)
			noise, freq = mode_split[0], mode_split[1]
			accuracy_matrix[noise_list.index(noise), freq_list.index(freq)] = np.mean(mode_accuracies[mode]) * 100

		# add to final mat, which will eventually be normalized to get average accuracy
		final_mat += accuracy_matrix
		all_mats.append(accuracy_matrix)

	final_mat /= len(blocked_datas)

	return final_mat, all_mats

# -------------------------
# CHANNEL METRICS
# -------------------------
def find_thresholds(accuracy_matrix, threshold_acc=50):
	"""
	Computes threshold indices for each spatial frequency given
	a matrix of accuracies for each noise condition
	"""
	# compute threshold indices
	thresholds = []
	for i in range(7):
		for j in range(5):
			jr = len(accuracy_matrix) - 1 - j
			if accuracy_matrix[jr, i] > threshold_acc:
				break
		thresholds.append(j)

	thresholds = np.array(thresholds)

	return thresholds

def fit_gaussian(thresholds, vb=True):
	'''
	Fits a gauss function to array of thresholds
	'''
	def gauss(x, *p):
		A, mu, sigma = p
		return A * np.exp(-(x-mu)**2/(2.*sigma**2))
	
	p0 = [1., 0., 1.]
	(A, mu, sigma), var_matrix = curve_fit(gauss, np.arange(len(thresholds)), thresholds, p0=p0, bounds=((0,-np.inf,-np.inf),(4, np.inf, np.inf)))
	if vb:
		print("fit_gaussian coeffs: A = {:.2f}, mu = {:.2f}, sigma = {:.2f}\n".format(
			A, mu, sigma
		))

	hist_fit = gauss(np.linspace(0, len(thresholds)-1, 60), *(A, mu, sigma))

	return hist_fit, (A, mu, sigma)

def channel_props(A, mu, sigma, vb=True):
	"""
	Calculates channel properties, given fit gaussian parameters
	"""
	bw = 2 * np.sqrt(np.log(4)) * sigma
	cf = 1.75 * 2 ** mu
	pns = 2**(A-4)

	if vb:
		print("channel_props:\nBW: {:.2f} octaves,\nCF: {:.2f} cycles/image,\nPNS: {:.2f} per unit SD\n".format(
			bw, cf, pns
		))

	return bw, cf, pns
