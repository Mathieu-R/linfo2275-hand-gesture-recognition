import numpy as np 
import numpy.typing as npt 

from scipy.stats import mode
from tqdm.notebook import tqdm

from enum import Enum

from recognizer.dtw import full_dtw, fast_dtw
from recognizer.misc import cumulative_distance


class Metrics(Enum):
	DTW = "dtw"
	FAST_DTW = "fast_dtw"

# https://nbviewer.org/github/markdregan/K-Nearest-Neighbors-with-Dynamic-Time-Warping/blob/master/K_Nearest_Neighbor_Dynamic_Time_Warping.ipynb

# http://alexminnaar.com/2014/04/16/Time-Series-Classification-and-Clustering-with-Python.html

# https://www.sciencedirect.com/science/article/pii/S0031320317304387#bib0037

class KNN:
	def __init__(self, n_neighbors = 1, radius = 1, metric = "fast_dtw"):
		self.n_neigbors = n_neighbors
		self.radius = radius
		self.metric = fast_dtw

		self.X = None 
		self.y = None

		self.set_metric(metric)

	def set_metric(self, metric: str):
		# only dtw metric supported in this algorithm
		if (metric == Metrics.DTW.value):
			self.metric = full_dtw
		elif (metric == Metrics.FAST_DTW.value):
			self.metric = fast_dtw
		else:
			raise ValueError("Metric unsupported by the KNN algorithm...")
	
	def fit(self, X: npt.NDArray, y: npt.NDArray):
		"""Fit the model using X as training data and y as target

		Parameters
		----------
		X : npt.NDArray, shape (n_samples, n_timepoints, vec_dimension)
			Training data
		y : npt.NDArray, shape (n_samples,)
			Target data
		"""
		self.X = X
		self.y = y

	def distance_matrix(self, X_train: npt.NDArray, X_test: npt.NDArray):
		n = len(X_test)
		m = len(X_train)
		distance_matrix = np.zeros((n, m))

		for i in tqdm(range(0, n)):
			for j in range(0, m):
				# compare each time series in the testing set 
				# with every other one in the training set
				dist, warping_path = self.metric(X_test[i], X_train[j], dist_function=cumulative_distance, radius=self.radius)

				# keep track of the dtw distance between time series i and j
				distance_matrix[i, j] = dist
		
		return distance_matrix

	def predict(self, X: npt.NDArray) -> npt.NDArray:
		"""Predict the target values (here: class label) for the given testing dataset

		Parameters
		----------
		X : npt.NDArray, shape (n_samples, n_timepoints, vec_dimension)
			Testing data

		Returns
		-------
		npt.NDArray, shape (n_samples,)
			Array of predicted target values
		"""
		distance_matrix = self.distance_matrix(self.X, X)

		knn_idx = distance_matrix.argsort()[:, :self.n_neigbors]
		knn_labels = self.y[knn_idx]

		prediction = mode(knn_labels, keepdims=True, axis=1)[0]

		return prediction.ravel()