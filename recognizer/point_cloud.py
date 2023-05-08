import numpy as np
import numpy.typing as npt

from tqdm.notebook import tqdm

from sklearn.decomposition import PCA

from dollarpy import Recognizer, Template, Point

class PointCloudRecognizer:
	def __init__(self, N = 64):
		self.N = N
		self.recognizer = None

	def fit(self, X: npt.NDArray, y: npt.NDArray):
		"""_summary_

		Parameters
		----------
		X : npt.NDArray, shape (n_samples, n_timepoints, vec_dimension)
			Training data
		y : npt.NDArray, shape (n_samples,)
			Training target values

		Returns
		-------
		_type_
			_description_
		"""
		self.X = X
		self.y = y

		training_data = []

		# for each vector of data points
		for (gesture, label) in zip(X, y):
			# we have 3D gesture data points 
			# $p-recognizer only accept 2D data points so we need to reduce to 2 dimensions
			pca = PCA(n_components=2)
			gesture_2d = pca.fit_transform(gesture)

			# convert each data points to point object
			points = []
			for (x1, x2) in gesture_2d:
				point = Point(x1, x2)
				points.append(point)
			
			training_data.append(Template(label, points))
		
		self.recognizer = Recognizer(training_data)
		return self.recognizer
			
	def predict(self, X: npt.NDArray):
		"""_summary_

		Parameters
		----------
		X : npt.NDArray, shape (n_samples, n_timepoints, vec_dimension)
			Testing data

		Returns
		-------
		_type_
			_description_
		"""
		if not self.recognizer:
			raise ValueError("You must fit the model first.") 

		results = []

		for gesture in tqdm(X):
			pca = PCA(n_components=2)
			gesture_2d = pca.fit_transform(gesture)

			points = []
			for (x1, x2) in gesture_2d:
				point = Point(x1, x2)
				points.append(point)

			result = self.recognizer.recognize(points, n=64)
			results.append(result[0])

		return results