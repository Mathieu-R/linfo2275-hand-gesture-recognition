import numpy as np
import numpy.typing as npt 

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from scipy.interpolate import interp1d

from tqdm.notebook import tqdm

from math import sqrt, floor

def vector_quantize(X: npt.NDArray, ts_length: float = 0.8) -> npt.NDArray:
	"""Preproces the dataset by performing a vector quantization of the signals. The idea is to reduce the length of the time series by ... % in order to reduce the computation time of the algorithms.

	Parameters
	----------
	X : npt.NDArray, shape (n_samples, n_timepoints, vec_dimensions)
		Dataset of the time series data points. Time series can vary in length (n_timepoints vary among samples).
	ts_length: float, in the range ]0,1]
		The percentage of time series to keep.

	Returns
	-------
	npt.NDArray, shape (n_samples, n_clusters, vec_dimensions)
		Dataset of the reduced length time series datapoints.
	"""
	if ts_length <= 0 or ts_length > 1:
		raise ValueError("ts_length should be a number in the range ]0,1]")

	X_compressed = []
	for sketch_data in tqdm(X):
		# vector quantization of the signal
		n_clusters = floor(len(sketch_data) * ts_length)
		kmeans = KMeans(n_clusters=n_clusters, n_init="auto")
		kmeans.fit(sketch_data)
		X_compressed.append(kmeans.cluster_centers_)
	
	return np.array(X_compressed)

def resample_ts(ts: npt.NDArray, N: int = 64) -> npt.NDArray:
	"""Resample a time series of m data points into a time series of n_points data points by interpolation.

	Parameters
	----------
	ts : npt.NDArray, shape (n_timepoints, vec_dimension)
		the original time series
	N : int, optional
		the number of points to resample, by default 64

	Returns
	-------
	npt.NDArray, shape (N_timepoints, vec_dimension)
		the resampled time series
	"""
	ts = np.array(ts)
	
	n, p = ts.shape
	resampled_ts = np.zeros((N, p))

	t_old = np.arange(0, n, 1)
	t_new = np.linspace(0, n - 1, N)

	for dim in range(0, p):
		x_old = ts[:, dim]
		f = interp1d(t_old, ts[:, dim], axis=0)
		resampled_ts[:, dim] = f(t_new)
	
	return resampled_ts

def resample_dataset(X: npt.NDArray, N: int = 64):
	n_samples = len(X)
	resampled_X = np.zeros((n_samples, N, 3))

	for i in range(0, n_samples):
		# resample the time series to N = 64 data points
		# http://faculty.washington.edu/wobbrock/pubs/uist-07.01.pdf 
		ts = X[i]
		resampled_ts = resample_ts(ts, N = N)
		resampled_X[i] = resampled_ts
	
	#plot_sketch(resampled_X)
	return resampled_X

def standardize_dataset(X: npt.NDArray):
	n_samples = len(X)
	standardized_X = []

	scaler = StandardScaler()

	for i in range(0, n_samples):
		ts = X[i]
		scaled_ts = scaler.fit_transform(ts)
		standardized_X[i] = scaled_ts
	
	return standardized_X