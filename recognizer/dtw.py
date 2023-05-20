import numpy as np 
import numpy.typing as npt

from math import sqrt
from collections import defaultdict

# https://towardsdatascience.com/dynamic-time-warping-3933f25fcdd
def full_dtw(ts_a: npt.NDArray, ts_b: npt.NDArray, dist_function, radius: int = 3) -> float:
	"""Compute the DTW distance for two time series

	Parameters
	----------
	ts_a : npt.NDArray, shape (n_timepoints, vec_dimension)
		Time series of hand gesture data points
	ts_b : npt.NDArray, shape (m_timepoints, vec_dimension)
		Time series of hand gesture data points
	window : int, optional
		Window size, by default 3

	Returns
	-------
	float
		_description_
	"""

	n = len(ts_a)
	m = len(ts_b)

	DTW_matrix = np.full((n+1, m+1), fill_value=np.inf)

	# initial conditions
	DTW_matrix[0, 0] = 0.0

	for i in range(0, n):
		DTW_matrix[i, 0] = np.inf
	
	for j in range(0, m):
		DTW_matrix[0, j] = np.inf

	# window optimization
	window = max(radius, abs(n - m))

	for i in range(0, n+1):
		for j in range(max(0, i - radius), min(m, i + radius) + 1):
			DTW_matrix[i, j] = 0.0

	# fill the matrix
	for i in range(1, n+1):
		for j in range(max(1, i - radius), min(m, i + radius) + 1):
			# manhattan distance
			cost = dist_function(ts_a[i-1], ts_b[j-i])
			#cost = distance.sqeuclidean(ts_a[i-1], ts_b[j-1])
			optimal_warping_path = min(
				DTW_matrix[i-1, j], # insertion
				DTW_matrix[i, j-1], # deletion
				DTW_matrix[i-1, j-1] # match
			)

			DTW_matrix[i, j] = cost + optimal_warping_path
	
	return sqrt(DTW_matrix[n, m])

def coarsening(ts):
	"""Reduce the size of the time series

	Parameters
	----------
	ts : Array-like, shape (n_timepoints, vec_dimension)
		Time series of hand gesture data points
	"""
	def averaging(ts, i):
		return (ts[i] + ts[i + 1]) / 2

	n = len(ts)
	averaged_ts = [averaging(ts, i) for i in range(0, n - (n % 2), 2)]
	return np.asarray(averaged_ts)

def expand_searching_window(warping_path, n, m, radius):
	"""Expand the searching window around the warping path by a radius factor.

	Parameters
	----------
	warping_path : Array-like, shape ()
		Warping path
	n : _type_
		_description_
	m : _type_
		_description_
	radius : int
		Radius by which we expand the searching window

	Returns
	-------
	Array-like, shape
		The extended searching window
	"""
	warping_path_set = set(warping_path)

	for (i, j) in warping_path:
		for move_i in range(-radius, radius + 1):
			for move_j in range(-radius, radius + 1):
				neighbor_i = i + move_i
				neighbor_j = j + move_j
				if (neighbor_i >= 0 and neighbor_j >= 0):
					warping_path_set.add((neighbor_i, neighbor_j))
	
	window = set()
	
	for (i, j) in warping_path_set:
		for (a, b) in ((i * 2, j * 2), (i * 2, (j * 2) + 1), ((i * 2) + 1, j * 2), ((i * 2) + 1, (j * 2) + 1)):
			window.add((a, b))

	#print(window)

	searching_window = []
	start_j = 0

	for i in range(0, n):
		new_start_j = None
		for j in range(start_j, m):
			if (i, j) in window:
				searching_window.append((i, j))
				if new_start_j is None:
					new_start_j = j
			elif new_start_j is not None:
				break
		
		start_j = new_start_j

	#print(searching_window)
	return searching_window

def constrained_dtw(ts_a, ts_b, dist_function, window):
	"""Compute the dtw distance between two time series on a constrained window.

	Parameters
	----------
	ts_a : npt.NDArray, shape (n_timepoints, vec_dimension)
		Time series of hand gesture data points
	ts_b : npt.NDArray, shape (m_timepoints, vec_dimension)
		Time series of hand gesture data points
	dist_function : _type_
		A distance function to compute the distance between two data points
	window : Array-like, shape ()
		The constrained window

	Returns
	-------
	float, list[(int, int)]
		The computed distance between the two time series as well as the shortest path
	"""
	n = len(ts_a)
	m = len(ts_b)

	#print(window)

	if window is None:
		window = [(i, j) for i in range(0, n) for j in range(0, m)]

	# shift (i, j) cell to (i + 1, j + 1)
	window = [(i + 1, j + 1) for i, j in window]

	cost_matrix = defaultdict(lambda: (float("inf"),))

	#print(window)

	# initial conditions
	cost_matrix[0, 0] = (0, (0, 0))

	# search for the optimal warping path
	for (i, j) in window:
		#print(i, j)
		#print(ts_a[i-1], ts_b[j-1])
		cost = dist_function(ts_a[i-1], ts_b[j-1])
		accumulated_cost_and_cell = min(
			(cost + cost_matrix[i-1, j][0], (i-1, j)), # insertion
			(cost + cost_matrix[i, j-1][0], (i, j-1)), # deletion
			(cost + cost_matrix[i-1, j-1][0], (i-1, j-1)), # match
			key = lambda x: x[0]
		)

		cost_matrix[i, j] = accumulated_cost_and_cell

	# reconstruct the path
	path = []
	i, j = n, m

	while not (i == 0 and j == 0):
		path.append((i - 1, j - 1))
		i, j = cost_matrix[i, j][1]
	path.reverse()

	return cost_matrix[n, m][0], path

def fast_dtw(ts_a, ts_b, dist_function, radius = 1):
	"""Run the fast dtw algorithm

	Parameters
	----------
	ts_a : npt.NDArray, shape (n_timepoints, vec_dimension)
		Time series of hand gesture data points
	ts_b : npt.NDArray, shape (m_timepoints, vec_dimension)
		Time series of hand gesture data points
	dist_function : _type_
		A distance function to compute the distance between two data points
	radius : int, optional
		The factor by which we should expand the searching window, by default 1

	Returns
	-------
	float, list[(int, int)]
		The computed distance between the two time series as well as the shortest path
	"""
	n = len(ts_a)
	m = len(ts_b)

	# min size of the coarsest resolution
	min_ts_size = radius + 2

	# base case: for really small time series,
	# run the full dtw algorithm
	if n <= min_ts_size or m <= min_ts_size:
		return constrained_dtw(ts_a, ts_b, dist_function, window=None)

	# coarsening
	ts_a_shrunk = coarsening(ts_a)
	ts_b_shrunk = coarsening(ts_b)

	# recursively compute a lower resolution path
	dist, low_res_path = fast_dtw(ts_a_shrunk, ts_b_shrunk, dist_function, radius)

	# compute a searching window
	searching_window = expand_searching_window(low_res_path, n, m, radius)

	return constrained_dtw(ts_a, ts_b, dist_function, window=searching_window)