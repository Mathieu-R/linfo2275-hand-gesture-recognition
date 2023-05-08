import numpy as np
import numpy.typing as npt 

import re

from tqdm.notebook import tqdm
from typing import TextIO

def load_datasets(domain: str):
	dir_path = f"datasets/{domain}"

	start = ((int(domain.split("0")[1]) - 1) * 1000) + 1
	stop = start + 1000

	X = []
	Y = []

	# matrix (number_drawn, user_id)
	# at each cell we have a list of multiple try of a specific number drawing (number_drawn)
	# drawn by a specific user (user_id), each try contains a list of data points (x, y, z)
	hand_gesture_data_matrix = np.zeros((10, 10), dtype=object)
	for i in range(0, 10):
		for j in range(0, 10):
			hand_gesture_data_matrix[i, j] = []

	for filename in tqdm(range(start, stop)):
		file_path = f"{dir_path}/{filename}.txt"
		with open(file=file_path, mode="r") as f:
			# get the target, user_id 
			# and a list of positions vectors \in \R^3: <x, y, z>
			# that represents the drawing
			number_drawn, user_id, gesture_datas = load_gesture_data(file=f)

			X.append(gesture_datas)
			Y.append(number_drawn)

			hand_gesture_data_matrix[number_drawn - 1, user_id - 1].append(gesture_datas)

	return np.asarray(X), np.asarray(Y), hand_gesture_data_matrix

def load_gesture_data(file: TextIO):
	"""Load a hand gesture data file structured in the following way:
	-----
	Domain id = <domain-id>
	Class id = <class-id>
	User id = <user-id>

	"<x>,<y>,<z>,<t>"
	...
	-----

	Parameters
	----------
	file : TextIOWrapper
		Hand gesture data file

	Returns
	-------
	int
		Number drawn
	int 
		User id
	npt.NDArray, shape (n_timepoints, vec_dimension)
		Time series of hand gesture data points
	"""
	lines = file.readlines()
	
	number_drawn = lines[1].strip()
	match_number = re.search("=", number_drawn)
	number_drawn = int(match_number.string[match_number.end():].strip()) - 1

	user_id = lines[2].strip()
	match_user = re.search("=", user_id)
	user_id = int(match_user.string[match_user.end():].strip())

	gesture_datas = []

	for row in range(5, len(lines)):
		gesture_data = lines[row].split(",")
		# we only keep <x, y, z> coordinates
		gesture_data = np.asarray([float(data.strip()) for data in gesture_data[0:-1]])
		gesture_datas.append(gesture_data)
	
	return number_drawn, user_id, gesture_datas

# multivariate dtw distance computed as independent dtw
# compute the sum of distance of multiple univariate time series
# it can therefore benefits from optimization of univariate time series like LB_Keogh
# https://www.sciencedirect.com/science/article/pii/S0020025522013731#b0155 
def cumulative_distance(vec_a: npt.NDArray, vec_b: npt.NDArray):
	n = len(vec_a)
	m = len(vec_b)

	if n != m:
		raise ValueError("vec_a and vec_b are expected to have the same dimension.")
	
	distance = 0.0
	for i in range(0, n):
		# square of the euclidean distance
		distance += (vec_a[i] - vec_b[i]) ** 2

	return distance