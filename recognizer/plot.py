import numpy as np
import numpy.typing as npt

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix

def plot_sketch(X: npt.NDArray):
	data = np.array(X[0])

	fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
	ax.scatter(data[:,0], data[:,1], data[:,2])

def plot_confusion_matrix(y_true, y_pred, i, ax = None):
	if ax is None:
		fig, ax = plt.subplots()
	else:
		fig = ax.figure

	conf_matrix = np.array(confusion_matrix(y_true=y_true, y_pred=y_pred))
	sns.heatmap(conf_matrix, annot=True, cbar=False, ax=ax)
	ax.set_title(f"split {i}")

def plot_multiple_confusion_matrix(y_trues, y_preds):
	if len(y_trues) != len(y_preds):
		raise ValueError("y_trues and y_preds should be of same length.")

	n = len(y_trues)
	fix, axes = plt.subplots(nrows = 2, ncols = 5, figsize=(20, 7))

	for i in range(0, n):
		plot_confusion_matrix(y_true=y_trues[i], y_pred=y_preds[i], i=i+1, ax=axes.flatten()[i])