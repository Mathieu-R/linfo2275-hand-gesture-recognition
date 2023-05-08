import numpy as np
import numpy.typing as npt 

from sklearn.metrics import accuracy_score

from .constants import USER_IDS

def split_dataset(X, Y, slices):
	X_train = np.delete(X, slices, axis=0)
	y_train = np.delete(Y, slices, axis=0)

	X_test = X[slices]
	y_test = Y[slices]

	return X_train, X_test, y_train, y_test

def user_independent_cross_validation(estimator, X, y):
	y_preds = []
	y_tests = []
	accuracies = []

	for user_id in USER_IDS:
		slices = range(user_id * 100, 100 + (user_id * 100))
		X_train, X_test, y_train, y_test = split_dataset(X, y, slices)

		estimator.fit(X_train, y_train)
		y_pred = estimator.predict(X_test)

		accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
		accuracies.append(accuracy)

		y_tests.append(y_test)
		y_preds.append(y_pred)

	accuracy_mean = np.mean(accuracies)
	accuracy_std = np.std(accuracies)
	
	return accuracy_mean, accuracy_std, y_preds, y_tests

def user_dependent_cross_validation(estimator, X, y):
	n_samples = len(X)

	y_preds = []
	y_tests = []
	accuracies = []

	for user_id in USER_IDS:
		slices = np.arange(user_id, n_samples + user_id, 10)
		X_train, X_test, y_train, y_test = split_dataset(X, y, slices)

		estimator.fit(X_train, y_train)
		y_pred = estimator.predict(X_test)

		accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
		accuracies.append(accuracy)

		y_tests.append(y_test)
		y_preds.append(y_pred)

	accuracy_mean = np.mean(accuracies)
	accuracy_std = np.std(accuracies)
	
	return accuracy_mean, accuracy_std, y_preds, y_tests

def test_algo(estimator, X, y):
	user_id = 0
	slices = range(user_id * 100, 100 + (user_id * 100))
	X_train, X_test, y_train, y_test = split_dataset(X, y, slices)

	estimator.fit(X_train, y_train)
	y_pred = estimator.predict(X_test)

	accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

	return accuracy, y_pred, y_test