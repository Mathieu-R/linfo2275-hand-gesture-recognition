import numpy as np 
import numpy.typing as npt

from math import sqrt

# CAVEAT: ONLY WORKS FOR MONODIMENSIONAL TIME SERIES...
def lb_keogh(ts_query: npt.NDArray, ts_candidate: npt.NDArray, window: int) -> float:
	"""Implement LB Keogh algorithm which is a fast lowerbounding method for constrained DTW distance

	Parameters
	----------
	ts_query : npt.NDArray, shape (n_timepoints,)
		Query time series to compare with the envelope of the candidate time series.
	ts_candidate : npt.NDArray, shape (m_timepoints,)
		Candidate time series, used to compute the envelope.
	window : int
		Windows size to use for the envelope generation

	Returns
	-------
	float
		Distance between the query time series and the envelope of the candidate time series
	"""
	n = min(len(ts_query), len(ts_candidate))
	lb_tot = 0

	for i in range(0, n):
		start = (i - window if i - window >= 0 else 0)
		stop = (i + window if i + window <= n - 1 else n - 1)

		lower_bound = min(ts_candidate[start:stop + 1])
		upper_bound = min(ts_candidate[start:stop + 1])

		if ts_query[i] > upper_bound[j]:
			lb_tot += (ts_query[i] - upper_bound) ** 2
		elif ts_query[i] < lower_bound[j]:
			lb_tot += (lower_bound - ts_query[i]) ** 2
			
	return sqrt(lb_tot)