# linfo2275-hand-gesture-recognition

In order to run the notebook `hand-gesture-recognition.ipynb` you need well known datascience libraries like `numpy`, `matplotlib`, `sklearn` and `scipy`. If you have Anaconda on your computer they are likely already installed.

You also need the `dollarpy` library for the $P-Recognizer algorithm. You can install it with pip.
```bash
$ python3 -m pip install dollarpy
```

## Algorithms

This project aims to design a hand gesture recongnizer. Multiple datasets are provided and consist in a bunch of time series of hand gesture data points (in 3 dimensions). These hand gesture consist in shapes made multiple times by 10 different users. To compare these time series we use a **Fast DTW** algorithm and a **Point Clound recognizer** (more informations in the report folder).
