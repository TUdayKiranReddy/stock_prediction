import numpy as np
import pandas as pd
import datetime
from sklearn.preprocessing import *

def data_extractor_date(file_name):
	data = pd.read_excel(file_name)
	df = pd.DataFrame(data)
	y_data = df["NAV"].to_numpy()
	x_data = df["Date"].to_numpy()
	y =[]
	for i in range(y_data.shape[0]):
		y.append(float(y_data[i][:-1]))
	x = []
	month = 0
	date = 0
	year = 0
	for i in range(y_data.shape[0]):
		date = int(x_data[i][:2])
		month = datetime.datetime.strptime(x_data[i][3:6], "%b").month
		year = int(x_data[i][7:-1])
		x.append(np.array([date, month, year]))
	return np.array(x), np.array(y)

def data_extractor(file_name, no_of_prev_dpoints):
	data = pd.read_excel(file_name)
	df = pd.DataFrame(data)
	y_data = df["NAV"].to_numpy()[1:]
	scaler = MinMaxScaler(feature_range = (0,1))
	y =[]
	for i in range(y_data.shape[0]):
		y.append(float(y_data[i][:-1]))
	y = np.array(y).reshape((len(y),1))
	scaled_data = scaler.fit_transform(y)
	X, Y = [], []
	for j in range(no_of_prev_dpoints, len(y)):
		X.append(scaled_data[(j - no_of_prev_dpoints):j])
		Y.append(scaled_data[j])
	return scaler, np.array(X), np.array(Y)
