import numpy as np
import matplotlib.pyplot as plt
import cvxpy
import pandas as pd

class Investment_Optimiser:
	
	def __init__(self, Data, inital_sum):
		self.data = Data
		self.s = inital_sum
		self.n = None
		self.c_i = self.data["current_value"]
		self.c = self.data["future_value"]
		
	def run(self):
		x = cvxpy.Variable(self.c_i.shape[0])
		max_stocks = np.divide(self.s, self.c_i)
		constraint = [ self.c_i.T@x == self.s, np.zeros((self.c_i.shape[0],)) <= x, x <= max_stocks ]
		returns = self.c.T@x
		Problem = cvxpy.Problem(cvxpy.Maximize(returns), constraint)
		Problem.solve(solver=cvxpy.GLPK)
		self.n = x
		#print(self.s, returns.value)
		print("\n\nInvestment:", self.s)
		print("Stock", '\t|\t', 'Quantity', '\t|\t', 'Invested_value', '\t|\t', 'Predicted_Value')
		for i in range(self.c_i.shape[0]):
			if(x.value[i] >= 0):
				print(self.data["stocks"][i], '\t|\t', round(x.value[i], 3), '\t|\t', round(self.c_i[i]*x.value[i], 3), '\t\t|\t', round(self.c[i]*x.value[i], 3))
		print("Predicted Sum Value:", returns.value)


data = {}
data["stocks"] = ['a', 'b', 'c', 'd', 'f']
data["current_value"] = np.array([20, 30, 50, 58, 98])
data["future_value"] = np.array([25.065, 35.0565, 52.098, 58.21, 105])
x = Investment_Optimiser(data, 10000)

x.run()
#print(x.n.value)
