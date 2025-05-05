import csv
import numpy as np
import  seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def computeCost(X, y, theta):
    m = len(y)
    hypothesis = X.dot(theta)
    loss = hypothesis - y
    cost = sum(loss ** 2)
    return cost / (2 * m)

def gradientDescent(X, y, theta, alpha, iterations):
    m = len(y)
    cost = []
    for i in range(iterations):
        loss = X.dot(theta) - y
        gradient = X.T.dot(loss)
        theta = theta - (alpha/m) * gradient
        cost.append(computeCost(X, y, theta))
        print("Iteration: {0}, Cost: {1}".format(i, cost[i]))
    return theta, cost


datpath = "/home/veer/Downloads/Dataset/coursera.csv"
df = pd.read_csv(datpath)

# Extract the columns as NumPy arrays
population = df['Population'].to_numpy()
profit = df['Profit'].to_numpy()

df = pd.DataFrame()
df['Population'] = population
df['Profit'] = profit
sns.lmplot(x='Population', y='Profit', data=df, fit_reg= True,scatter_kws={'s':45})
plt.show()

# Converting loaded dataset into numpy array
# Example:
# X = [[1, 10],
#      [1, 20],
#      [1, 30]]
#
X = np.concatenate((np.ones(len(population)).reshape(len(population),1), population.reshape(len(population),1)), axis=1)
# Example
# y = [[1],
#      [2],
#      [3]]
y= np.array(profit).reshape(len(profit),1)
theta = np.zeros((2,1))
alpha = 0.01
iterations = 15000
theta, cost = gradientDescent(X, y, theta, alpha, iterations)