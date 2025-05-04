import  matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dataPath = "/home/veer/Downloads/Dataset/iris.csv"

dataFrame = pd.read_csv(dataPath, header=0)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Sepal Length vs. Sepal Width')
plt.scatter(dataFrame['sepal_length'], dataFrame['sepal_width'])
plt.show()
# x = np.linspace(-10, 10, 150)
# y = 2*np.sin(x)
# plt.scatter(x, y)
# plt.title("Plot of y = sin(x)")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.show()