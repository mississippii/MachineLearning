import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

file_path = "/home/veer/Downloads/iris/iris.data"
column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]

data = pd.read_csv(file_path, header=None, names=column_names)

print(data.head())

plt.scatter(data['sepal_length'], data['sepal_width'])
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Sepal Length vs Sepal Width')
plt.show()

x = data[['sepal_length']]
y = data[['sepal_width']]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train,y_train)

slope = model.coef_[0]
intercept = model.intercept_
print(f"Slope (m): {slope}")
print(f"Intercept (c): {intercept}")

y_pred = model.predict(X_test)

plt.scatter(X_test, y_test, color='blue', label='Actual Data')
plt.plot(X_test, y_pred, color='red', label='Regression Line')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Simple Linear Regression')
plt.legend()
plt.show()

pre_y = model.predict(pd.DataFrame([[7.98]], columns=['sepal_length']))
print(f"Predict value (sepal_width) for sepal_length = 7.98: {pre_y}")
