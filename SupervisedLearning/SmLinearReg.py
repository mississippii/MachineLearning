import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

file_path = "/home/veer/Downloads/Boston.csv"
data = pd.read_csv(file_path)
plt.scatter(data['rm'],data['medv'])
plt.xlabel('room Number')
plt.ylabel('median')
plt.title('Data set')
plt.show()

x = data[['rm']]
y = data[['medv']]

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
plt.xlabel('rm (Number of Rooms)')
plt.ylabel('medv (Median Value)')
plt.title('Simple Linear Regression')
plt.legend()
plt.show()

pre_y = model.predict([[7.98]])
print(f"Predict value (y): {pre_y}")
