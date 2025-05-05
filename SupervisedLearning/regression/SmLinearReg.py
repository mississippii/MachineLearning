import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

file_path = "/home/veer/Downloads/Dataset/iris.csv"
data = pd.read_csv(file_path, header=0)

print(data.head())

plt.scatter(data['sepal_length'], data['sepal_width'])
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Sepal Length vs Sepal Width')
plt.show()

x = data[str('sepal_length')]
y = data[str('sepal_width')]
# theta_0= 1.5;
# theta_1= 0.5;
# h = theta_0 + theta_1 * x;
re=0;
m = len(x)
for i in range(m):
    a=0.5 * x[i] + 1.5;
    a= y[i] - a;
    a=a**2;
    re=re+a;

cost = re/(2*m);
print("Cost: ", cost)
