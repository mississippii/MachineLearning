import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.impute import SimpleImputer


# Data loading
dat_path = "/home/veer/Downloads/Dataset/diabetes1.csv"
data_frame = pd.read_csv(dat_path, header=0)

#Data preprocessing

del data_frame['skin']
mp_diabetes = {True: 1, False: 0}
data_frame['diabetes'] = data_frame['diabetes'].map(mp_diabetes)
print(data_frame.head())
correlation = data_frame.corr()
fig, heatmap = plt.subplots(figsize=(10, 10))
heatmap.matshow(correlation)
plt.xticks(range(len(correlation.columns)), correlation.columns)
plt.yticks(range(len(correlation.columns)), correlation.columns)
plt.show()

# Splitting data
feature_column_names = ['num_preg', 'glucose_conc', 'diastolic_bp', 'thickness', 'insulin', 'bmi', 'diab_pred', 'age']
predicted_class_name = ['diabetes']

X = data_frame[feature_column_names].values
Y = data_frame[predicted_class_name].values

split_test_size = 0.3
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=split_test_size,random_state=42)
print("{0:0.2f}% in training set".format((len(X_train)/len(data_frame.index)) * 100))
print("{0:0.2f}% in test set".format((len(X_test)/len(data_frame.index)) * 100))

print("# rows in dataframe {0}".format(len(data_frame)))
print("# rows missing glucose_conc: {0}".format(len(data_frame.loc[data_frame['glucose_conc'] == 0])))
print("# rows missing diastolic_bp: {0}".format(len(data_frame.loc[data_frame['diastolic_bp'] == 0])))
print("# rows missing thickness: {0}".format(len(data_frame.loc[data_frame['thickness'] == 0])))
print("# rows missing insulin: {0}".format(len(data_frame.loc[data_frame['insulin'] == 0])))
print("# rows missing bmi: {0}".format(len(data_frame.loc[data_frame['bmi'] == 0])))
print("# rows missing diab_pred: {0}".format(len(data_frame.loc[data_frame['diab_pred'] == 0])))
print("# rows missing age: {0}".format(len(data_frame.loc[data_frame['age'] == 0])))

fill_0 = SimpleImputer(missing_values=0, strategy='mean')
X_train = fill_0.fit_transform(X_train)
X_test = fill_0.transform(X_test)
#Model building
model = GaussianNB()
model.fit(X_train,Y_train.ravel())
prediction_from_trained_data = model.predict(X_train)
accuracy = metrics.accuracy_score(Y_train, prediction_from_trained_data)

print ("Accuracy of our naive bayes model is : {0:.4f}".format(accuracy))
prediction_from_test_data = model.predict(X_test)

accuracy = metrics.accuracy_score(Y_test, prediction_from_test_data)

print ("Accuracy of our naive bayes model is: {0:0.4f}".format(accuracy))

print ("Confusion Matrix")

print ("{0}".format(metrics.confusion_matrix(Y_test, prediction_from_test_data, labels=[1, 0])))
print ("Classification Report")

print ("{0}".format(metrics.classification_report(Y_test, prediction_from_test_data, labels=[1, 0])))