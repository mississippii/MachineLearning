import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

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

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, Y_train.ravel())

rf_predictionsTrain = rf_model.predict(X_train)
rf_accuracy = metrics.accuracy_score(Y_train, rf_predictionsTrain)
print("Random Forest Accuracy: {}".format(rf_accuracy))

rf_predictionsTest = rf_model.predict(X_test)
rf_accuracy = metrics.accuracy_score(Y_test, rf_predictionsTest)
print("Random Forest Accuracy: {}".format(rf_accuracy))
# fpr, tpr
naive_bayes = np.array([0.28, 0.63])
logistic = np.array([0.77, 0.77])
random_forest = np.array([0.88, 0.24])
ann = np.array([0.12, 0.76])

# plotting
plt.scatter(naive_bayes[0], naive_bayes[1], label = 'Naive Bayes', facecolors='black', edgecolors='orange', s=300)
plt.scatter(logistic[0], logistic[1], label = 'Logistic Regression', facecolors='orange', edgecolors='orange', s=300)
plt.scatter(random_forest[0], random_forest[1], label = 'Random Forest', facecolors='blue', edgecolors='black', s=300)
plt.scatter(ann[0], ann[1], label = 'Artificial Neural Network', facecolors='red', edgecolors='black', s=300)

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc='lower center')
plt.show()