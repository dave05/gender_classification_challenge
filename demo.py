from sklearn import tree
from sklearn.naive_bayes import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy
clf_TD = tree.DecisionTreeClassifier()
clf_SVC = svm.SVC()
clf_gnb= GaussianNB()
clf_Kneigh = KNeighborsClassifier(n_neighbors=3)
# CHALLENGE - create 3 more classifiers...
# 1
# 2
# 3

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

# CHALLENGE - ...and train them on our data
clf_TD = clf_TD.fit(X_train, y_train)
clf_gnb = clf_gnb.fit(X_train, y_train)
clf_Kneigh = clf_Kneigh.fit(X_train, y_train)
clf_SVC = clf_SVC.fit(X_train, y_train)

prediction_TD = clf_TD.predict(X_test)
prediction_gnb = clf_gnb.predict(X_test)
prediction_Kneigh = clf_Kneigh.predict(X_test)
prediction_SVC= clf_SVC.predict(X_test)

acc_TD = accuracy_score(y_test, prediction_TD)
acc_gnb= accuracy_score(y_test, prediction_gnb)
acc_svc= accuracy_score(y_test, prediction_SVC)
acc_Kneigh= accuracy_score(y_test, prediction_Kneigh)
index = numpy.argmax([acc_TD,acc_gnb,acc_svc,acc_Kneigh])
print([acc_TD,acc_gnb,acc_svc,acc_Kneigh])
classifers={0: 'DecisionTree', 1: 'NaiveBayes',2:'SVM', 3:'KNN'}

print(f"The Best Classifer for this dataset is {classifers[index]}")

# CHALLENGE compare their reusults and print the best one!

#print(f"DecisionTreeClassifierprediction:  {prediction_TD}")
#print(f"GaussianNBprediction:  {prediction_gnb}")
#print(f"Knearestprediction:  {prediction_Kneigh}")
#print(f"SVMprediction:  {prediction_SVC}")
...
