from sklearn import tree
from sklearn.naive_bayes import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
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


# CHALLENGE - ...and train them on our data
clf_TD = clf_TD.fit(X, Y)
clf_gnb = clf_gnb.fit(X, Y)
clf_Kneigh = clf_Kneigh.fit(X, Y)
clf_SVC = clf_SVC.fit(X, Y)

prediction_TD = clf_TD.predict([[190, 70, 43]])
prediction_gnb = clf_gnb.predict([[190, 70, 43]])
prediction_Kneigh = clf_Kneigh.predict([[190, 70, 43]])
prediction_SVC= clf_SVC.predict([[190, 70, 43]])


# CHALLENGE compare their reusults and print the best one!

print(f"DecisionTreeClassifierprediction:  {prediction_TD}")
print(f"GaussianNBprediction:  {prediction_gnb}")
print(f"Knearestprediction:  {prediction_Kneigh}")
print(f"SVMprediction:  {prediction_SVC}")
