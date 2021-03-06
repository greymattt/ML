# ALGORITHM:
# Step 1 : Start
# Step 2 : Import the required packages, from sklearn import svm.
# Step 3 : Import the dataset.
# Step 4 : Shape the data for training the model.
# Step 5 : Define and train the model.
# Step 6 : Get the weight value for linear equation from the trained SVM model
# Step 7 : Get the y- offset value for the linear equation and make the x-axis space for the data points.
# Step 8 : Plot the decision boundary by getting the y- value 
# Step 9 : Plot the decision boundary
# Step 10: Display the output
# Step 11: Stop


import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm

X = np.array([2, 5, 1, 6, 1, 9, 7, 8.7, 2.9, 5.5, 7.7, 6.9]) 
y = np.array([1, 8, 1, 7, 0.6, 11, 10, 9.4, 4, 3, 7.9, 6.1]) 
training_X = np.vstack((X, y)).T
training_y = [0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1]

clf = svm.SVC(kernel='linear', C=1.0) 
clf.fit(training_X, training_y)

w = clf.coef_[0]
a = -w[0] / w[1]
XX = np.linspace(0, 13)
yy = a * XX - clf.intercept_[0] / w[1]

plt.plot(XX, yy, 'k-')
plt.scatter(training_X[:, 0], training_X[:, 1], c=training_y) 
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()
plt.show()