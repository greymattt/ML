# ALGORITHM
# 1. Import the required libraries
# 2. Using the numpy get the values to do the prediction
# 3. Initialize the LinearRegression model from the sklearn library
# 4. Predict the slope and intercept using the model
# 5. predict the 'Y' value using the model
# 6. Plot the inital values of Y and predicted 'Y' using matplotlib

import numpy as np
import matplotlib.pyploy as plt
from sklearn.linear_model import LinearRegression

x = np.array([1,2,3,4,5])
y = np.array([10,20,40,40,50])
x.reshape(-1,1)

lr = LinearRegression()
lr.ft(x,y)
m = lr.coef_[0]
c = lr.intercept_
print("Slope: ", m)
print("Intercept: ", c)

x_pred = np.append(x, [6]).reshape(-1,1)
y_pred = lr.predict(x_pred)
print("Predicted value for ", x_pred[-1], "is", " y_pred[-1]")

plt.scatter(x, y, color='red', marker='o')
plt.scatter(x_pred, y_pred, color='blue', marker='^')
plt.plot(x_pred, y_pred, color='yellow')
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')
pl.show()