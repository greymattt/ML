# Procedure:
# Step 1: Importing necessary python libraries.
# Step 2: Loading the dataset and pre-processing. 
# Step 3: Splitting the dataset for x and y values. 
# Step 4: Splitting the dataset for training and testing.
# Step 5: Creating the model using DecsisionTreeClassifier with CART algorithm. 
# Step 6: Fitting the model using training data
# Step 7: Predicting the values and testing.
# Step 8: Visualizing the decision tree.

import pandas as pd
import graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz


data = pd.read_csv('data.csv')
le = LabelEncoder()
dataset = data.iloc[:, :]

for i in dataset:
	dataset[i] = le.fit_transform(dataset[i])

X = dataset[:, :4].values
y = dataset[:, 4].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1, random_state=2)

model = DecisionTreeClassifier(criterion='gini')
model.fit(X_train, y_train)

if model.predict([[2, 1, 0, 1]] == 1:
	print("laptop can be provided")
else:
	print("laptop cannot be provided")

dot = export_graphviz(model, out_file=None)
graph = graphviz.Source(dot)
graph