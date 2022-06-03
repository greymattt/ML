# ALGORITHM:
# Step 1: start
# Step 2: Specify number of clusters K.
# Step 3: Initialize centroids by first shuffling the dataset and then randomly
# selecting K data points for the centroids without replacement.
# Step 4: Keep iterating until there is no change to the centroids. i.e assignment of data
# points to clusters isn’t changing.
# Step 5: Compute the sum of the squared distance between data points and all centroids.
# Step 6: Assign each data point to the closest cluster (centroid).
# Step 7: Compute the centroids for the clusters by taking the average of the all data points that belong to each cluster.
# Step 8: stop

PROGRAM:
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_iris
import sklearn.metrics as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=load_iris()
X=pd.DataFrame(dataset.data) 
X.columns=['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width' ]
y=pd.DataFrame(dataset.target)
y.columns=['Targets']
plt.figure(figsize=(14,7)) 
colormap=np.array(['red','lime','black'])
plt.subplot(1,3,1) 
plt.scatter(X.Petal_Length,X.Petal_Width,c=colormap[y.Targets],s=40) 
plt.title('Real')
plt.subplot(1,3,2)
model=KMeans(n_clusters=3)
model.fit(X) 
predY=np.choose(model.labels_,[0,1,2]).astype(np.int64) 
plt.scatter(X.Petal_Length,X.Petal_Width,c=colormap[predY],s=40) 
plt.title('KMeans')