# Procedure:
# 1. Load the iris dataset from datasets package. To keep things simple, take only first two columns (i.e sepal length and sepal width respectively).
# 2. Now plot the dataset.
# 3. Fit the data as a mixture of 3 Gaussians.
# 4. Then do the clustering, i.e assign a label to each observation. Also find the number of
# iterations needed for the log-likelihood function to converge and the converged log-likelihood value.
# 5. Print the converged log-likelihood value and no. of iterations needed for the model to converge.

# CODE:
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.mixture import GaussianMixture

iris = datasets.load_iris()
X = iris.data[:, :2]
d = pd.DataFrame(X)
plt.scatter(d[0], d[1])

gmm = GaussianMixture(n_components=3)
gmm.fit(d)

labels = gmm.predict(d)
d['labels'] = labels
d0 = d[d['labels'] == 0]
d1 = d[d['labels'] == 1]
d2 = d[d['labels'] == 2]

plt.scatter(d0[0], d0[1], c='red')
plt.scatter(d1[0], d1[1], c='yellow')
plt.scatter(d2[0], d2[1], c='green')

print(gmm.lower_bound_)
print(gmm.n_iter_)