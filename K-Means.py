import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

 # Import dataset
data = pd.read_csv('clustering.csv')
data.head()

# we will use only two features [‘ApplicantIncome’, ‘LoanAmount’] so that we can visualize the clusters on a 2D plane.
data = data.loc[:, ['ApplicantIncome', 'LoanAmount']]
data.head(2)

# Convert to numpy array
X = data.values
X[:5]

# Visualize the data points
sns.scatterplot(X[:,0], X[:, 1])

plt.xlabel('Income')
plt.ylabel('Loan')
plt.show()

# a Method to Calculate the Cost
def calculate_cost(X, centroids, cluster):
    sum = 0
    for i, val in enumerate(X):
        sum += np.sqrt((centroids[int(cluster[i]), 0] - val[0]) ** 2 + (centroids[int(cluster[i]), 1] - val[1]) ** 2)

    return sum



def kmeans(X, k):
    diff = 1
    cluster = np.zeros(X.shape[0])

    # select k random centroids
    random_indices = np.random.choice(len(X), size=k, replace=False)
    centroids = X[random_indices, :]

    while diff:

        # for each observation
        for i, row in enumerate(X):

            mn_dist = float('inf')
            # dist of the point from all centroids
            for idx, centroid in enumerate(centroids):
                d = np.sqrt((centroid[0] - row[0]) ** 2 + (centroid[1] - row[1]) ** 2)

                # store closest centroid
                if mn_dist > d:
                    mn_dist = d
                    cluster[i] = idx

        new_centroids = pd.DataFrame(X).groupby(by=cluster).mean().values

        # if centroids are same then leave
        if np.count_nonzero(centroids - new_centroids) == 0:
            diff = 0
        else:
            centroids = new_centroids
    return centroids, cluster


cost_list = []

for k in range(1, 10):
    centroids, cluster = kmeans(X, k)

    # WCSS (Within cluster sum of square)
    cost = calculate_cost(X, centroids, cluster)
    cost_list.append(cost)

k = 4
centroids, cluster = kmeans(X, k)

# The green dots represent the centroid for each cluster.
sns.scatterplot(X[:,0], X[:, 1], hue=cluster)
sns.scatterplot(centroids[:,0], centroids[:, 1], s=100, color='y')

plt.xlabel('Income')
plt.ylabel('Loan')
plt.show()




