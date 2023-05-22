import streamlit as st
from PIL import Image

st.title("K-Means Clustering Algorithm")

header = 'Algorithm principle'
st.header(header)

msg = '''
K-means clustering is one of the most popular clustering algorithms and is the first algorithm applied by practitioners when solving clustering tasks to obtain the structure of a dataset. \n
It's an algorithm that, given a dataset, will identify which data points belong to each of k clusters, take your data and learn how to group it. \n
Through a series of iterations, the algorithm creates groups of data points called clusters that have similar variances and minimizes a specific cost function: the sum of squares within the cluster.
'''
st.markdown(msg)

img = Image.open('images/K-Means Clustering Algorithm.jpg')
st.image(img)

header = 'sample code'
st.header(header)

code = '''
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt

# Though the following import is not directly being used, it is required
# for 3D projection to work with matplotlib < 3.2
import mpl_toolkits.mplot3d  # noqa: F401

from sklearn.cluster import KMeans
from sklearn import datasets

np.random.seed(5)

iris = datasets.load_iris()
X = iris.data
y = iris.target

estimators = [
    ("k_means_iris_8", KMeans(n_clusters=8)),
    ("k_means_iris_3", KMeans(n_clusters=3)),
    ("k_means_iris_bad_init", KMeans(n_clusters=3, n_init=1, init="random")),
]

fignum = 1
titles = ["8 clusters", "3 clusters", "3 clusters, bad initialization"]
for name, est in estimators:
    fig = plt.figure(fignum, figsize=(4, 3))
    ax = fig.add_subplot(111, projection="3d", elev=48, azim=134)
    ax.set_position([0, 0, 0.95, 1])
    est.fit(X)
    labels = est.labels_

    ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=labels.astype(float), edgecolor="k")

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel("Petal width")
    ax.set_ylabel("Sepal length")
    ax.set_zlabel("Petal length")
    ax.set_title(titles[fignum - 1])
    ax.dist = 12
    fignum = fignum + 1

# Plot the ground truth
fig = plt.figure(fignum, figsize=(4, 3))
ax = fig.add_subplot(111, projection="3d", elev=48, azim=134)
ax.set_position([0, 0, 0.95, 1])

for name, label in [("Setosa", 0), ("Versicolour", 1), ("Virginica", 2)]:
    ax.text3D(
        X[y == label, 3].mean(),
        X[y == label, 0].mean(),
        X[y == label, 2].mean() + 2,
        name,
        horizontalalignment="center",
        bbox=dict(alpha=0.2, edgecolor="w", facecolor="w"),
    )
# Reorder the labels to have colors matching the cluster results
y = np.choose(y, [1, 2, 0]).astype(float)
ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=y, edgecolor="k")

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel("Petal width")
ax.set_ylabel("Sepal length")
ax.set_zlabel("Petal length")
ax.set_title("Ground Truth")
ax.dist = 12

fig.show()
'''

st.code(code)

if st.button('run this example'):

    import numpy as np
    import matplotlib.pyplot as plt

    # Though the following import is not directly being used, it is required
    # for 3D projection to work with matplotlib < 3.2
    import mpl_toolkits.mplot3d  # noqa: F401

    from sklearn.cluster import KMeans
    from sklearn import datasets

    np.random.seed(5)

    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    estimators = [
        ("k_means_iris_8", KMeans(n_clusters=8)),
        ("k_means_iris_3", KMeans(n_clusters=3)),
        ("k_means_iris_bad_init", KMeans(n_clusters=3, n_init=1, init="random")),
    ]

    fignum = 1
    titles = ["8 clusters", "3 clusters", "3 clusters, bad initialization"]
    for name, est in estimators:
        fig = plt.figure(fignum, figsize=(4, 3))
        ax = fig.add_subplot(111, projection="3d", elev=48, azim=134)
        ax.set_position([0, 0, 0.95, 1])
        est.fit(X)
        labels = est.labels_

        ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=labels.astype(float), edgecolor="k")

        ax.w_xaxis.set_ticklabels([])
        ax.w_yaxis.set_ticklabels([])
        ax.w_zaxis.set_ticklabels([])
        ax.set_xlabel("Petal width")
        ax.set_ylabel("Sepal length")
        ax.set_zlabel("Petal length")
        ax.set_title(titles[fignum - 1])
        ax.dist = 12
        fignum = fignum + 1

    # Plot the ground truth
    fig = plt.figure(fignum, figsize=(4, 3))
    ax = fig.add_subplot(111, projection="3d", elev=48, azim=134)
    ax.set_position([0, 0, 0.95, 1])

    for name, label in [("Setosa", 0), ("Versicolour", 1), ("Virginica", 2)]:
        ax.text3D(
            X[y == label, 3].mean(),
            X[y == label, 0].mean(),
            X[y == label, 2].mean() + 2,
            name,
            horizontalalignment="center",
            bbox=dict(alpha=0.2, edgecolor="w", facecolor="w"),
        )
    # Reorder the labels to have colors matching the cluster results
    y = np.choose(y, [1, 2, 0]).astype(float)
    ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=y, edgecolor="k")

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel("Petal width")
    ax.set_ylabel("Sepal length")
    ax.set_zlabel("Petal length")
    ax.set_title("Ground Truth")
    ax.dist = 12

    st.pyplot(fig)
