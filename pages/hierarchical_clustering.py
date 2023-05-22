import streamlit as st
from PIL import Image

st.title("Hierarchical Clustering Algorithm")

header = 'Algorithm principle'
st.header(header)

msg = '''
Hierarchical clustering refers to creating a tree of clusters by iteratively grouping or separating data points. There are two types of hierarchical clustering known as agglomerative clustering and divisive clustering. \n
Aggregative clustering is a bottom-up approach. It merges the two points that are most similar until all points are merged into one cluster. \n
Split clusters are a top-down approach. It starts with all points as a cluster and splits out the most dissimilar clusters at each step until only single data points remain. \n
One of the advantages of hierarchical clustering is that we don't have to specify the number of clusters (but we can)。
'''
st.markdown(msg)

img = Image.open('images/Hierarchical Clustering Algorithm.jpg')
st.image(img)

header = 'sample code'
st.header(header)

code = '''
import numpy as np

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


iris = load_iris()
X = iris.data

# setting distance_threshold=0 ensures we compute the full tree.
model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)

model = model.fit(X)
plt.title('Hierarchical Clustering Dendrogram')
# plot the top three levels of the dendrogram
plot_dendrogram(model, truncate_mode='level', p=3)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()
'''
st.code(code)

if st.button("run this example"):
    import numpy as np

    from matplotlib import pyplot as plt
    from scipy.cluster.hierarchy import dendrogram
    from sklearn.datasets import load_iris
    from sklearn.cluster import AgglomerativeClustering


    def plot_dendrogram(model, np, dendrogram,**kwargs):
        # Create linkage matrix and then plot the dendrogram

        # create the counts of samples under each node
        counts = np.zeros(model.children_.shape[0])
        n_samples = len(model.labels_)
        for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        linkage_matrix = np.column_stack([model.children_, model.distances_,
                                        counts]).astype(float)

        # Plot the corresponding dendrogram
        dendrogram(linkage_matrix, **kwargs)


    iris = load_iris()
    X = iris.data

    # setting distance_threshold=0 ensures we compute the full tree.
    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)

    model = model.fit(X)
    fig = plt.figure()
    plt.title('Hierarchical Clustering Dendrogram')
    # plot the top three levels of the dendrogram
    plot_dendrogram(model, np, dendrogram, truncate_mode='level', p=3)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    st.pyplot(fig)


