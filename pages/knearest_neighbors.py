import streamlit as st
from PIL import Image

st.title("K-Nearest Neighbors Algorithm")

header = 'Algorithm principle'
st.header(header)

msg = '''
k Nearest Neighbors (kNN) is a supervised learning algorithm that can be used to solve classification and regression tasks. \n
The main idea behind this algorithm is that the value or class of a data point is determined by the data points around it. \n
The kNN classifier uses majority voting to determine the class of a data point. Since the model needs to store all the data points, kNN becomes very slow as the number of data points increases. \n
Therefore, it is also not very memory efficient. Another disadvantage of kNN is that it is sensitive to outliers. \n
'''
st.markdown(msg)

img = Image.open('images/K-Nearest Neighbors Algorithm.jpg')
st.image(img)

header = 'sample code'
st.header(header)

code = '''

# Import necessary modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
 
irisData = load_iris()
 
# Create feature and target arrays
X = irisData.data
y = irisData.target
 
# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(
             X, y, test_size = 0.2, random_state=42)
 
neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))
 
# Loop over K values
for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
     
    # Compute training and test data accuracy
    train_accuracy[i] = knn.score(X_train, y_train)
    test_accuracy[i] = knn.score(X_test, y_test)
 
# Generate plot
plt.plot(neighbors, test_accuracy, label = 'Testing dataset Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training dataset Accuracy')
 
plt.legend()
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.show()
'''

st.code(code)

if st.button("run this example"):

# Import necessary modules
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_iris
    import numpy as np
    import matplotlib.pyplot as plt
    
    irisData = load_iris()
    
    # Create feature and target arrays
    X = irisData.data
    y = irisData.target
    
    # Split into training and test set
    X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size = 0.2, random_state=42)
    
    neighbors = np.arange(1, 9)
    train_accuracy = np.empty(len(neighbors))
    test_accuracy = np.empty(len(neighbors))
    
    # Loop over K values
    for i, k in enumerate(neighbors):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        
        # Compute training and test data accuracy
        train_accuracy[i] = knn.score(X_train, y_train)
        test_accuracy[i] = knn.score(X_test, y_test)
    
    # Generate plot
    fig = plt.figure()
    plt.plot(neighbors, test_accuracy, label = 'Testing dataset Accuracy')
    plt.plot(neighbors, train_accuracy, label = 'Training dataset Accuracy')
    
    plt.legend()
    plt.xlabel('n_neighbors')
    plt.ylabel('Accuracy')
    st.pyplot(fig)  
