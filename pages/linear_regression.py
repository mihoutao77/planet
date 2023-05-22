import imp
import streamlit as st
from PIL import Image

st.title("Linear Regression")

header = 'Simple Linear Regression'
st.header(header)

subheader = 'Algorithm principle'
st.subheader(subheader)

msg = '''
Simple Linear Regression is a *statistical method* that allows us to summarize and study the relationship between two continuous (quantitative) variables. \n
It is probably one of the most popular algorithms for good inference in statistics and machine learning. \n
One of the variables is used as an explanatory variable and the other as a dependent variable.
'''
st.markdown(msg)

img = Image.open('images/Linear Regression.jpg')
st.image(img)

subheader = 'sample code'
st.subheader(subheader)

code = '''
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Load the diabetes dataset
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

# Use only one feature
diabetes_X = diabetes_X[:, np.newaxis, 2]

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

# Make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X_test)

# The coefficients
print("Coefficients: \n", regr.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(diabetes_y_test, diabetes_y_pred))

# Plot outputs
plt.scatter(diabetes_X_test, diabetes_y_test, color="black")
plt.plot(diabetes_X_test, diabetes_y_pred, color="blue", linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()
'''
st.code(code)

if st.button('run this example'):
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn import datasets, linear_model
    from sklearn.metrics import mean_squared_error, r2_score

    # Load the diabetes dataset
    diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

    # Use only one feature
    diabetes_X = diabetes_X[:, np.newaxis, 2]

    # Split the data into training/testing sets
    diabetes_X_train = diabetes_X[:-20]
    diabetes_X_test = diabetes_X[-20:]

    # Split the targets into training/testing sets
    diabetes_y_train = diabetes_y[:-20]
    diabetes_y_test = diabetes_y[-20:]

    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(diabetes_X_train, diabetes_y_train)

    # Make predictions using the testing set
    diabetes_y_pred = regr.predict(diabetes_X_test)

    fig = plt.figure()
    # Plot outputs
    plt.scatter(diabetes_X_test, diabetes_y_test, color="black")
    plt.plot(diabetes_X_test, diabetes_y_pred, color="blue", linewidth=3)
    plt.xticks(())
    plt.yticks(())
    st.pyplot(fig)

    st.write(
        "Coefficients: \n", regr.coef_
    )
    st.write(
        "Mean squared error:  %.2f" % mean_squared_error(diabetes_y_test, diabetes_y_pred)
    )
    st.write(
        "Coefficient of determination:  %.2f" % r2_score(diabetes_y_test, diabetes_y_pred)
    )

header = 'Multiple Linear Regression'
st.header(header)

subheader = 'Algorithm principle'
st.subheader(subheader)

msg = '''
Multiple linear regression (MLR), referred to as multiple regression, is a machine learning algorithm that uses multiple explanatory variables to predict the outcome of a response variable. \n
In reality, multiple regression is an extension of ordinary least squares regression (OLS) because it includes multiple explanatory variables.
'''
st.markdown(msg)

img = Image.open('images/Multiple Linear Regression.jpg')
st.image(img)

subheader = 'sample code'
st.subheader(subheader)

code = '''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from mpl_toolkits.mplot3d import Axes3D

######################################## Data preparation #########################################

file = 'https://aegis4048.github.io/downloads/notebooks/sample_data/unconv_MV_v5.csv'
df = pd.read_csv(file)

X = df[['Por', 'Brittle']].values.reshape(-1,2)
Y = df['Prod']

######################## Prepare model data point for visualization ###############################

x = X[:, 0]
y = X[:, 1]
z = Y

x_pred = np.linspace(6, 24, 30)   # range of porosity values
y_pred = np.linspace(0, 100, 30)  # range of brittleness values
xx_pred, yy_pred = np.meshgrid(x_pred, y_pred)
model_viz = np.array([xx_pred.flatten(), yy_pred.flatten()]).T

################################################ Train #############################################

ols = linear_model.LinearRegression()
model = ols.fit(X, Y)
predicted = model.predict(model_viz)

############################################## Evaluate ############################################

r2 = model.score(X, Y)

############################################## Plot ################################################

plt.style.use('default')

fig = plt.figure(figsize=(12, 4))

ax1 = fig.add_subplot(131, projection='3d')
ax2 = fig.add_subplot(132, projection='3d')
ax3 = fig.add_subplot(133, projection='3d')

axes = [ax1, ax2, ax3]

for ax in axes:
    ax.plot(x, y, z, color='k', zorder=15, linestyle='none', marker='o', alpha=0.5)
    ax.scatter(xx_pred.flatten(), yy_pred.flatten(), predicted, facecolor=(0,0,0,0), s=20, edgecolor='#70b3f0')
    ax.set_xlabel('Porosity (%)', fontsize=12)
    ax.set_ylabel('Brittleness', fontsize=12)
    ax.set_zlabel('Gas Prod. (Mcf/day)', fontsize=12)
    ax.locator_params(nbins=4, axis='x')
    ax.locator_params(nbins=5, axis='x')

ax1.text2D(0.2, 0.32, 'aegis4048.github.io', fontsize=13, ha='center', va='center',
           transform=ax1.transAxes, color='grey', alpha=0.5)
ax2.text2D(0.3, 0.42, 'aegis4048.github.io', fontsize=13, ha='center', va='center',
           transform=ax2.transAxes, color='grey', alpha=0.5)
ax3.text2D(0.85, 0.85, 'aegis4048.github.io', fontsize=13, ha='center', va='center',
           transform=ax3.transAxes, color='grey', alpha=0.5)

ax1.view_init(elev=28, azim=120)
ax2.view_init(elev=4, azim=114)
ax3.view_init(elev=60, azim=165)

fig.suptitle('$R^2 = %.2f$' % r2, fontsize=20)

fig.tight_layout()
'''
st.code(code)

if st.button('run the program'):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn import linear_model
    from mpl_toolkits.mplot3d import Axes3D

    ######################################## Data preparation #########################################

    file = 'https://aegis4048.github.io/downloads/notebooks/sample_data/unconv_MV_v5.csv'
    df = pd.read_csv(file)

    X = df[['Por', 'Brittle']].values.reshape(-1,2)
    Y = df['Prod']

    ######################## Prepare model data point for visualization ###############################

    x = X[:, 0]
    y = X[:, 1]
    z = Y

    x_pred = np.linspace(6, 24, 30)   # range of porosity values
    y_pred = np.linspace(0, 100, 30)  # range of brittleness values
    xx_pred, yy_pred = np.meshgrid(x_pred, y_pred)
    model_viz = np.array([xx_pred.flatten(), yy_pred.flatten()]).T

    ################################################ Train #############################################

    ols = linear_model.LinearRegression()
    model = ols.fit(X, Y)
    predicted = model.predict(model_viz)

    ############################################## Evaluate ############################################

    r2 = model.score(X, Y)

    ############################################## Plot ################################################

    plt.style.use('default')

    fig = plt.figure(figsize=(12, 4))

    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(133, projection='3d')

    axes = [ax1, ax2, ax3]

    for ax in axes:
        ax.plot(x, y, z, color='k', zorder=15, linestyle='none', marker='o', alpha=0.5)
        ax.scatter(xx_pred.flatten(), yy_pred.flatten(), predicted, facecolor=(0,0,0,0), s=20, edgecolor='#70b3f0')
        ax.set_xlabel('Porosity (%)', fontsize=12)
        ax.set_ylabel('Brittleness', fontsize=12)
        ax.set_zlabel('Gas Prod. (Mcf/day)', fontsize=12)
        ax.locator_params(nbins=4, axis='x')
        ax.locator_params(nbins=5, axis='x')

    ax1.text2D(0.2, 0.32, 'aegis4048.github.io', fontsize=13, ha='center', va='center',
            transform=ax1.transAxes, color='grey', alpha=0.5)
    ax2.text2D(0.3, 0.42, 'aegis4048.github.io', fontsize=13, ha='center', va='center',
            transform=ax2.transAxes, color='grey', alpha=0.5)
    ax3.text2D(0.85, 0.85, 'aegis4048.github.io', fontsize=13, ha='center', va='center',
            transform=ax3.transAxes, color='grey', alpha=0.5)

    ax1.view_init(elev=28, azim=120)
    ax2.view_init(elev=4, azim=114)
    ax3.view_init(elev=60, azim=165)

    fig.suptitle('$R^2 = %.2f$' % r2, fontsize=20)
    st.pyplot(fig)




