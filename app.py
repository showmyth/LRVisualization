# streamlit app header
import streamlit as st
st.title('Linear Regression Workings Visualization and associated EDA on Boston Housing Dataset')
# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# cache the data set
@st.cache_data
def load_data():
    boston_df = pd.read_csv("BostonHousing.csv")
    return boston_df

boston_df = load_data()

# show the table we're working with
st.write("We will be working with the Boston Housing Dataset which is derived from information collected by the U.S. Census Service concerning housing in the area of Boston MA")
st.dataframe(boston_df.head())
st.write("Here's the information about our Boston Housing Dataset", boston_df.describe())

# finding missing values
st.write("Missing Values")
col1, col2 = st.columns([1, 1])
with col1:
    st.write(boston_df.isna().sum())

# clean dataset
st.write("After cleaning the data, we get the dataset as such:")
boston_df_cleaned = boston_df.dropna() # remove empty examples
st.dataframe(boston_df_cleaned.head())

# find correlation coefficients
corr_boston = boston_df_cleaned.corr().round(2) # round the correlation coeffs upto 2 decimal places 
# make a heatmap for the given correlation coeffs

fig, ax = plt.subplots(figsize = (9,9))
sns.heatmap(data = corr_boston, vmin = -1, vmax = 1, center=0, annot = True, cmap='vlag', ax = ax)
# auxillary appearance properties
ax.set_xticklabels(ax.get_xticklabels(), color='white')
ax.set_yticklabels(ax.get_yticklabels(), color='white')
fig.patch.set_alpha(0)
ax.patch.set_alpha(0)

# display plot
st.pyplot(fig)


# Numerical Determination
st.write("Alternatively, we use numerical determinaion to find which attributrs have the correlation above threshold value")
corr = abs(corr_boston.MEDV) > 0.5
st.write(corr[corr])

# Scatterplots

st.write("Comparing LSTAT to MEDV using scatterplot")
fig, ax = plt.subplots(figsize=(7,7))
plt.scatter(x = boston_df_cleaned['LSTAT'], y = boston_df_cleaned['MEDV'], marker = 'o', c = 'red')
plt.xlabel('LSTAT')
plt.ylabel('MEDV')

# auxillary appearance properties
ax.set_xticklabels(ax.get_xticklabels(), color='white')
ax.set_yticklabels(ax.get_yticklabels(), color='white')
fig.patch.set_alpha(0)

st.pyplot(fig)

st.write("Comparing RM to MEDV using scatterplot")
fig, ax = plt.subplots(figsize=(7,7))
plt.scatter(x = boston_df_cleaned['RM'], y = boston_df_cleaned['MEDV'], marker = 'o', c = 'green')
plt.xlabel('RM')
plt.ylabel('MEDV')

# auxillary appearance properties
ax.set_xticklabels(ax.get_xticklabels(), color='white')
ax.set_yticklabels(ax.get_yticklabels(), color='white')
fig.patch.set_alpha(0)

st.pyplot(fig)

# 2-D Scatterplot

# creating objects
fig = plt.figure(figsize = (9,9))
ax = fig.add_subplot(111, projection='3d')

# making the scatter plot
st.write("Here's a 2-D Scatterplot for the given attributes 'RM' and 'LSTAT' and their correlation to 'MEDV'")
ax.scatter3D(boston_df_cleaned['LSTAT'], boston_df_cleaned['RM'], boston_df_cleaned['MEDV'])
ax.set_xlabel('LSTAT')
ax.set_ylabel('RM')
ax.set_zlabel('MEDV')

st.pyplot(fig)

# Training Model

# define features = LSTAT and RM
st.write("The selected features are LSTAT and RM due to their high correlation magnitude")
X = pd.DataFrame(np.c_[boston_df_cleaned['LSTAT'], boston_df_cleaned['RM']], columns = ['LSTAT','RM'])
st.dataframe(X.head())

# define the target
st.write("Our target is thus MEDV")
y = boston_df_cleaned['MEDV']
st.dataframe(y.head())

# create training and test data
from sklearn.model_selection import train_test_split as tts
X_train, X_test, y_train, y_test = tts(X, y, test_size = 0.25, random_state = 5)


st.write("The shapes of test and train data are as follows: ")
st.write(X_train.shape)
st.write(y_train.shape)

# create the said model
from sklearn.linear_model import LinearRegression as LR

housing_model = LR()
housing_model.fit(X_train, y_train)

# model evaluation
price_pred = housing_model.predict(X_test)

# for corr coeff = R: 
st.write('R-squared: %.4f'%housing_model.score(X_test, y_test))


# finding MSE
from sklearn.metrics import mean_squared_error as MSE

Err = MSE(y_test, price_pred)
st.write("The error obtained is",Err)






