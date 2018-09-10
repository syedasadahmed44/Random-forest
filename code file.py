# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 20:27:33 2018

@author: DELL
"""


#IMPORTING REQUIRED LIBRARIES 
import sqlite3
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE


# Create your connection to soccer database
cnx = sqlite3.connect('database.sqlite')

# create a dataframe using table Player_Attributes of soccer database
df = pd.read_sql_query("SELECT * FROM Player_Attributes", cnx)
df.head()

# view summary of dataframe
df.describe()

# view shape of data frame
df.shape
print(f"Rows : {df.shape[0]} \nColumns : {df.shape[1]}")


# print column names in readable format
[(f"column {i+1} : {column}") for i,column in enumerate(df.columns)]


#DATA PROCESSING 
#Create a new dataframe after dropping some columns which are not useful to predict player overall ratings
soccer_data = df.drop(["id", "player_fifa_api_id", "player_api_id", "date"], axis = 1)

#Check whether there are duplicates entries present or not
soccer_data.duplicated().any()


#Drop duplicates entries from soccer_data dataframe
soccer_data.drop_duplicates(inplace=True)

#check dataframe shape after dropping duplicate entries
soccer_data.shape



#check number of missing parameters in the DataFrame - Columnwise
soccer_data.isnull().values.sum()



# functions to handle missing data
def data_preprocessing(df):
    df.convert_objects(convert_numeric=True)
    df.fillna(-99999, inplace=True)     
    return df

#call function to hanbdle missing data
data_preprocessing(soccer_data)

#check number of missing parameters in the DataFrame - Columnwise after handling missing data (should be 0)
soccer_data.isnull().values.sum()

soccer_data = pd.get_dummies(soccer_data)
soccer_data.head(1)



# view shape of data frame
soccer_data.shape
print(f"shape of the DataFrame after one hot encoding is : {soccer_data.shape}")

soccer_data.describe()


#Visualize column overall_rating of the dataframe
soccer_data['overall_rating'].value_counts()

soccer_data['overall_rating'].value_counts().plot(kind='bar',figsize=(20,10))



#Split dataframe into df_x and df_y
df_x = soccer_data.drop(['overall_rating'], 1)
df_y = np.array(soccer_data['overall_rating'])



#Use RandomForestRegressor to check feature_importances
rfc_1 = RandomForestRegressor(random_state=10)
rfc_1.fit(df_x, df_y)

feature_importances = pd.DataFrame({'feature':df_x.columns,'importance':np.round(rfc_1.feature_importances_,4)})
feature_importances = feature_importances.sort_values('importance',ascending=False).set_index('feature')
print(feature_importances[:35])


#Visualize top 35 features of the dataframe
feature_importances[:35].plot(kind='barh',figsize=(30,10))



#Create dataset for train, test and cross-validation
x, x_test, y, y_test = train_test_split(df_x,df_y,test_size=0.2,train_size=0.8, random_state = 55)
x_train, x_cv, y_train, y_cv = train_test_split(x,y,test_size = 0.20,train_size =0.80, random_state = 55)



#Apply scaling on dataframe df_x
from sklearn.preprocessing import StandardScaler, Normalizer, scale
df_x1 = scale(x_train)



#Perfrom PCA (dimensionality reduction technique) on scaled dataframe
from sklearn.decomposition import PCA
from sklearn.metrics import explained_variance_score
# on non-standardized data

pca = PCA(n_components=10).fit(x_train)

#df_x1 = PCA(n_components=10).fit_transform(df_x)


pca.explained_variance_ratio_


from sklearn.decomposition import PCA
# on non-standardized data
df_x2 = pca.transform(x_train)


#map test and cross-validation data using pca 
pca.transform(x_test)


pca.transform(x_cv)


#Apply Model (Linear regression, Decision tree, Random forest and xgboost)
#Apply linear regression model on the dataset
lr1 = LinearRegression()
lr = RFE(lr1, 20)
lr.fit(x_train,y_train)



#Apply decision tree model on the dataset
d_tree = DecisionTreeRegressor(min_samples_split=10, random_state=55)
d_tree.fit(x_train, y_train)




#Apply Random Forest model on the dataset
rfc = RandomForestRegressor(random_state=99)
rfc.fit(x_train, y_train)



# Apply xgboost model on the dataset
Boosting = xgb.XGBRegressor(n_estimators=200,learning_rate=1)
Boosting.fit(x_train,y_train)




#Analyze mse(mean squared error) and accuracy

models = pd.DataFrame(index=['train_mse','cv_mse','test_mse','accuracy_score'], columns=['linear_regression','decision_tree','random_forest','xgboost'])

models.loc['train_mse','linear_regression'] = mean_squared_error(y_pred=lr.predict(x_train), y_true=y_train)
models.loc['cv_mse','linear_regression'] = mean_squared_error(y_pred=lr.predict(x_cv), y_true=y_cv)
models.loc['test_mse','linear_regression'] = mean_squared_error(y_pred=lr.predict(x_test), y_true=y_test)
models.loc['accuracy_score','linear_regression'] = accuracy_score(y_pred=lr.predict(x_test).round(), y_true=y_test)

models.loc['train_mse','decision_tree'] = mean_squared_error(y_pred=d_tree.predict(x_train), y_true=y_train)
models.loc['cv_mse','decision_tree'] = mean_squared_error(y_pred=d_tree.predict(x_cv), y_true=y_cv)
models.loc['test_mse','decision_tree'] = mean_squared_error(y_pred=d_tree.predict(x_test), y_true=y_test)
models.loc['accuracy_score','decision_tree'] = accuracy_score(y_pred=d_tree.predict(x_test).round(), y_true=y_test)

models.loc['train_mse','random_forest'] = mean_squared_error(y_pred=rfc.predict(x_train), y_true=y_train)
models.loc['cv_mse','random_forest'] = mean_squared_error(y_pred=rfc.predict(x_cv), y_true=y_cv)
models.loc['test_mse','random_forest'] = mean_squared_error(y_pred=rfc.predict(x_test), y_true=y_test)
models.loc['accuracy_score','random_forest'] = accuracy_score(y_pred=rfc.predict(x_test).round(), y_true=y_test)

models.loc['train_mse','xgboost'] = mean_squared_error(y_pred=Boosting.predict(x_train), y_true=y_train)
models.loc['cv_mse','xgboost'] = mean_squared_error(y_pred=Boosting.predict(x_cv), y_true=y_cv)
models.loc['test_mse','xgboost'] = mean_squared_error(y_pred=Boosting.predict(x_test), y_true=y_test)
models.loc['accuracy_score','xgboost'] = accuracy_score(y_pred=Boosting.predict(x_test).round(), y_true=y_test)

models




#As observed from above table, random forest model is providing highest accuracy and lowest mean squared error.
 #So Random forest is best model to predict players rating.