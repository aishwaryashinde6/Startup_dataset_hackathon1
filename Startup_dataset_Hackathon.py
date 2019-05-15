# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 06:03:02 2019

@author: ADMIN
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler




dataset = pd.read_csv('startup_funding.csv')
#####################################################################################
np.random_seed=42
from sklearn.model_selection import train_test_split
train_set,test_set = train_test_split(dataset,test_size=0.2,random_state=42)
######################################################################################
"""for an updated data a common solution is to use each instances as identifier to decide
whether or not it should go in the test set(assuming instances have unique and immutable identifier)
we are computing each hash value of each instances identifier,keep just the last byte of the hash,
and put the instance in the test set if its value  is lower or equal to 51.This ensures the test set remain consistent across
multiple runs,even if you referesh the data.The new testset will contain 20% of the new instances,but  it will not contain
any instance that was previously in the training set."""   
 
import hashlib

def test_set_check(identifier, test_ratio, hash=hashlib.md5):
    return bytearray(hash(np.int64(identifier)).digest())[-1] < 256 * test_ratio  #return hash value of an identifier
    
def split_test_train_id(data,test_ratio,id_column,hash=hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_:test_set_check(id_,test_ratio,hash))
    return data.loc[~in_test_set], data.loc[in_test_set]

startup_dataset = dataset.reset_index()  #adds index column
train_set,test_set = split_test_train_id(startup_dataset,0.2,"index")

#################################################################################
#Handling null vlue in mumerical data

startup_dataset["AmountInUSD"] = startup_dataset["AmountInUSD"].str.replace(",","").astype("float")
median = startup_dataset["AmountInUSD"].median()
startup_dataset["AmountInUSD"]=startup_dataset["AmountInUSD"].fillna(median)
####################

#chnaging null value with mean using scikit learn lib
import numpy as np
from sklearn.impute import SimpleImputer
imp_mean =  SimpleImputer(missing_values=np.nan,strategy="mean")
imp_mean.fit([startup_dataset["AmountInUSD"]])
X = imp_mean.transform([startup_dataset["AmountInUSD"]])
X = X.reshape(2372,1)

###################################################################################
#categorical data handling
startup_dataset["CityLocation"].fillna("No City",inplace=True)
startup_dataset["StartupName"].fillna("No Name",inplace=True)
startup_dataset["IndustryVertical"].fillna("None",inplace=True)
startup_dataset["InvestorsName"].fillna("Nil",inplace=True)
startup_dataset["SubVertical"].fillna("not defined",inplace=True)
startup_dataset["Remarks"].fillna("...",inplace=True)

cat_column = startup_dataset[['StartupName','CityLocation','InvestorsName','IndustryVertical']]

#using columntranformer for the handling categorical and numerical data

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer
preprocess = make_column_transformer(
    (['AmountInUSD'], StandardScaler()),
    (['CityLocation', 'IndustryVertical', 'StartupName','InvestorsName'], OneHotEncoder())
)
pipeline = preprocess.fit_transform(startup_dataset).toarray()


################################################################################


#Training set(all values excepy amountinusd)
#startup_dataset train and test split on whole dataset
np.random_seed=42
from sklearn.model_selection import train_test_split
train_set_sd,test_set_sd = train_test_split(startup_dataset,test_size=0.2,random_state=42)



X_train = preprocess.fit_transform(train_set_sd).toarray()   #numpy array doesnt have drop function
X_train_final = pd.DataFrame(X_train)       #converted to Dataframe to drop AmountInUSD col
X_train_final = X_train_final.drop(X_train_final.columns[0], axis=1) #all values except  amount colm

X_test = preprocess.fit_transform(test_set_sd).toarray()   #numpy array doesnt have drop function
X_test_final = pd.DataFrame(X_test)       #converted to Dataframe to drop AmountInUSD col
X_test_final = X_test_final.drop(X_test_final.columns[0], axis=1) 

#testing data(amountinusd)
#label(X=AmountInUSD)
np.random_seed=42
from sklearn.model_selection import train_test_split
train_set_label,test_set_label = train_test_split(X,test_size=0.2,random_state=42)


#training and evaluating on the Training set
#Linear Regression
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X_train_final,train_set_label) 

"""lin_reg.score(X_train_final,train_set_label)
 0.999999567296408"""

from sklearn.metrics import mean_squared_error
predictions = lin_reg.predict(X_train_final)  
lin_mse = mean_squared_error(train_set_label,predictions)
lin_mse = np.sqrt(lin_mse)
lin_mse                 #37281.00 (prediction error)the model isnt powerful


#DecisonTree
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(X_train_final,train_set_label)
startup_prediction = tree_reg.predict(X_train_final)
tree_mse = mean_squared_error(train_set_label,startup_prediction)
tree_mse = np.sqrt(tree_mse)
tree_mse                       #16142.36

#better evaluation using cross_validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg,X_train_final,train_set_label,
                         scoring="neg_mean_squared_error",cv=10)
rmse_scores = np.sqrt(-scores)

def display_scores(scores):
    print("Scores" , scores)
    print("mean" , scores.mean())
    print("standard deviation" , scores.std())


display_scores(rmse_scores)
#mean 53449513.75095071
#standard deviation 36094668.66883968

#triying out same score on lin reg model
lin_score = cross_val_score(lin_reg,X_train_final,train_set_label,
                            scoring="neg_mean_squared_error",cv=10)
lin_rmse_score = np.sqrt(-lin_score)
display_scores(lin_rmse_score)
#mean 2.6307826032402707e+17
#standard deviation 2.8743767295734957e+17


#RandomForest
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(X_train_final,train_set_label)

forest_prediction=forest_reg.predict(X_train_final)
forest_mse=mean_squared_error(train_set_label,forest_prediction)
forest_rmse=np.sqrt(forest_mse)
forest_rmse                                  # 26104192.282713786


forest_scores=cross_val_score(forest_reg,X_train_final,train_set_label,cv=10,
                              scoring="neg_mean_squared_error")
forest_rmse_scores=np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)
#mean 43871834.1313106
#standard deviation 30942419.258820225


################################################################################
#fine_tune_your_model grid search
#best combination of hyperparameters for RandomForest
from sklearn.model_selection import GridSearchCV
param_grid = [
        {'n_estimators': [3,10,15] , 'max_features': [2,4,6,8]},
        {'bootstrap':[False], 'n_estimators':[3,10],'max_features':[2,3,4]}
    ]
forest_reg = RandomForestRegressor()
grid_search=GridSearchCV(forest_reg,param_grid,cv=5,scoring="neg_mean_squared_error")
grid_search.fit(X_train_final,train_set_label)
grid_search.best_params_   #{'bootstrap': False, 'max_features': 2, 'n_estimators': 3}
grid_search.best_estimator_
grid_search.cv_results_['mean_test_score']
grid_search.best_score_

result=grid_search.cv_results_
for mean_score,params in zip(result["mean_test_score"],result["params"]):
    print(np.sqrt(-mean_score),params)
#rmse =  54385010

################################################################################
 #evaluate on test set

final_model = grid_search.best_estimator_
grid_search.fit(X_test_final,test_set_label)
final_prediction = final_model.predict(X_test_final)
final_mse = mean_squared_error(final_prediction,test_set_label)
final_rmse = np.sqrt(final_mse)   #8897088.20380020


def display_scores(scores):
    print("Scores" , scores)
    print("mean" , scores.mean())
    print("standard deviation" , scores.std())
display_scores(final_rmse)    
############################################################################
#Data Visualization

#facet grids and categorical data
import seaborn as sns
sns.factorplot(x="AmountInUSD",y="Date",hue="InvestmentType",col="CityLocation",
               kind="bar",data=startup_dataset)

import matplotlib.pyplot as plt
plt.show()

g=sns.factorplot(x="AmountInUSD",y="CityLocation",kind="box",data=startup_dataset)
g.fig.set_size_inches(15,15)
plt.show()

#1.Year when most startups received funding=2016

#converted datatype to datetime,and created a year column 
startup_dataset['Date'] = startup_dataset['Date'].apply(lambda x:x.replace('.','/'))
startup_dataset['Date'] = startup_dataset['Date'].apply(lambda x:x.replace('/','//'))
startup_dataset['Date'] = pd.to_datetime(startup_dataset['Date'],dayfirst=True)
startup_dataset['Year'] = startup_dataset['Date'].apply(lambda x:x.year)

grouped = startup_dataset[["Year","StartupName"]].groupby(by="Year").count()
import seaborn as sns
plot1 = grouped.plot.barh(stacked="True",alpha=0.5)
plot1.fig.set_size_inches(15,15)
plt.show()

######################################################################################
#2.averAGE amount received by the startup
#very few startups received funding in the year 2017 yet the funnding was highest as compred to 2016
import seaborn as sns
sns.barplot(x='Year',y='AmountInUSD',data=data)
plt.xlabel('Year')
plt.ylabel('Average amount (in Millions USD)')
plt.title('Average amount received by startups (in Millions USD)')
plt.tight_layout()


#########################################################################
#3.Technology startups receiving most of the funds
#consumer Internet has most counts
startup_dataset["IndustryVertical"].value_counts
y = startup_dataset['IndustryVertical'].value_counts()[:20].plot(kind='barh')
y.fig.set_size_inches(25,15)

#realtionship between consumer internet industry with the AmountInUSD(53000000)
data = startup_dataset[['IndustryVertical','Year','AmountInUSD']]
print(data.loc[data['IndustryVertical'] == 'Consumer Internet'])

consumer_internet_data = data.loc[data['IndustryVertical'] == 'Consumer Internet']

sns.barplot(x='IndustryVertical',y='AmountInUSD',data=consumer_internet_data)
plt.xlabel('consumer internet')
plt.ylabel('Average amount (in Millions USD)')
plt.title('Average amount received by consumer internet (in Millions USD)')
plt.tight_layout()
#############################################################################



















