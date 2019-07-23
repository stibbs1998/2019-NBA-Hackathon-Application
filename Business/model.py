import numpy as np
import pandas as pd
import csv
import datetime as dt
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# Now we can create a Linear Regression model, using our description scores in place of the text!
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split 

# Load in the newly 'cleaned' data files.
train_clean = pd.read_csv('clean_training_set.csv')
test_clean = pd.read_csv('clean_test_set.csv')

# We are trying to predict the number of engagements on each post, so this column is our 'target' column.
# The remaining non-object columns will be used to model this number.
target = train_clean.Engagements
X = train_clean.drop(columns=['Engagements','Description'])

# Split the clean_training_set.csv file in order to train the Regression Model.
X_train, X_test, y_train, y_test = train_test_split(X, target, train_size = 0.75, 
                                                    random_state=1) 
# Create linear regression object. 
reg = LinearRegression() 

# Train the model using the training sets.
reg.fit(X_train, y_train);

# print(reg.score(X_train, y_train)) ## R^2 value

# Use the model to fit the clean_test_set.csv file.
X_holdout = test_clean.drop(columns=['Engagements','Description'])
engagement_prediction = reg.predict(X_holdout)

# Write these predicted values to the holdout_set.csv file.
holdout = pd.read_csv('holdout_set.csv',encoding='ISO-8859-1',dtype=str)
holdout.Engagements = engagement_prediction.astype(int)
holdout.to_csv('holdout_set.csv',index=False)
