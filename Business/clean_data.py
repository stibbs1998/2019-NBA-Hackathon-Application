import numpy as np
import pandas as pd
import csv
import datetime as dt
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# Load the original data files.
train = pd.read_csv('training_set.csv',encoding='ISO-8859-1',dtype=str) 
test = pd.read_csv('holdout_set.csv',encoding='ISO-8859-1',dtype=str)

# Convert 'Engagements' and 'Followers' columns to int64.
train['Engagements'] = train['Engagements'].astype(int)
train['Followers at Posting'] = train['Followers at Posting'].astype(int)
test['Followers at Posting'] = test['Followers at Posting'].astype(int)

# Convert the 'Created' columns to Datetime objects.
train.Created = pd.to_datetime(train.Created)
test.Created = pd.to_datetime(test.Created)

# Split the 'Created' column into the month, day of the week the post occured on, and the time of the day in seconds.
def split_time(df):
    df['Day_of_Week'] = [d.date().weekday() for d in df['Created']]
    df['Time'] = [d.time() for d in df['Created']]
    df[['Hour','Minute','Seconds']] = df.Time.astype(str).str.split(':',expand=True).astype(int)
    df['Time_in_seconds'] = df.Hour * 3600 + df.Minute*60 + df.Seconds
    df = df.drop(columns=['Created','Time','Hour','Minute','Seconds'])
    return df    

train = split_time(train)
test = split_time(test)

# One hot-encode the 'Type' column, and drop the original 'Type' column.
train[['Album','Photo','Video']] = pd.get_dummies(train.Type)
test[['Album','Photo','Video']] = pd.get_dummies(test.Type)

train = train.drop(columns='Type')
test = test.drop(columns='Type')

# Generate sentiment analysis scores for each post's description.
# Boilerplate for this code was taken from https://towardsdatascience.com/sentiment-analysis-with-python-part-1-5ce197074184

######################################################
# First convert all words to lower case and strip the text of any special characters.
import re

REPLACE_NO_SPACE = re.compile("[.;@:&!\'?,\"()\[\]]")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

def preprocess_reviews(text):
    text = [REPLACE_NO_SPACE.sub("", str(line).lower()) for line in text]
    text = [REPLACE_WITH_SPACE.sub(" ", str(line)) for line in text]
    
    return text

train_clean = preprocess_reviews(train['Description'])
test_clean = preprocess_reviews(test['Description'])
train.Description = train_clean
test.Description = test_clean
######################################################
# Perform a Logistic Regression of the post descriptions v. the quality of the number of engagements.
# The 'quality' of the posts will be defined as: "Good" posts are in the top 30% of engagements, all others are "Bad" posts.
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

cv = CountVectorizer(binary=True)
cv.fit(train_clean)
X = cv.transform(train_clean)
X_test = cv.transform(test_clean)

target = (train.Engagements > np.nanpercentile(train.Engagements,70) ).astype(int) # "Good" posts are assigned a 1, "Bad" are assigned 0's

X_train, X_val, y_train, y_val = train_test_split(
    X, target, train_size = 0.75
)

# Search for the best C value for the Logistic Regression
best_score = 0
best_c = 0 
for c in [0.01, 0.05, 0.25, 0.5, 1]:
    
    lr = LogisticRegression(C=c)
    lr.fit(X_train, y_train)
    print ("Accuracy for C=%s: %s" 
           % (c, accuracy_score(y_val, lr.predict(X_val))))
    if accuracy_score(y_val, lr.predict(X_val)) > best_score:
        best_score = accuracy_score(y_val, lr.predict(X_val))
        best_c = c
        
print("Best C value: C=%f" %best_c)

# Using the best value, create and fit the final model.
final_model = LogisticRegression(C=best_c)
final_model.fit(X, target);

# Create a dictionary containing all words from the pool of descriptions and their cooresponding sentiment values.
feature_to_coef = {
    word: coef for word, coef in zip(
        cv.get_feature_names(), final_model.coef_[0]
    )
}

# For each description, sum up the scores associated with each word in the description, and assign this total score to the description.
def get_description_score(df):
    post_score = []
    for row in df.Description.str.split():
        count= 0
        for word in row:
            if word in feature_to_coef.keys():
                count += feature_to_coef[word]
        post_score.append(count)
    df['Description_Score'] = post_score
    return df
train = get_description_score(train)
test = get_description_score(test)

# Write these 'cleaned' data files to new .csv files to run our model on.
train.to_csv('clean_training_set.csv',index=False)
test.to_csv('clean_test_set.csv',index=False)
