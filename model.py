import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split

#Load Data
train_df = pd.read_csv("train.csv")

#Reordering Data and getting the useful ones
cols = train_df.columns.tolist()
cols = cols[2:3] + cols[4:] + cols[1:2]
train_df = train_df[cols]

# X_train, X_test, y_train, y_Test = train_test_split(train_df[2:], train_df[1], test_size=0.2)