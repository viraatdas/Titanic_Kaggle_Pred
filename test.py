from model import preprocess, clf
import pandas as pd
import csv

#Load Data
test_df = pd.read_csv("test.csv")

#Preprocess
cols = test_df.columns.tolist()
cols = cols[0:7] + cols[8:9] + cols[10:]
test_df = test_df[cols]

#clean the data
test_df = test_df.dropna()

#get pass_id
cols = test_df.columns.tolist()
p_cols = cols[0:1]
pass_id = test_df[p_cols]
cols = cols[1:2] + cols[3:]
test_df = test_df[cols]

test_df = preprocess(test_df)

predict = clf.predict(test_df)

predict_df = pd.DataFrame({"Survived": predict})
print(pass_id)
print(predict_df)
