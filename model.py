import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
import numpy as np

#Load Data
train_df = pd.read_csv("train.csv")

#Reordering Data and getting the useful ones
cols = train_df.columns.tolist()
cols = cols[2:3] + cols[4:8] + cols[9:10] + cols[11:12] + cols[1:2]
train_df = train_df[cols]

age_avg = train_df['Age'].mean()
emb_mode = train_df['Embarked'].value_counts().idxmax() #S
for index, row in train_df.iterrows():

    #Make all Nan age the average
    if np.isnan(row[2]):
        train_df.at[index, 'Age'] = age_avg

    #Replace male with 0 and female with 1
    if row[1] == 'male':
        train_df.at[index, 'Sex'] = 0
    else:
        train_df.at[index, 'Sex'] = 1

    #Replacing embarked destinations with numeric values
    if row[6] == 'C':
        train_df.at[index, 'Embarked'] = 0
    elif row[6] == 'Q':
        train_df.at[index, 'Embarked'] = 1
    elif row[6] == 'S':
        train_df.at[index, 'Embarked'] = 2
    else: #For Nan values put the most showed location; determined above
        train_df.at[index, 'Embarked'] = 2


print(train_df)


#Split into train and test
X_train, X_test, y_train, y_test = train_test_split(train_df.iloc[:, 0:-1], train_df.iloc[:,-1], test_size=0.2)

#SVC gamma scale
clf = svm.SVC(gamma='scale')
clf.fit(X_train, y_train)
predict_y = clf.predict(X_test)
accuracy = sum(1 for x,y in zip(predict_y, y_test) if x == y) / float(len(predict_y))
print(accuracy)