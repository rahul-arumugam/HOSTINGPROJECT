import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

df=pd.read_csv('data.csv')



#preprocessing precessesss
df=df.rename(columns={'GRE Score':'GRE','TOEFL Score':'TOEFL','Chance of Admit':'Probability'})
df.drop('Serial No.', axis='columns',inplace=True)
df_copy = df.copy(deep=True)
df_copy[['GRE','TOEFL','University Rating','SOP','CGPA']] = df_copy[['GRE','TOEFL','University Rating','SOP','CGPA']].replace(0, np.NaN)
df_copy.isnull().sum()
#print(df.head())
#Splitting the data in features and label
x=df_copy.iloc[:,:-1]
y=df_copy.iloc[:,-1:]

#Splitting the dataset into train and test
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.20,random_state=5)
#Model creation
model=LinearRegression()
model.fit(X_train,Y_train)
#make a pickle file of our model
pickle.dump(model,open("model.pkl","wb"))