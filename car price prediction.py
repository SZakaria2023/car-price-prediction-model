
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("/Users/DELL/OneDrive/سطح المكتب/project/train.csv")

print(df.columns) #the name of the columns
print(df.dtypes) #Data type of each column
print(df.shape) #shape
#Generate descriptive statistics / central tendency, dispersion and shape 

print(df.describe()) 
print(df.describe(include=[object()]))
print(df.describe(include='all'))

print(df) #return the first 5 rows,and the last 5 rows
print(pd.options.display.max_rows) #check your system's maximum rows

print(pd.options.display.max_columns) #check your system's maximum columns

pd.options.display.max_rows=1000 #change the maximum rows number 
pd.options.display.max_columns=200 #change the maximum columns number 

df.head() #returns the headers and a specified number of rows, starting from the top
df.head(1000)

print(df.taill()) #returns the headers and a specified number of rows, starting from the bottom
print(df.info()) #gives you more information about the data set

print(df.insull().sum()) #Number of missing values for each column

df1=df.drop(columns='Cabin') #Drop the Cabin column

#replacing the missing values from the colomn by it's mean

mean=df1['Age'].mean()
df1['Age'].fillna(mean,inplace=True)
print(df1['Age'])

# or replacing with the most probabel values for the qualitative data 
most_repeated=df1['Embarked'].mode()[0]
df1['Embarked'].fillna(most_repeated,inplace=True)

#Counting survived and not survived passengers

Survived=df1['Survived'].value_counts()
print(Survived)

#bar plot

labels=['Died','Survived']
plt.bar(labels, Survived)
plt.xlabel("Survived or not")
plt.ylabel("Passangers count")
plt.title("Counting survived and not survived passangers")
plt.show()

#bar 2

Pclass=df1['Pclass'].value_counts()
print(Pclass)

labels=['1st','2nd','3rd']
plt.bar(labels, Pclass)
plt.xlabel("Passanger class")
plt.ylabel("Passangers count")
plt.show()

#bar 3

Sex=df1['Sex'].value_counts()
print(Sex)

labels=['male','female']
plt.bar(labels, Sex)
plt.xlabel("Sex")
plt.ylabel("Passangers count")
plt.show()

#Histogram plot

df1['Age'].hist()
plt.title("Distribution of passangers ages on the titanic")
plt.xlabel("Age")
plt.ylabel("count")
plt.show()

#removing unnecery columns

print(df1.columns)
df2=df1.drop(columns=['PassengerId','Name','Ticket'])
print(df2.columns)

#Encoding categorical columns

#Map the 'sex' column to numirical values

sex_map= {'male':0,'female':1}
df2['Sex'] =df2['Sex'].replaced(sex_map)
print(df2)

#Map the 'Embarked' column to numirical values

Embarked_map= {'S':0,'C':1,'Q':2}
df2['Embarked'] =df2['Embarked'].replaced(Embarked_map)
print(df2)

#Plotting Boxplot 
#for age
df2.boxplot(column ='Age' ,vert=False)
plt.show()

#for fare
df2.boxplot(column ='Fare' ,vert=True)
plt.show()

#for the two of them
df2.boxplot(column =['Age','Fare'] ,vert=False)
plt.show()

#Survival Prediction Model

#Separating features and target
Y_train= df2['Survived']
X_train=df2.drop(columns=['Survived'])

#Define the model Logistic Regression
from sklearn.linear_model import LogisticRegression
model=LogisticRegression() 

#Train the model with training data

model.fit(X_train,Y_train)

#Evaluate the model

#Evaluate the model on the training data

Y_train_prediction= model.predict(X_train)

#compare with the original values
from sklearn.metrics import accuracy_score
training_accuracy= accuracy_score(Y_train,Y_train_prediction,round())

#save the model

import pickle

filename="my_model.pickle"

pickle.dump(decision_tree, open(filename,"wb"))

#Separating features and target of test_dataset
Y_test=df['Survived']
X_test=df.drop(columns=['Survived'])

#load model

loaded_model = pickle.load(open("my_model.pickle","rb"))

#you can use loaded models to compute predictions
Y_predicted = loaded_model.predict(X_test)
test_accuracy=accuracy_score(Y_test,Y_predicted)







