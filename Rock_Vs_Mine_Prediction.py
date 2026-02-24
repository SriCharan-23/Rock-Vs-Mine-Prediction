#Importing necessary libraries needed for our project
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Reading data from csv file

sonar_data = pd.read_csv("Copy of sonar data.csv", header=None)
# print(sonar_data.head())

#Now assigning feature value and target values to variables 'x' and 'y' respectively

x = sonar_data.drop(columns=60, axis=1)
y = sonar_data[60]
#Because the 60th column contains the name of object i.e., whether it is a rock or a mine based on sonar data that is present.

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1,stratify=y,random_state=1)
# Above I am splitting the data into testing and training data:
# test_size = 0.1 means 90% of data is used for training purpose and remaining 10% for testing purpose
# 'stratify' parameter is used to maintain same class distribution in trianing and testing data
# "random_state" parameter is used to control and handle randomness 

# Creating our model 
model = LogisticRegression()
model.fit(x_train,y_train) # Training our data

# Just analyzing the accuracy of training and test data

x_train_predict = model.predict(x_train)
training_data_accuracy = accuracy_score(x_train_predict,y_train)
print(training_data_accuracy)

x_test_predict = model.predict(x_test)
test_data_accuracy = accuracy_score(x_test_predict,y_test) 
print(test_data_accuracy)

input_data = input("Enter Sonar Data (Comma Separated): ")
input_data_list = input_data.split(',')  # splitting by comma
input_data_float = [float(i) for i in input_data_list] #Converting each individual string (sonar data) to float value
input_to_numpyArray = np.asarray(input_data_float, dtype=np.float64)  # Converting the input into a numpy array
input_data_reshaped = input_to_numpyArray.reshape(1,-1) #Reshaping our array from 1D to 2D

prediction = model.predict(input_data_reshaped)
print(prediction)

if(prediction[0] == 'R'):      # I have wrote 'prediction[0]' because it returns the result as a list..
    print("The object is a Rock!")
else:
    print("The object is a Mine!")
