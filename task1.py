#TASK 1
#M.PRANEETHAA

#importing the required libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# importing the dataset as a CSV file
data=pd.read_csv("C:/Users/praneethaa m/Desktop/task1dataset.csv")
data
data.head(10)
data.corr()
#plotting the given dataset
df=data[['Hours','Scores']]
sns.pairplot(df,kind="scatter")

plt.show()

#SPLITTING THE DATASET
X = data.iloc[:, :-1].values  
y = data.iloc[:, 1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=0) 


#training dataset
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 


line = regressor.predict(X)

# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line,color='red');
plt.show()

#predicted values vs actual values
y_pred = regressor.predict(X_test) 
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df 
df.plot.bar()

#The predicted value
hours=9.25
pred_score=regressor.predict([[hours]])
print("HOURS={}".format(hours))
print("PREDICTED SCORES={}".format(pred_score[0]))


#EVALUATING THE ACCURACY OF THE MODEL
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred))
