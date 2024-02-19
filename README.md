# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: Roopak C S
### Register Number: 212223220088
```python
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from google.colab import auth
import gspread
from google.auth import default

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

worksheet = gc.open('MyMLData').sheet1
data = worksheet.get_all_values()

dataset1 = pd.DataFrame(data[1:], columns=data[0])
dataset1 = dataset1.astype({'Input':'float'})
dataset1 = dataset1.astype({'Output':'float'})

dataset1.head()

#Assigning input column to X and output column to y
X = dataset1[['Input']].values
y = dataset1[['Output']].values

#Splitting testing and training data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,random_state = 33)

#Pre processing
Scaler = MinMaxScaler()

#Scaling the input for training
Scaler.fit(X_train)

#Transforming the scaled input
X_train1 = Scaler.transform(X_train)

#Creating the model
ai_brain=Sequential([
    Dense(units=1,activation='relu',input_shape=[1]),
    Dense(units=1)
])

#Compiling the model
ai_brain.compile(optimizer='rmsprop',loss='mse')

#Fitting the model
ai_brain.fit(X_train1,y_train,epochs=2000)

loss_df = pd.DataFrame(ai_brain.history.history)

loss_df.plot()

X_test1 = Scaler.transform(X_test)

ai_brain.evaluate(X_test1,y_test)

X_n1 = [[30]]

X_n1_1 = Scaler.transform(X_n1)

ai_brain.predict(X_n1_1)

```
## Dataset Information

Include screenshot of the dataset

## OUTPUT

### Training Loss Vs Iteration Plot

Include your plot here

### Test Data Root Mean Squared Error

Find the test data root mean squared error

### New Sample Data Prediction

Include your sample input and output here

## RESULT

Include your result here
