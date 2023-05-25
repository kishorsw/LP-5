import tensorflow as tf
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
#Loads the Boston Housing dataset into training and testing sets. The input features are stored in x_train and x_test, while the corresponding target values are stored in y_train and y_test.
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

# Normalize the features
mean = x_train.mean(axis=0)#Computes the mean value for each feature in the training data along the 0th axis (column-wise). This will be used for normalization.
std = x_train.std(axis=0)
x_train = (x_train - mean) / std #Normalizes the input features by subtracting the mean and dividing by the standard deviation. This step standardizes the features, making them have zero mean and unit variance.
x_test = (x_test - mean) / std

#x_test[5] // input for prediction
#y_test[5] // actual value

model = Sequential()
model.add(Dense(128,activation='relu',input_shape = (x_train[0].shape))) #Adds a Dense layer with 128 units and ReLU activation function as the first hidden layer. It takes the shape of the input features as the input shape.
model.add(Dense(64,activation='relu'))#Adds a Dense layer with 64 units and ReLU activation function as the second hidden layer.
model.add(Dense(32,activation='relu'))
model.add(Dense(1))

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')#onfigures the model for training. The optimizer is set to Adam with a learning rate of 0.001, and the loss function is set to mean squared error (MSE).
#Trains the model on the training data. The training data (x_train and y_train) is used, and the training is performed for 100 epochs with a batch size of 32. The validation data (x_test and y_test) is used to evaluate the model's performance during training. The verbose argument is set to 1 to display training progress during each epoch.
model.fit(x_train, y_train, batch_size=32, epochs=100, verbose=1, validation_data=(x_test,y_test))

loss = model.evaluate(x_test, y_test, verbose=0)
print(f"Mean Squared Error (MSE): {loss:.2f}")
#Evaluates the model on the testing data (x_test and y_test) and computes the loss.
test_input = [[-0.3754937 , -0.48361547, -0.20791668, -0.25683275,  0.23597582,-0.48113631, -0.94641237, -0.67000565, -0.39603557, -0.08965908, 0.32944629,  0.44807713,  0.11720047]]
predicted_value = model.predict(test_input)
print("actual value is :",y_test[4])
print("predicted value is : ",predicted_value)
