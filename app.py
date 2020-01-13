import pandas as pd
from keras.models import Sequential
from keras.layers import *
training_data_df = pd.read_csv("sales_data_training_scaled.csv")

X = training_data_df.drop('total_earnings', axis=1).values
Y = training_data_df[['total_earnings']].values

model = Sequential()
model.add(Dense(50,input_dim=9 , activation="relu"))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(
    X,
    Y,
    epochs=50,
    shuffle=True,
    verbose=2
)

testing_data_df = pd.read_csv("sales_data_test_scaled.csv")

X_test = testing_data_df.drop('total_earnings', axis=1).values
Y_test = testing_data_df[['total_earnings']].values
test_error_rate = model.evaluate(X_test, Y_test, verbose=0)

print(test_error_rate)
prediction = model.predict(X)
prediction = prediction[0][0]
prediction = prediction + 0.1159
prediction = prediction / 0.0000036968
print(prediction)
actual = Y_test[0]
actual = actual + 0.1159
actual = actual / 0.0000036968
print(actual)


                