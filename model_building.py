import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def build_model(input_shape):
    model = Sequential()
    model.add(Dense(64, input_shape=(input_shape,), activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(3, activation='sigmoid'))  # Output layer for 3 diseases
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
