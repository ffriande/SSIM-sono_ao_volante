from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from config import *
import tensorflow_addons as tfa


def init_model(input_size, output_size):
    # define the keras model
    model = Sequential()
    model.add(Embedding(input_dim=input_size, output_dim=output_size))
    model.add(LSTM(40))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(NR_STAGES, activation='sigmoid'))
    # compile the keras model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[tfa.metrics.CohenKappa(num_classes=NR_STAGES), 'accuracy'])
    return model

def fit_model(Xtrain, ytrain, model, Xtest, ytest):
    # fit the keras model on the dataset
    model.fit(Xtrain, ytrain, epochs=15, batch_size=64, validation_data =(Xtest, ytest))

    # evaluate the keras model
    loss, kappa, accuracy = model.evaluate(Xtest, ytest)
    print('Accuracy: %.2f' % (accuracy*100), 'Kappa: %.2f' % (kappa), 'Loss: %.2f' % (loss))