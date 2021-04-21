from keras.applications.vgg16 import VGG16
from keras.backend import relu
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, LSTM

vgg_model = VGG16()
# print (vgg_model.summary())

def temporal_learning(data_dim, timesteps_TIM, classes, weights_path=None):
    model = Sequential()
    model.add(LSTM(3000, return_sequences=False, input_shape=(timesteps_TIM, data_dim)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(classes, activation='sigmoid')) # try replacing sigmoid with softmax
    
    if weights_path:
	    model.load_weights(weights_path)

    return model

