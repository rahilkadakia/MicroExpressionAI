import keras
from keras.applications.vgg16 import VGG16
from keras.engine import training
from keras.models import Sequential
from keras.layers import Flatten, Dense, LSTM, Input
from keras.models import Model

IMAGE_SIZE = [224, 224]
train_path = 'D:\\BE_Project\\SAMM\\006\\006_1_2'
test_path = ''

def temporal_learning(data_dim, timesteps_TIM, classes, weights_path=None):
    model = Sequential()
    model.add(LSTM(3000, return_sequences=False, input_shape=(timesteps_TIM, data_dim)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(classes, activation='sigmoid')) # try replacing sigmoid with softmax
    
    if weights_path:
	    model.load_weights(weights_path)

    return model


vgg_model = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False) # If RGB, then [3]; if Grayscale, then [1]
vgg_model.trainable = False # Freezing the base layer
# print (vgg_model.summary())

inputs = Input(shape=(224,224,3))
features = vgg_model(inputs, training=False)

features = keras.layers.GlobalAveragePooling2D()(features)
outputs = Dense(10)(features)

model = Model(inputs, outputs)
print(model.summary())