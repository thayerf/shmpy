from keras.models import Sequential
from keras.layers import Dense, TimeDistributed, SimpleRNN, Dropout, Conv2D, Flatten, Conv1D, MaxPooling2D, Reshape


def build_crnn_2d(encoding_length):
      model = Sequential()
      model.add(TimeDistributed(Conv2D(64,kernel_size = (10,encoding_length), strides = (1,1), activation = 'relu'), input_shape = (None,308,encoding_length,1)))
      model.add(TimeDistributed(Reshape((299,64,1))))
      model.add(TimeDistributed(MaxPooling2D()))
      model.add(TimeDistributed(Conv2D(32, kernel_size = (8,4), strides = (1,1), activation = 'relu')))
      model.add(TimeDistributed(MaxPooling2D()))
      model.add(TimeDistributed(Flatten()))
      model.add(TimeDistributed(Dense(16, activation = 'relu')))
      model.add(SimpleRNN(9, activation = 'linear'))
      return(model)
      
def build_crnn_1d(encoding_length):
      model = Sequential()
      model.add(TimeDistributed(Conv1D(64,kernel_size = 10, strides = 1, activation = 'relu'), input_shape = (None,308*encoding_length,1)))
      model.add(TimeDistributed(Reshape((308*encoding_length-9,64,1))))
      model.add(TimeDistributed(MaxPooling2D()))
      model.add(TimeDistributed(Conv2D(32, kernel_size = (8,4), strides = (1,1), activation = 'relu')))
      model.add(TimeDistributed(MaxPooling2D()))
      model.add(TimeDistributed(Flatten()))
      model.add(TimeDistributed(Dense(16, activation = 'relu')))
      model.add(SimpleRNN(9, activation = 'linear'))
      return(model)
      
def build_ffrnn(encoding_length):
      model = Sequential()
      model.add(TimeDistributed(Dense(512, activation = 'relu'), input_shape = (None,308*encoding_length)))
      model.add(Dropout(.2))
      model.add(TimeDistributed(Dense(256, activation = 'relu')))
      model.add(Dropout(.2))
      model.add(TimeDistributed(Dense(128, activation = 'relu')))
      model.add(Dropout(.2))
      model.add(TimeDistributed(Dense(64, activation = 'relu')))
      model.add(Dropout(.2))
      model.add(SimpleRNN(9, activation = 'linear'))
      return(model)