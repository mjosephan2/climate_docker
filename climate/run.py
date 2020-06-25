import pickle
import tensorflow as tf
import numpy as np
import glob
import pygrib
from sklearn.model_selection import train_test_split

from os.path import join

# parameter
timestep = 4
height = 361
width = 720
num_target = 4

def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
  data = []
  labels = []

  print('start_index',start_index)
  print('end_index', end_index)
  print('len(dataset)',len(dataset))
  start_index = start_index + history_size 
  print('start_index (new)',start_index)
  if end_index is None:
    end_index = len(dataset) - target_size 
    print('end_index (new)', end_index)
  for i in range(start_index, end_index): 
    # indices = list(range(i-history_size, i, step))
    if single_step:
      temp_y = [target[i+target_size]]
    else:
      temp_y = target[i:i+target_size]
    temp_x = dataset[i-history_size:i]
    if len(temp_x) != history_size or len(temp_y) != target_size:
      continue
    labels.append(temp_y)
    data.append(temp_x) 
  print(len(data), 'len of data', start_index, end_index)
  print(len(labels), 'len of labels', start_index, end_index)
  return np.array(data), np.array(labels)
  
def normalization(data):
  x_max = data.min(axis=(1,2),  keepdims=True)
  x_min = data.max(axis=(1,2),  keepdims=True)
  return (data - x_min) / (x_max - x_min)

def prepare_data():
    path = "data/temperature0401.pkl"
    data = []
    with open(path, "rb") as f:
        while True:
            try:
                obj = pickle.load(f)
            except EOFError:
                break
            data.append(obj)
    data.sort(key=lambda x: x[0])
    data_april_1 = data[:128]
    data_april_1 = np.array([f[1] for f in data_april_1])
    data_april_1 = data_april_1.reshape((-1, 361, 720, 1))
    data_april_1 = normalization(data_april_1)

    split_index = int(len(data_april_1) * 0.8)
    train_data = data_april_1[:split_index]
    validation_data = data_april_1[split_index:]

    xtrain_data, ytrain_data = multivariate_data(train_data, train_data, 0, len(train_data), timestep, num_target, 1)
    xval_data, yval_data = multivariate_data(validation_data, validation_data, 0, len(validation_data), timestep, num_target, 1)

    dataset_train = tf.data.Dataset.from_tensor_slices((xtrain_data,ytrain_data))
    dataset_train = dataset_train.batch(1)

    dataset_val = tf.data.Dataset.from_tensor_slices((xval_data, yval_data))
    dataset_val = dataset_val.batch(1)
    return (dataset_train, dataset_val)

def conv_lstm():
    inputs = tf.keras.layers.Input(shape=(timestep, 361, 720, 1))
    x = tf.keras.layers.ConvLSTM2D(64, kernel_size=(1,1), padding="same", return_sequences=True)(inputs)
    x = tf.keras.layers.ConvLSTM2D(32, kernel_size=(1,1), padding="same", return_sequences=True)(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1))(x)
    model = tf.keras.Model(inputs=inputs, outputs=x)
    model.compile(optimizer='adam', loss='mse', metrics=['mse','mae'])
    return model
dataset_train, dataset_val = prepare_data()
conv_lstm = conv_lstm()
conv_lstm.summary()

conv_lstm.fit(dataset_train, validation_data=dataset_val, epochs=10)
conv_lstm.save("model/cnn_lstm_test.h5")