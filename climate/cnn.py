import pickle
import tensorflow as tf
import numpy as np
import glob
import pygrib
from sklearn.model_selection import train_test_split

from os.path import join

# parameter
timestep = 2
height = 361
width = 720
num_parameters = 241
num_target = 2

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

def normalize(datas):
  x_min = datas.min(axis=(0,1),  keepdims=True)
  x_max = datas.max(axis=(0,1),  keepdims=True)
  x_norm = (datas - x_min) / (x_max - x_min)
  # there might be nan if all the values are 0
  # replace all nan to 0
  wherenan = np.isnan(x_norm)
  x_norm[wherenan] = 0
  return x_norm.astype("float32")

def _parse_pickle_file(x):
    x = x.numpy()
    with open(x, 'rb') as f:
        data = pickle.load(f)
        # data = next(iter(data.values()))
        # temp_x, temp_y = data["x"], data["y"]
        # temp = np.concatenate([temp_x, temp_y], 2)
        x_data = normalize(data["x"])
        y_data = normalize(data["y"])
    return x_data, y_data

def _function(x):
  x, y = tf.py_function(_parse_pickle_file, [x], [tf.float32, tf.float32])
  x.set_shape((height, width, num_parameters))
  y.set_shape((height, width, 1))
  return x,y

def cnn_lstm():
  # parameter
  input_data = tf.keras.layers.Input((height, width, num_parameters))
  x = tf.keras.layers.Conv2D(128, (5,5), activation='relu', padding='same')(input_data)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Conv2D(1, (1,1), activation='relu', padding='same')(x)
  x = tf.keras.layers.BatchNormalization()(x)

  x = tf.keras.layers.Dense(1, activation='linear')(x)
  # x = tf.keras.layers.Reshape((num_target, height, width, 1))(x)
  cnn = tf.keras.Model(inputs=input_data, outputs=x)

  cnn.compile(optimizer='adam', loss='mse', metrics=['mse','mae'])
  return cnn

if __name__ == "__main__":
    # BASE_URL = f"/content/drive/My Drive/Climate Modeling/Processed data/cnn/normalize/"
    BASE_URL = f"/workspace/climate/data/timeseries"
    paths = glob.glob(join(BASE_URL,"*.pkl"), recursive=True)
    sorted(paths)

    # limit the number of paths
    paths = paths[:256]

    # train_paths, validation_paths = train_test_split(paths, test_size=0.2, random_state = seed)
    split_index = int(len(paths) * 0.8)
    train_paths = paths[:split_index]
    validation_paths = paths[split_index:]

    dataset = tf.data.Dataset.from_tensor_slices(train_paths)
    dataset = dataset.map(_function)
    dataset = dataset.batch(1)
    dataset = dataset.cache("./")

    val_dataset = tf.data.Dataset.from_tensor_slices(validation_paths)
    val_dataset = val_dataset.map(_function)
    val_dataset = val_dataset.batch(1)
    val_dataset = val_dataset.cache("./")

    cnn_lstm = cnn_lstm()
    cnn_lstm.fit(dataset, epochs=15, validation_data=val_dataset)
    cnn_lstm.save("model/cnn_batch_small_v1.h5")
