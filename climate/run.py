import pickle
import tensorflow as tf
import numpy as np
import glob
import pygrib
from sklearn.model_selection import train_test_split

from os.path import join
# BASE_URL = f"/content/drive/My Drive/Climate Modeling/Processed data/cnn/normalize/"
BASE_URL = f"./data"
paths = glob.glob(join(BASE_URL,"*.pkl"), recursive=True)

# increase number of sets
paths = paths[:16]
print(sorted(paths))

# seed = 10
# train_paths, validation_paths = train_test_split(paths, test_size=0.2, random_state = seed)
split_index = int(len(paths) * 0.8)
train_paths = paths[:split_index]
validation_paths = paths[split_index:]

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

x_path, y_path = multivariate_data(train_paths, train_paths, 0, len(train_paths), timestep, num_target, 1)

def proccess_gfs(file):
  path = file.numpy()
  gfs = pygrib.open(path)
  parameters = ['pressure in hpa', 'gh', 't', 'u', 'v', 'icmr', 'rwmr', 'snmr', 'grle', 'w', 'dlwrf','dswrf', 'uswrf', 'ulwrf']
  datas = np.array(list(map(lambda x: x.data()[0],gfs.select(shortName=parameters, typeOfLevel=['isobaricInhPa'])[:]))).reshape((361, 720, -1))
  temperature = np.array([]).reshape(361, 720, 0)
  solar = np.array([]).reshape(361, 720, 0)
  for g in gfs.select(shortName=parameters, typeOfLevel=['surface'])[:]:
    data = np.array(g.data()[0]).reshape((361,720,1))
    if g['name'] == "Temperature":
      temperature = np.concatenate([temperature, data], 2)
    else:
      solar = np.concatenate([solar, data], 2)
  datas = np.concatenate([datas, solar],2)
  return (datas, temperature)

def normalize(datas):
  x_min = datas.min(axis=(0,1),  keepdims=True)
  x_max = datas.max(axis=(0,1),  keepdims=True)
  x_norm = (datas - x_min) / (x_max - x_min)
  # there might be nan if all the values are 0
  # replace all nan to 0
  wherenan = np.isnan(x_norm)
  x_norm[wherenan] = 0
  return x_norm.astype("float32")

def _parse_file(file):
  x , y = proccess_gfs(file)
  x = normalize(x)
  return (x, y)

def _parse_pickle_file(x, y):
  x = x.numpy()
  y = y.numpy()
  x_data = []
  y_data = []
  for path in x:
    with open(path, 'rb') as f:
      data = pickle.load(f)
      # data = next(iter(data.values()))
      x_data.append(normalize(data["x"]))
  for path in y:
    with open(path, 'rb') as f:
      data = pickle.load(f)
      # data = next(iter(data.values()))
      y_data.append(data["y"])
  x_data = np.array(x_data)
  y_data = np.array(y_data)
  return x_data, y_data

def _function(x, y):
  x, y = tf.py_function(_parse_pickle_file, [x, y], [tf.float32, tf.float32])
  x.set_shape((timestep, 361, 720, 241))
  y.set_shape((num_target, 361, 720, 1))
  return x,y

dataset = tf.data.Dataset.from_tensor_slices((x_path,y_path))

# dataset = dataset.map(lambda x: tuple(tf.py_function(_parse_file, [x], [tf.float32, tf.float32])))
# dataset = dataset.map(_function, num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=False)
dataset = dataset.map(_function)
dataset = dataset.batch(1)
dataset = dataset.cache("./")
print(dataset.element_spec)

def cnn_lstm():
  # parameter
  TimeDistributed = tf.keras.layers.TimeDistributed
  input_data = tf.keras.layers.Input((timestep, height, width, num_parameters))
  x = TimeDistributed(tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding='same'))(input_data)
  x = TimeDistributed(tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'))(x)
  x = TimeDistributed(tf.keras.layers.Conv2D(1, (3,3), activation='relu', padding='same'))(x)

  x = tf.keras.layers.ConvLSTM2D(128, (3,3), activation='relu', padding='same', return_sequences=True)(x)
  x = tf.keras.layers.ConvLSTM2D(128, (3,3), activation='relu', padding='same')(x)
  # x = tf.keras.layers.Conv3D(1, (3,3,3), activation='linear', padding='same')(x)
  x = tf.keras.layers.Dense(num_target, activation='linear')(x)
  x = tf.keras.layers.Reshape((num_target, height, width, 1))(x)
  cnn = tf.keras.Model(inputs=input_data, outputs=x)

  cnn.compile(optimizer='adam', loss='mse', metrics=['mse','mae'])
  return cnn

cnn_lstm = cnn_lstm()
cnn_lstm.summary()
cnn_lstm.fit(dataset, epochs=10)
cnn_lstm.save("./model/cnn_lstm_timedist.h5")