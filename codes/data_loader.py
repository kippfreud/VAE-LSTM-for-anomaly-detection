from base import BaseDataGenerator
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pyplot import savefig
from boto3_tools import create_dataset, get_dataset, update_dataset, lookup_iot_dataset_by_name, \
  get_all_in_use_measuring_points, get_full_sql_query

# ------------------------------------------------------------------------------

DATASET_NAME = "test_temp"
schedule = None
all_mps, mapping = get_all_in_use_measuring_points(return_mapping_dict=True)
sql_items = [
    'tstamp',
    #"CAST(tstamp as float) tstamp",
    "sensor_id",
    "measure_id",
    "sTime as time",
    "rmsX as x",
    "rmsY as y",
    "rmsZ as z",
    "bat as bat",
    "temp as temp",
    "sound_rms as sound",
    "status as status"
]
datastore = "m00xx_datastore"
short_sql = f"SELECT {','.join(sql_items)} FROM {datastore}"
#interval_str = "tstamp BETWEEN To_unixtime(current_timestamp - interval '30' day) * 1000 AND To_unixtime(current_timestamp) * 1000"
interval_str = None #..todo: fix, should be able to handle this
sql_query = get_full_sql_query(short_sql, all_mps, interval_str=interval_str, filter_key="measure_id",
                               order_by_key="tstamp")
if lookup_iot_dataset_by_name(DATASET_NAME):
    print(f"Updating {DATASET_NAME}.")
    update_dataset(DATASET_NAME, sql_query, schedule)
else:
    print(f"Creating {DATASET_NAME}.")
    create_dataset(DATASET_NAME, sql_query, schedule)
# Load the data
full_df = get_dataset(DATASET_NAME)

# ------------------------------------------------------------------------------

class DataGenerator(BaseDataGenerator):
  def __init__(self, config):
    super(DataGenerator, self).__init__(config)
    # load data here: generate 3 state variables: train_set, val_set and test_set
    self.load_NAB_dataset(self.config['dataset'], self.config['y_scale'])

  def load_NAB_dataset(self, dataset, y_scale=6):
    data_dir = '../datasets/NAB-known-anomaly/'
    data = np.load(data_dir + dataset + '.npz')

    # normalise the dataset by training set mean and std
    train_m = data['train_m']
    train_std = data['train_std']
    readings_normalised = (data['readings'] - train_m) / train_std

    # plot normalised data
    fig, axs = plt.subplots(1, 1, figsize=(18, 4), edgecolor='k')
    fig.subplots_adjust(hspace=.4, wspace=.4)
    axs.plot(data['t'], readings_normalised)
    if data['idx_split'][0] == 0:
      axs.plot(data['idx_split'][1] * np.ones(20), np.linspace(-y_scale, y_scale, 20), 'b-')
    else:
      for i in range(2):
        axs.plot(data['idx_split'][i] * np.ones(20), np.linspace(-y_scale, y_scale, 20), 'b-')
    axs.plot(*np.ones(20), np.linspace(-y_scale, y_scale, 20), 'b--')
    for j in range(len(data['idx_anomaly'])):
      axs.plot(data['idx_anomaly'][j] * np.ones(20), np.linspace(-y_scale, 0.75 * y_scale, 20), 'r--')
    axs.grid(True)
    axs.set_xlim(0, len(data['t']))
    axs.set_ylim(-y_scale, y_scale)
    axs.set_xlabel("timestamp (every {})".format(data['t_unit']))
    axs.set_ylabel("readings")
    axs.set_title("{} dataset\n(normalised by train mean {:.4f} and std {:.4f})".format(dataset, train_m, train_std))
    axs.legend(('data', 'train test set split', 'anomalies'))
    savefig(self.config['result_dir'] + '/raw_data_normalised.pdf')

    # slice training set into rolling windows
    n_train_sample = len(data['training'])
    n_train_vae = n_train_sample - self.config['l_win'] + 1
    rolling_windows = np.zeros((n_train_vae, self.config['l_win']))
    for i in range(n_train_sample - self.config['l_win'] + 1):
      rolling_windows[i] = data['training'][i:i + self.config['l_win']]

    # create VAE training and validation set
    idx_train, idx_val, self.n_train_vae, self.n_val_vae = self.separate_train_and_val_set(n_train_vae)
    self.train_set_vae = dict(data=np.expand_dims(rolling_windows[idx_train], -1))
    self.val_set_vae = dict(data=np.expand_dims(rolling_windows[idx_val], -1))
    self.test_set_vae = dict(data=np.expand_dims(rolling_windows[idx_val[:self.config['batch_size']]], -1))

    # create LSTM training and validation set
    for k in range(self.config['l_win']):
      n_not_overlap_wins = (n_train_sample - k) // self.config['l_win']
      n_train_lstm = n_not_overlap_wins - self.config['l_seq'] + 1
      cur_lstm_seq = np.zeros((n_train_lstm, self.config['l_seq'], self.config['l_win']))
      for i in range(n_train_lstm):
        cur_seq = np.zeros((self.config['l_seq'], self.config['l_win']))
        for j in range(self.config['l_seq']):
          # print(k,i,j)
          cur_seq[j] = data['training'][k + self.config['l_win'] * (j + i): k + self.config['l_win'] * (j + i + 1)]
        cur_lstm_seq[i] = cur_seq
      if k == 0:
        lstm_seq = cur_lstm_seq
      else:
        lstm_seq = np.concatenate((lstm_seq, cur_lstm_seq), axis=0)

    n_train_lstm = lstm_seq.shape[0]
    idx_train, idx_val, self.n_train_lstm, self.n_val_lstm = self.separate_train_and_val_set(n_train_lstm)
    self.train_set_lstm = dict(data=np.expand_dims(lstm_seq[idx_train], -1))
    self.val_set_lstm = dict(data=np.expand_dims(lstm_seq[idx_val], -1))
    self.data = data
    return

  def plot_time_series(self, data, time, data_list):
    fig, axs = plt.subplots(1, 4, figsize=(18, 2.5), edgecolor='k')
    fig.subplots_adjust(hspace=.8, wspace=.4)
    axs = axs.ravel()
    for i in range(4):
      axs[i].plot(time / 60., data[:, i])
      axs[i].set_title(data_list[i])
      axs[i].set_xlabel('time (h)')
      axs[i].set_xlim((np.amin(time) / 60., np.amax(time) / 60.))
    savefig(self.config['result_dir'] + '/raw_training_set_normalised.pdf')


class SoralinkDataGenerator(BaseDataGenerator):
  def __init__(self, config):
    super(SoralinkDataGenerator, self).__init__(config)
    # load data here: generate 3 state variables: train_set, val_set and test_set
    self.load_NAB_dataset(self.config['dataset'], self.config['y_scale'])

  def load_NAB_dataset(self, dataset, y_scale=6):
    # data_dir = '../datasets/NAB-known-anomaly/'
    # data = np.load(data_dir + dataset + '.npz')
    FEATURE = "sound"
    df = full_df.loc[full_df["measure_id"] == dataset]
    df["vib"] = df[["x","y","z"]].mean(axis=1)
    train_m = df[FEATURE].mean()
    train_std = df[FEATURE].std()
    readings_normalised = (df[FEATURE] - train_m) / train_std
    data = {"readings": df[FEATURE],
            "t": df["tstamp"],
            "train_m": train_m,
            "train_std": train_std,
            "t_unit": "1 ms"}
    data["idx_split"] = np.array([int(len(df[FEATURE]) / 3), 2 * int(len(df[FEATURE]) / 3)])
    data['idx_anomaly'] = []
    data['idx_anomaly_test'] = []
    data['idx_anomaly_val'] = []
    data["val"] = data["readings"][0:data["idx_split"][0]]
    data["t_val"] = df["tstamp"][0:data["idx_split"][0]]

    data["test"] = data["readings"][data["idx_split"][0]:data["idx_split"][1]]
    data["t_test"] = df["tstamp"][data["idx_split"][0]:data["idx_split"][1]]

    data["training"] = data["readings"][data["idx_split"][1]:]
    data["t_train"] = df["tstamp"][data["idx_split"][1]:]
    #### ----

    # plot normalised data
    fig, axs = plt.subplots(1, 1, figsize=(18, 4), edgecolor='k')
    fig.subplots_adjust(hspace=.4, wspace=.4)
    axs.plot(data['t'], readings_normalised)
    if data['idx_split'][0] == 0:
      axs.plot(data['t'].iloc[data['idx_split'][1]] * np.ones(20), np.linspace(-y_scale, y_scale, 20), 'b-')
    else:
      for i in range(2):
        axs.plot(data['t'].iloc[data['idx_split'][i]] * np.ones(20), np.linspace(-y_scale, y_scale, 20), 'b-')
    axs.plot(min(data['t'])*np.ones(20), np.linspace(-y_scale, y_scale, 20), 'b--')
    for j in range(len(data['idx_anomaly'])):
      axs.plot(data['idx_anomaly'][j] * np.ones(20), np.linspace(-y_scale, 0.75 * y_scale, 20), 'r--')
    axs.grid(True)
    axs.set_xlim( min(data['t']), max(data['t']))
    axs.set_ylim(-y_scale, y_scale)
    axs.set_xlabel("timestamp (every {})".format(data['t_unit']))
    axs.set_ylabel("readings")
    axs.set_title("{} dataset\n(normalised by train mean {:.4f} and std {:.4f})".format(dataset, train_m, train_std))
    axs.legend(('data', 'train test set split', 'anomalies'))
    savefig(self.config['result_dir'] + '/raw_data_normalised.pdf')

    # slice training set into rolling windows
    n_train_sample = len(data['training'])
    n_train_vae = n_train_sample - self.config['l_win'] + 1
    rolling_windows = np.zeros((n_train_vae, self.config['l_win']))
    for i in range(n_train_sample - self.config['l_win'] + 1):
      rolling_windows[i] = data['training'][i:i + self.config['l_win']]

    # create VAE training and validation set
    idx_train, idx_val, self.n_train_vae, self.n_val_vae = self.separate_train_and_val_set(n_train_vae)
    self.train_set_vae = dict(data=np.expand_dims(rolling_windows[idx_train], -1))
    self.val_set_vae = dict(data=np.expand_dims(rolling_windows[idx_val], -1))
    self.test_set_vae = dict(data=np.expand_dims(rolling_windows[idx_val[:self.config['batch_size']]], -1))

    # create LSTM training and validation set
    for k in range(self.config['l_win']):
      n_not_overlap_wins = (n_train_sample - k) // self.config['l_win']
      n_train_lstm = n_not_overlap_wins - self.config['l_seq'] + 1
      cur_lstm_seq = np.zeros((n_train_lstm, self.config['l_seq'], self.config['l_win']))
      for i in range(n_train_lstm):
        cur_seq = np.zeros((self.config['l_seq'], self.config['l_win']))
        for j in range(self.config['l_seq']):
          # print(k,i,j)
          cur_seq[j] = data['training'][k + self.config['l_win'] * (j + i): k + self.config['l_win'] * (j + i + 1)]
        cur_lstm_seq[i] = cur_seq
      if k == 0:
        lstm_seq = cur_lstm_seq
      else:
        lstm_seq = np.concatenate((lstm_seq, cur_lstm_seq), axis=0)

    n_train_lstm = lstm_seq.shape[0]
    idx_train, idx_val, self.n_train_lstm, self.n_val_lstm = self.separate_train_and_val_set(n_train_lstm)
    self.train_set_lstm = dict(data=np.expand_dims(lstm_seq[idx_train], -1))
    self.val_set_lstm = dict(data=np.expand_dims(lstm_seq[idx_val], -1))
    self.data = data
    return

  def plot_time_series(self, data, time, data_list):
    fig, axs = plt.subplots(1, 4, figsize=(18, 2.5), edgecolor='k')
    fig.subplots_adjust(hspace=.8, wspace=.4)
    axs = axs.ravel()
    for i in range(4):
      axs[i].plot(time / 60., data[:, i])
      axs[i].set_title(data_list[i])
      axs[i].set_xlabel('time (h)')
      axs[i].set_xlim((np.amin(time) / 60., np.amax(time) / 60.))
    savefig(self.config['result_dir'] + '/raw_training_set_normalised.pdf')

