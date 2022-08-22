from base import BaseDataGenerator
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pyplot import savefig
from boto3_tools import create_dataset, get_dataset, update_dataset, lookup_iot_dataset_by_name, \
  get_all_in_use_measuring_points, get_full_sql_query

# ------------------------------------------------------------------------------

DATASET_NAME = "test_temp"
schedule = None

#all_mps, mapping = get_all_in_use_measuring_points(return_mapping_dict=True)
all_mps = ['E100000M01', 'E100001M02', 'E100001M03', 'E100002M02', 'E100002M03', 'E100003M01', 'E100005M01', 'E100005M02', 'E100006M01', 'E100006M02', 'E100008M01', 'E100008M02', 'E100009M01', 'E100010M02', 'E100012M01', 'E100013M01', 'E100014M01', 'E100015M01', 'E100016M01', 'E100017M01', 'E100018M01', 'E100019M01', 'E100020M01', 'E100021M01', 'E100022M01', 'E100023M01', 'E100024M01', 'E100025M01', 'E100026M01', 'E100027M01', 'E100028M01', 'E100029M03', 'E100030M02', 'E100031M03', 'E100033M01', 'E100034M01', 'E100035M01', 'E100036M01', 'E100037M01']
#all_mps = ['E100008M01', 'E100008M02', 'E100009M01', 'E100010M02', 'E100012M01', 'E100013M01', 'E100014M01', 'E100015M01', 'E100016M01', 'E100017M01', 'E100018M01', 'E100019M01', 'E100020M01', 'E100021M01', 'E100022M01', 'E100023M01', 'E100024M01', 'E100025M01', 'E100026M01', 'E100027M01', 'E100028M01', 'E100029M03', 'E100030M02', 'E100031M03', 'E100033M01', 'E100034M01', 'E100035M01', 'E100036M01', 'E100037M01']
mapping = {'E100000M01': {'serial_number': 's00000026', 'measure_point_id': 'E100000M01', 'equipment_serial_number': 'E100000', 'equipment_name': 'Compresseur 3', 'location': 'ON_TOP', 'threshold_id': 1, 'name': 'Sherbrook', 'vibration_on': ('0.317333333'), 'vibration_off': ('0.105666667'), 'vib_avg_history': ('0.439083173629743')},
           'E100015M01': {'serial_number': 's00000061', 'measure_point_id': 'E100015M01', 'equipment_serial_number': 'E100015', 'equipment_name': 'Agitateur Reservoir Premix 5', 'location': 'ON_GEARBOX', 'threshold_id': 18, 'name': 'Lassonde', 'vibration_on': ('0.2'), 'vibration_off': ('0'), 'vib_avg_history': ('0.0172744893976844')},
           'E100014M01': {'serial_number': 's00000062', 'measure_point_id': 'E100014M01', 'equipment_serial_number': 'E100014', 'equipment_name': 'Agitateur Reservoir Premix 6', 'location': 'ON_GEARBOX', 'threshold_id': 17, 'name': 'Lassonde', 'vibration_on': ('0.046'), 'vibration_off': ('0.041'), 'vib_avg_history': ('0.013157241395601543')},
           'E100029M03': {'serial_number': 's00000060', 'measure_point_id': 'E100029M03', 'equipment_serial_number': 'E100029', 'equipment_name': 'Pompe Glycol 0137 (au fond)', 'location': 'ON_PUMP_SHAFT', 'threshold_id': 33, 'name': 'Lassonde', 'vibration_on': ('0.1'), 'vibration_off': ('0.02'), 'vib_avg_history': ('0.1001924646707717')},
           'E100002M03': {'serial_number': 's00000009', 'measure_point_id': 'E100002M03', 'equipment_serial_number': 'E100002', 'equipment_name': 'Compresseur 3', 'location': 'ON_OUTLET', 'threshold_id': 5, 'name': 'Olymel', 'vibration_on': ('0.253'), 'vibration_off': ('0.069333333'), 'vib_avg_history': ('0.3397385719555958')},
           'E100003M01': {'serial_number': 's00000008', 'measure_point_id': 'E100003M01', 'equipment_serial_number': 'E100003', 'equipment_name': 'Pompe Glycol', 'location': 'ON_TOP', 'threshold_id': 6, 'name': 'Olymel', 'vibration_on': ('0.046'), 'vibration_off': ('0.044'), 'vib_avg_history': ('0.07483400375435774')},
           'E100005M01': {'serial_number': 's00000025', 'measure_point_id': 'E100005M01', 'equipment_serial_number': 'E100005', 'equipment_name': 'Compresseur 2', 'location': 'ON_SCREW', 'threshold_id': 7, 'name': 'Olymel', 'vibration_on': ('0.128'), 'vibration_off': ('0.048333333'), 'vib_avg_history': ('0.22773712600931417')},
           'E100005M02': {'serial_number': 's00000004', 'measure_point_id': 'E100005M02', 'equipment_serial_number': 'E100005', 'equipment_name': 'Compresseur 2', 'location': 'ON_OUTLET', 'threshold_id': 8, 'name': 'Olymel', 'vibration_on': ('0.285'), 'vibration_off': ('0.05866666666666667'), 'vib_avg_history': ('0.3430932293749167')},
           'E100006M01': {'serial_number': 's00000027', 'measure_point_id': 'E100006M01', 'equipment_serial_number': 'E100006', 'equipment_name': 'Compresseur 1', 'location': 'ON_OUTLET', 'threshold_id': 9, 'name': 'Olymel', 'vibration_on': ('0.22766666666666668'), 'vibration_off': ('-1'), 'vib_avg_history': ('0.3713920862109232')},
           'E100006M02': {'serial_number': 's00000020', 'measure_point_id': 'E100006M02', 'equipment_serial_number': 'E100006', 'equipment_name': 'Compresseur 1', 'location': 'ON_SCREW', 'threshold_id': 10, 'name': 'Olymel', 'vibration_on': ('0.076333333'), 'vibration_off': ('-1'), 'vib_avg_history': ('0.1595705660629107')},
           'E100030M02': {'serial_number': 's00000063', 'measure_point_id': 'E100030M02', 'equipment_serial_number': 'E100030', 'equipment_name': 'Agitateur Reservoir A6', 'location': 'ON_GEARBOX', 'threshold_id': 34, 'name': 'Lassonde', 'vibration_on': ('0.1'), 'vibration_off': ('0.02'), 'vib_avg_history': ('0.0822738155063332')},
           'E100009M01': {'serial_number': 's00000007', 'measure_point_id': 'E100009M01', 'equipment_serial_number': 'E100009', 'equipment_name': 'Booster 1 500psi', 'location': 'ON_MOTOR', 'threshold_id': 13, 'name': 'Saputo', 'vibration_on': ('0.18299999999999997'), 'vibration_off': ('0.02766666666666667'), 'vib_avg_history': ('0.10585395841663332')},
           'E100033M01': {'serial_number': 's00000068', 'measure_point_id': 'E100033M01', 'equipment_serial_number': 'E100033', 'equipment_name': 'Ventilateur chaudière 4', 'location': 'ON_MOTOR', 'threshold_id': 39, 'name': 'Saputo', 'vibration_on': ('0.1'), 'vibration_off': ('0.02'), 'vib_avg_history': ('0.2214482599464599')},
           'E100031M03': {'serial_number': 's00000059', 'measure_point_id': 'E100031M03', 'equipment_serial_number': 'E100031', 'equipment_name': 'Pompe Glycol 0136 (pompe 2)', 'location': 'ON_PUMP_SHAFT', 'threshold_id': 35, 'name': 'Lassonde', 'vibration_on': ('0.1'), 'vibration_off': ('0.02'), 'vib_avg_history': ('0.23222857109119466')},
           'E100013M01': {'serial_number': 's00000015', 'measure_point_id': 'E100013M01', 'equipment_serial_number': 'E100013', 'equipment_name': 'Pompe Glycol 0135', 'location': 'ON_PUMP_SHAFT', 'threshold_id': 16, 'name': 'Lassonde', 'vibration_on': ('0.283666667'), 'vibration_off': ('-1'), 'vib_avg_history': ('0.3115614117533649')},
           'E100016M01': {'serial_number': 's00000030', 'measure_point_id': 'E100016M01', 'equipment_serial_number': 'E100016', 'equipment_name': 'Agitateur Reservoir B4', 'location': 'ON_GEARBOX', 'threshold_id': 19, 'name': 'Lassonde', 'vibration_on': ('0.041'), 'vibration_off': ('0.039'), 'vib_avg_history': ('0.03356570943725427')},
           'E100017M01': {'serial_number': 's00000013', 'measure_point_id': 'E100017M01', 'equipment_serial_number': 'E100017', 'equipment_name': 'Agitateur Reservoir B5', 'location': 'ON_GEARBOX', 'threshold_id': 20, 'name': 'Lassonde', 'vibration_on': ('0.104439669'), 'vibration_off': ('0.008034023'), 'vib_avg_history': ('0.02379379269800726')},
           'E100018M01': {'serial_number': 's00000012', 'measure_point_id': 'E100018M01', 'equipment_serial_number': 'E100018', 'equipment_name': 'Compresseur 150HP', 'location': 'ON_MOTOR', 'threshold_id': 21, 'name': 'Saputo', 'vibration_on': ('0.195333333'), 'vibration_off': ('0.014'), 'vib_avg_history': ('0.10455889199255121')},
           'E100019M01': {'serial_number': 's00000010', 'measure_point_id': 'E100019M01', 'equipment_serial_number': 'E100019', 'equipment_name': 'Pompe degazage (condense) 1', 'location': 'ON_MOTOR', 'threshold_id': 22, 'name': 'Saputo', 'vibration_on': ('0.3126666666666667'), 'vibration_off': ('0.13'), 'vib_avg_history': ('0.18959779971416227')},
           'E100001M02': {'serial_number': 's00000040', 'measure_point_id': 'E100001M02', 'equipment_serial_number': 'E100001', 'equipment_name': 'Compresseur 4', 'location': 'ON_SCREW', 'threshold_id': 2, 'name': 'Olymel', 'vibration_on': ('0.203333333'), 'vibration_off': ('0.105666667'), 'vib_avg_history': ('0.22415395904209193')},
           'E100001M03': {'serial_number': 's00000041', 'measure_point_id': 'E100001M03', 'equipment_serial_number': 'E100001', 'equipment_name': 'Compresseur 4', 'location': 'ON_OUTLET', 'threshold_id': 3, 'name': 'Olymel', 'vibration_on': ('0.36220081'), 'vibration_off': ('0.026977778'), 'vib_avg_history': ('0.28017160025948307')},
           'E100002M02': {'serial_number': 's00000042', 'measure_point_id': 'E100002M02', 'equipment_serial_number': 'E100002', 'equipment_name': 'Compresseur 3', 'location': 'ON_SCREW', 'threshold_id': 4, 'name': 'Olymel', 'vibration_on': ('0.297104052'), 'vibration_off': ('0.009214833'), 'vib_avg_history': ('0.38399413109496017')},
           'E100008M01': {'serial_number': 's00000043', 'measure_point_id': 'E100008M01', 'equipment_serial_number': 'E100008', 'equipment_name': 'Compresseur KAESER', 'location': 'ON_SCREW_1', 'threshold_id': 11, 'name': 'Olymel', 'vibration_on': ('0.129031267'), 'vibration_off': ('0.007994949'), 'vib_avg_history': ('0.22660912698412697')},
           'E100008M02': {'serial_number': 's00000044', 'measure_point_id': 'E100008M02', 'equipment_serial_number': 'E100008', 'equipment_name': 'Compresseur KAESER', 'location': 'ON_SCREW_2', 'threshold_id': 12, 'name': 'Olymel', 'vibration_on': ('0.153877332'), 'vibration_off': ('0.006713086'), 'vib_avg_history': ('0.12032116160613877')},
           'E100021M01': {'serial_number': 's00000049', 'measure_point_id': 'E100021M01', 'equipment_serial_number': 'E100021', 'equipment_name': "Pompe d'eau condenseur MP94033", 'location': 'ON_IMPELLER_SHAFT', 'threshold_id': 25, 'name': 'Saputo', 'vibration_on': ('0.1'), 'vibration_off': ('0.02'), 'vib_avg_history': ('0.14518242303872889')},
           'E100022M01': {'serial_number': 's00000050', 'measure_point_id': 'E100022M01', 'equipment_serial_number': 'E100022', 'equipment_name': "Pompe d'eau condenseur MP94034", 'location': 'ON_IMPELLER_SHAFT', 'threshold_id': 26, 'name': 'Saputo', 'vibration_on': ('0.1'), 'vibration_off': ('0.02'), 'vib_avg_history': ('0.11069606382626544')},
           'E100023M01': {'serial_number': 's00000051', 'measure_point_id': 'E100023M01', 'equipment_serial_number': 'E100023', 'equipment_name': "Pompe d'eau condenseur MP94035", 'location': 'ON_IMPELLER_SHAFT', 'threshold_id': 27, 'name': 'Saputo', 'vibration_on': ('0.1'), 'vibration_off': ('0.02'), 'vib_avg_history': ('0.08550705765407556')},
           'E100024M01': {'serial_number': 's00000052', 'measure_point_id': 'E100024M01', 'equipment_serial_number': 'E100024', 'equipment_name': "Pompe d'eau condenseur MP94036", 'location': 'ON_IMPELLER_SHAFT', 'threshold_id': 28, 'name': 'Saputo', 'vibration_on': ('0.1'), 'vibration_off': ('0.02'), 'vib_avg_history': ('0.09693865396069089')},
           'E100025M01': {'serial_number': 's00000054', 'measure_point_id': 'E100025M01', 'equipment_serial_number': 'E100025', 'equipment_name': 'Pompe Glycol MP93041', 'location': 'ON_IMPELLER_SHAFT', 'threshold_id': 29, 'name': 'Saputo', 'vibration_on': ('0.1'), 'vibration_off': ('0.02'), 'vib_avg_history': ('0.2487777336510458')},
           'E100026M01': {'serial_number': 's00000055', 'measure_point_id': 'E100026M01', 'equipment_serial_number': 'E100026', 'equipment_name': 'Pompe Glycol MP93042', 'location': 'ON_IMPELLER_SHAFT', 'threshold_id': 30, 'name': 'Saputo', 'vibration_on': ('0.1'), 'vibration_off': ('0.02'), 'vib_avg_history': ('0.02826289633237253')},
           'E100027M01': {'serial_number': 's00000056', 'measure_point_id': 'E100027M01', 'equipment_serial_number': 'E100027', 'equipment_name': 'Pompe Glycol MP93043', 'location': 'ON_IMPELLER_SHAFT', 'threshold_id': 31, 'name': 'Saputo', 'vibration_on': ('0.1'), 'vibration_off': ('0.02'), 'vib_avg_history': ('0.1444159796033244')},
           'E100028M01': {'serial_number': 's00000057', 'measure_point_id': 'E100028M01', 'equipment_serial_number': 'E100028', 'equipment_name': 'Pompe Glycol MP93044', 'location': 'ON_IMPELLER_SHAFT', 'threshold_id': 32, 'name': 'Saputo', 'vibration_on': ('0.1'), 'vibration_off': ('0.02'), 'vib_avg_history': ('0.14842667990731548')},
           'E100012M01': {'serial_number': 's00000058', 'measure_point_id': 'E100012M01', 'equipment_serial_number': 'E100012', 'equipment_name': 'Pompe Glycol 0134', 'location': 'ON_PUMP_SHAFT', 'threshold_id': 15, 'name': 'Lassonde', 'vibration_on': ('0.369'), 'vibration_off': ('-1'), 'vib_avg_history': ('0.40350803163008403')},
           'E100010M02': {'serial_number': 's00000065', 'measure_point_id': 'E100010M02', 'equipment_serial_number': 'E100010', 'equipment_name': 'Ventilateur chaudière 3', 'location': 'ON_MOTOR', 'threshold_id': 14, 'name': 'Saputo', 'vibration_on': ('0.12233333333333334'), 'vibration_off': ('0.02266666666666667'), 'vib_avg_history': ('0.14868364197530864')},
           'E100034M01': {'serial_number': 's00000069', 'measure_point_id': 'E100034M01', 'equipment_serial_number': 'E100034', 'equipment_name': 'Ventilateur chaudière 5', 'location': 'ON_MOTOR', 'threshold_id': 40, 'name': 'Saputo', 'vibration_on': ('0.1'), 'vibration_off': ('0.02'), 'vib_avg_history': ('0.1572806238141515')},
           'E100036M01': {'serial_number': 's00000067', 'measure_point_id': 'E100036M01', 'equipment_serial_number': 'E100036', 'equipment_name': "Pompe recirculation tour d'eau 1", 'location': 'ON_IMPELLER_SHAFT', 'threshold_id': 42, 'name': 'Saputo', 'vibration_on': ('0.1'), 'vibration_off': ('0.02'), 'vib_avg_history': ('0.03654473684210527')},
           'E100020M01': {'serial_number': 's00000137', 'measure_point_id': 'E100020M01', 'equipment_serial_number': 'E100020', 'equipment_name': 'Infuseur', 'location': 'ON_MOTOR', 'threshold_id': 23, 'name': 'Saputo', 'vibration_on': ('0.128950834'), 'vibration_off': ('0.00727537'), 'vib_avg_history': ('0.11821020613373556')},
           'E100035M01': {'serial_number': 's00000070', 'measure_point_id': 'E100035M01', 'equipment_serial_number': 'E100035', 'equipment_name': 'Pompe degazage (condense) 2', 'location': 'ON_MOTOR', 'threshold_id': 41, 'name': 'Saputo', 'vibration_on': ('0.1'), 'vibration_off': ('0.02'), 'vib_avg_history': ('0.22410497547302036')},
           'E100037M01': {'serial_number': 's00000138', 'measure_point_id': 'E100037M01', 'equipment_serial_number': 'E100037', 'equipment_name': "Pompe recirculation tour d'eau 2", 'location': 'ON_IMPELLER_SHAFT', 'threshold_id': 43, 'name': 'Saputo', 'vibration_on': ('0.1'), 'vibration_off': ('0.02'), 'vib_avg_history': ('0.289124618796784')}}



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

