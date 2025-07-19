import numpy as np
from sklearn import preprocessing
import warnings
import warnings
from sklearn.model_selection import train_test_split
from scipy.signal import resample

warnings.filterwarnings("ignore", category=UserWarning, module='sklearn.preprocessing._data')
def prepare_data(X_train_path, y_train_path, train_per=0.8):

    X_train = np.load(X_train_path)
    # max_pos_flat = np.argmax(X_train)  # 返回展平后的索引（一维）
    # max_pos = np.unravel_index(max_pos_flat, X_train.shape)  # 转换为多维索引
    #
    # print("最大值的位置（多维索引）:", max_pos)
    # print("数据范围:", np.min(X_train), np.max(X_train))
    # print("均值:", np.mean(X_train, axis=0))
    # print("标准差:", np.std(X_train, axis=0))
    y_train = np.load(y_train_path)

    X_train = X_train.reshape(15162, 400)
    y_train = y_train.reshape(15162, 400)

    # 对时间维度进行插值，从400变为512
    X_train = resample(X_train, 512, axis=-1)
    y_train = resample(y_train, 512, axis=-1)

    eeg_all = []
    sn_all = []
    noise_all = []

    for i in range(X_train.shape[0]):

        eeg = y_train[i]
        eeg = preprocessing.scale(eeg)

        eog = X_train[i] - y_train[i]
        eog = preprocessing.scale(eog)

        signal_noise = eeg + eog
        # noise = X_train[i]
        # noise = preprocessing.scale(noise)
        noise_all.append(eog)
        eeg_all.append(eeg)
        sn_all.append(signal_noise)

    noise_all = np.array(noise_all)
    eeg_all = np.array(eeg_all)
    sn_all = np.array(sn_all)
    print(eeg_all.shape)

    eeg_train, eeg_temp, sn_train, sn_temp, noise_train, noise_temp = train_test_split(
        eeg_all, sn_all, noise_all, train_size=train_per, random_state=666, shuffle=True
    )

    eeg_val, eeg_test, sn_val, sn_test, noise_val, noise_test = train_test_split(
        eeg_temp, sn_temp, noise_temp, test_size=0.5, random_state=666, shuffle=True
    )

    print("Train shape:", eeg_train.shape)
    print("Val shape:", eeg_val.shape)
    print("Test shape:", eeg_test.shape)
    '''
    (15162, 400)
    Train shape: (12129, 400)
    Val shape: (1516, 400)
    Test shape: (1517, 400)
    '''

    label_train = np.zeros(eeg_train.shape[0])
    label_val = np.zeros(eeg_val.shape[0])
    label_test = np.zeros(eeg_test.shape[0])

    X_train = np.expand_dims(sn_train, axis=1)
    y_train = np.expand_dims(eeg_train, axis=1)
    y_train_noise = np.expand_dims(noise_train, axis=1)

    X_val = np.expand_dims(sn_val, axis=1)
    y_val = np.expand_dims(eeg_val, axis=1)
    y_val_noise = np.expand_dims(noise_val, axis=1)

    X_test = np.expand_dims(sn_test, axis=1)
    y_test = np.expand_dims(eeg_test, axis=1)
    y_test_noise = np.expand_dims(noise_test, axis=1)

    np.save('../data-ssed-512/data_for_train/X_train.npy', X_train)
    np.save('../data-ssed-512/data_for_train/y_train.npy', y_train)
    np.save('../data-ssed-512/data_for_train/y_train_noise.npy', y_train_noise)
    np.save('../data-ssed-512/data_for_train/label_train.npy', label_train)

    np.save('../data-ssed-512/data_for_val/X_val.npy', X_val)
    np.save('../data-ssed-512/data_for_val/y_val.npy', y_val)
    np.save('../data-ssed-512/data_for_val/y_val_noise.npy', y_val_noise)
    np.save('../data-ssed-512/data_for_val/label_val.npy', label_val)

    np.save('../data-ssed-512/data_for_test/X_test.npy', X_test)
    np.save('../data-ssed-512/data_for_test/y_test.npy', y_test)
    np.save('../data-ssed-512/data_for_test/y_test_noise.npy', y_test_noise)
    np.save('../data-ssed-512/data_for_test/label_test.npy', label_test)

    dataset = [eeg_train, noise_train]

    return dataset

if __name__ == '__main__':
    X_train_path = './data/ssed_noise.npy'
    y_train_path = './data/ssed_eeg.npy'
    [X_train, y_train] = prepare_data(X_train_path, y_train_path, train_per=0.8)