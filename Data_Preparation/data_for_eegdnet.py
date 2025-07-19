import numpy as np
from sklearn import preprocessing

def get_rms(records, multi_channels):
    """
    The root mean square value reflects the effective value rather than the average value
    """
    if multi_channels == 1:
        n = records.shape[0]
        rms = 0
        for i in range(n):
            rms_t = np.sum([records[i]**2]) / len(records[i])
            rms = rms + rms_t
        return rms / n

    if multi_channels == 0:
        rms = np.sum([records**2]) / len(records)
        return rms


def get_SNR(signal, noisy):
    snr = 10 * np.log10(signal / noisy)
    return snr

def random_signal(signal, combin_num):
    random_result = []
    # combin_num:
    for i in range(combin_num):
        random_num = np.random.permutation(signal.shape[0])
        shuffled_dataset = signal[random_num, :]
        shuffled_dataset = shuffled_dataset.reshape(signal.shape[0], signal.shape[1])
        random_result.append(shuffled_dataset)

    random_result = np.array(random_result)

    return random_result


def prepare_data(combin_num, train_per, noise_type):

    file_location = './data/'  ############ change it to your own location #########
    if noise_type == 'd4pm':
        EEG_all = np.load(file_location + 'EEG_all_epochs.npy')# 4514
        EOG_all = np.load(file_location + 'EOG_all_epochs.npy')# 3400
        EMG_all = np.load(file_location + 'EMG_all_epochs.npy')# 5598
        ECG_all = np.load(file_location + 'ECG_all_epochs.npy')# 3600

    EEG_all_random = np.squeeze(random_signal(signal=EEG_all, combin_num=1))
    EOG_all_random = np.squeeze(random_signal(signal=EOG_all, combin_num=1))
    EMG_all_random = np.squeeze(random_signal(signal=EMG_all, combin_num=1))
    ECG_all_random = np.squeeze(random_signal(signal=ECG_all, combin_num=1))
    # noise_all_random = np.squeeze(random_signal(signal=EOG_all, combin_num=1))

    if noise_type == 'd4pm':
        EEGforEOG_all_random = EEG_all_random[0:EOG_all_random.shape[0]]
        print('EEG segments for EOG after drop: ', EEGforEOG_all_random.shape[0])

        reuse_num = EMG_all_random.shape[0] - EEG_all_random.shape[0]
        EEGforEMG_reuse = EEG_all_random[0: reuse_num, :]
        EEGforEMG_all_random= np.vstack([EEGforEMG_reuse, EEG_all_random])
        print('EEG segments for EMG after reuse: ', EEGforEMG_all_random.shape[0])

        EEGforECG_all_random = EEG_all_random[0:ECG_all_random.shape[0]]
        print('EEG segments for ECG after drop: ', EEGforECG_all_random.shape[0])


    # Number of data partitions
    timepoint = EEGforEOG_all_random.shape[1]

    train_numforEOG = round(train_per * EEGforEOG_all_random.shape[0])
    train_numforEMG = round(train_per * EEGforEMG_all_random.shape[0])
    train_numforECG = round(train_per * EEGforECG_all_random.shape[0])

    remainingforEOG = round(EEGforEOG_all_random.shape[0] - train_numforEOG)
    remainingforEMG = round(EEGforEMG_all_random.shape[0] - train_numforEMG)
    remainingforECG = round(EEGforECG_all_random.shape[0] - train_numforECG)

    val_numforEOG = remainingforEOG  // 2
    val_numforEMG = remainingforEMG // 2
    val_numforECG = remainingforECG // 2

    test_numforEOG = remainingforEOG - val_numforEOG
    test_numforEMG = remainingforEMG - val_numforEMG
    test_numforECG = remainingforECG - val_numforECG
    # for EEG Branch
    trainforEOG_eeg = EEGforEOG_all_random[0: train_numforEOG, :]
    valforEOG_eeg = EEGforEOG_all_random[train_numforEOG: train_numforEOG + val_numforEOG]
    testforEOG_eeg = EEGforEOG_all_random[train_numforEOG + val_numforEOG: train_numforEOG + val_numforEOG + test_numforEOG, :]

    trainforEMG_eeg = EEGforEMG_all_random[0: train_numforEMG, :]
    valforEMG_eeg = EEGforEMG_all_random[train_numforEMG: train_numforEMG + val_numforEMG]
    testforEMG_eeg = EEGforEMG_all_random[train_numforEMG + val_numforEMG: train_numforEMG + val_numforEMG + test_numforEMG, :]

    trainforECG_eeg = EEGforECG_all_random[0: train_numforECG, :]
    valforECG_eeg = EEGforECG_all_random[train_numforECG: train_numforECG + val_numforECG]
    testforECG_eeg = EEGforECG_all_random[train_numforECG + val_numforECG: train_numforECG + val_numforECG + test_numforECG, :]
    # for Artifacts Branch
    trainforEOG_noise = EOG_all_random[0: train_numforEOG, :]
    valforEOG_noise = EOG_all_random[train_numforEOG: train_numforEOG + val_numforEOG]
    testforEOG_noise = EOG_all_random[train_numforEOG + val_numforEOG: train_numforEOG + val_numforEOG + test_numforEOG, :]

    trainforEMG_noise = EMG_all_random[0: train_numforEMG, :]
    valforEMG_noise = EMG_all_random[train_numforEMG: train_numforEMG + val_numforEMG]
    testforEMG_noise = EMG_all_random[train_numforEMG + val_numforEMG: train_numforEMG + val_numforEMG + test_numforEMG, :]

    trainforECG_noise = ECG_all_random[0: train_numforECG, :]
    valforECG_noise = ECG_all_random[train_numforECG: train_numforECG + val_numforECG]
    testforECG_noise = ECG_all_random[train_numforECG + val_numforECG: train_numforECG + val_numforECG + test_numforECG, :]
    # data enhancement
    EEGforEOG_train = random_signal(signal=trainforEOG_eeg, combin_num=combin_num).reshape(combin_num * trainforEOG_eeg.shape[0],
                                                                               timepoint)
    NOISEforEOG_train = random_signal(signal=trainforEOG_noise, combin_num=combin_num).reshape(combin_num * trainforEOG_noise.shape[0],
                                                                                   timepoint)

    EEGforEMG_train = random_signal(signal=trainforEMG_eeg, combin_num=combin_num).reshape(combin_num * trainforEMG_eeg.shape[0],
                                                                               timepoint)
    NOISEforEMG_train = random_signal(signal=trainforEMG_noise, combin_num=combin_num).reshape(combin_num * trainforEMG_noise.shape[0],
                                                                                   timepoint)

    EEGforECG_train = random_signal(signal=trainforECG_eeg, combin_num=combin_num).reshape(combin_num * trainforECG_eeg.shape[0],
                                                                               timepoint)
    NOISEforECG_train = random_signal(signal=trainforECG_noise, combin_num=combin_num).reshape(combin_num * trainforECG_noise.shape[0],
                                                                                   timepoint)

    EEGforEOG_val = random_signal(signal=valforEOG_eeg, combin_num=combin_num).reshape(combin_num * valforEOG_eeg.shape[0],
                                                                               timepoint)
    NOISEforEOG_val = random_signal(signal=valforEOG_noise, combin_num=combin_num).reshape(combin_num * valforEOG_noise.shape[0],
                                                                                   timepoint)

    EEGforEMG_val = random_signal(signal=valforEMG_eeg, combin_num=combin_num).reshape(combin_num * valforEMG_eeg.shape[0],
                                                                               timepoint)
    NOISEforEMG_val = random_signal(signal=valforEMG_noise, combin_num=combin_num).reshape(combin_num * valforEMG_noise.shape[0],
                                                                                   timepoint)

    EEGforECG_val = random_signal(signal=valforECG_eeg, combin_num=combin_num).reshape(combin_num * valforECG_eeg.shape[0],
                                                                               timepoint)
    NOISEforECG_val = random_signal(signal=valforECG_noise, combin_num=combin_num).reshape(combin_num * valforECG_noise.shape[0],
                                                                                   timepoint)

    EEGforEOG_test = random_signal(signal=testforEOG_eeg, combin_num=combin_num).reshape(combin_num * testforEOG_eeg.shape[0],
                                                                             timepoint)
    NOISEforEOG_test = random_signal(signal=testforEOG_noise, combin_num=combin_num).reshape(combin_num * testforEOG_noise.shape[0],
                                                                                 timepoint)

    EEGforEMG_test = random_signal(signal=testforEMG_eeg, combin_num=combin_num).reshape(combin_num * testforEMG_eeg.shape[0],
                                                                             timepoint)
    NOISEforEMG_test = random_signal(signal=testforEMG_noise, combin_num=combin_num).reshape(combin_num * testforEMG_noise.shape[0],
                                                                                 timepoint)

    EEGforECG_test = random_signal(signal=testforECG_eeg, combin_num=combin_num).reshape(combin_num * testforECG_eeg.shape[0],
                                                                             timepoint)
    NOISEforECG_test = random_signal(signal=testforECG_noise, combin_num=combin_num).reshape(combin_num * testforECG_noise.shape[0],
                                                                                 timepoint)

    print(EEGforEOG_train.shape)
    print(NOISEforEOG_train.shape)
    print(EEGforEMG_train.shape)
    print(NOISEforEMG_train.shape)
    print(EEGforECG_train.shape)
    print(NOISEforECG_train.shape)

    sn_train = []
    eeg_train = []
    noise_train = []
    noise_val = []
    all_sn_test = []
    all_eeg_test = []
    all_noise_test = []

    # Adding noise to train
    SNRforEOG_train_dB = np.random.uniform(-5, 5, (EEGforEOG_train.shape[0]))
    print(SNRforEOG_train_dB.shape)
    SNRforEOG_train = np.sqrt(10 ** (0.1 * (SNRforEOG_train_dB)))

    for i in range(EEGforEOG_train.shape[0]):

        noise = preprocessing.scale(NOISEforEOG_train[i])
        EEG = preprocessing.scale(EEGforEOG_train[i])

        coe = get_rms(EEG, 0) / (get_rms(noise, 0) * SNRforEOG_train[i])
        noise = noise * coe
        signal_noise = EEG + noise

        sn_train.append(signal_noise)
        eeg_train.append(EEG)
        noise_train.append(noise)

    SNRforEMG_train_dB = np.random.uniform(-5, 5, (EEGforEMG_train.shape[0]))
    print(SNRforEMG_train_dB.shape)
    SNRforEMG_train = np.sqrt(10 ** (0.1 * (SNRforEMG_train_dB)))

    for i in range(EEGforEMG_train.shape[0]):

        noise = preprocessing.scale(NOISEforEMG_train[i])
        EEG = preprocessing.scale(EEGforEMG_train[i])

        coe = get_rms(EEG, 0) / (get_rms(noise, 0) * SNRforEMG_train[i])
        noise = noise * coe
        signal_noise = EEG + noise

        sn_train.append(signal_noise)
        eeg_train.append(EEG)
        noise_train.append(noise)

    SNRforECG_train_dB = np.random.uniform(-5, 5, (EEGforECG_train.shape[0]))
    print(SNRforECG_train_dB.shape)
    SNRforECG_train = np.sqrt(10 ** (0.1 * (SNRforECG_train_dB)))

    for i in range(EEGforECG_train.shape[0]):

        noise = preprocessing.scale(NOISEforECG_train[i])
        EEG = preprocessing.scale(EEGforECG_train[i])

        coe = get_rms(EEG, 0) / (get_rms(noise, 0) * SNRforECG_train[i])
        noise = noise * coe
        signal_noise = EEG + noise

        sn_train.append(signal_noise)
        eeg_train.append(EEG)
        noise_train.append(noise)

    # Add label for training
    label_train_eog = np.zeros(EEGforEOG_train.shape[0])
    label_train_emg = np.ones(EEGforEMG_train.shape[0])
    label_train_ecg = np.full(EEGforECG_train.shape[0], 2)
    label_train = np.concatenate([label_train_eog, label_train_emg, label_train_ecg], axis=0)

    # Adding noise to val
    sn_val = []
    eeg_val = []

    SNRforEOG_val_dB = np.random.uniform(-5, 5, (EEGforEOG_val.shape[0]))
    SNRforEOG_val = np.sqrt(10 ** (0.1 * (SNRforEOG_val_dB)))

    for i in range(EEGforEOG_val.shape[0]):
    
        noise = preprocessing.scale(NOISEforEOG_val[i])
        EEG = preprocessing.scale(EEGforEOG_val[i])

        coe = get_rms(EEG, 0) / (get_rms(noise, 0) * SNRforEOG_val[i])
        noise = noise * coe
        signal_noise = EEG + noise

        sn_val.append(signal_noise)
        eeg_val.append(EEG)
        noise_val.append(noise)

    SNRforEMG_val_dB = np.random.uniform(-5, 5, (EEGforEMG_val.shape[0]))
    SNRforEMG_val = np.sqrt(10 ** (0.1 * (SNRforEMG_val_dB)))

    for i in range(EEGforEMG_val.shape[0]):

        
        noise = preprocessing.scale(NOISEforEMG_val[i])
        EEG = preprocessing.scale(EEGforEMG_val[i])

        coe = get_rms(EEG, 0) / (get_rms(noise, 0) * SNRforEMG_val[i])
        noise = noise * coe
        signal_noise = EEG + noise

        sn_val.append(signal_noise)
        eeg_val.append(EEG)
        noise_val.append(noise)

    SNRforECG_val_dB = np.random.uniform(-5, 5, (EEGforECG_val.shape[0]))
    SNRforECG_val = np.sqrt(10 ** (0.1 * (SNRforECG_val_dB)))

    for i in range(EEGforECG_val.shape[0]):

        
        noise = preprocessing.scale(NOISEforECG_val[i])
        EEG = preprocessing.scale(EEGforECG_val[i])

        coe = get_rms(EEG, 0) / (get_rms(noise, 0) * SNRforECG_val[i])
        noise = noise * coe
        signal_noise = EEG + noise

        sn_val.append(signal_noise)
        eeg_val.append(EEG)
        noise_val.append(noise)

    # 为 val 添加 label
    label_val_eog = np.zeros(EEGforEOG_val.shape[0])
    label_val_emg = np.ones(EEGforEMG_val.shape[0])
    label_val_ecg = np.full(EEGforECG_val.shape[0], 2)
    label_val = np.concatenate([label_val_eog, label_val_emg, label_val_ecg], axis=0)

    # # Adding noise to test
    SNRforEOG_test_dB = np.linspace(-5.0, 5.0, num=(11))
    SNRforEOG_test = np.sqrt(10 ** (0.1 * (SNRforEOG_test_dB)))

    for i in range(11):

        sn_test = []
        eeg_test = []
        noise_test = []

        for j in range(EEGforEOG_test.shape[0]):
            noise = preprocessing.scale(NOISEforEOG_test[j])
            EEG = preprocessing.scale(EEGforEOG_test[j])

            coe = get_rms(EEG, 0) / (get_rms(noise, 0) * SNRforEOG_test[i])
            noise = noise * coe
            signal_noise = EEG + noise

            sn_test.append(signal_noise)
            eeg_test.append(EEG)
            noise_test.append(noise)

        sn_test = np.array(sn_test)
        eeg_test = np.array(eeg_test)
        noise_test = np.array(noise_test)

        all_sn_test.append(sn_test)
        all_eeg_test.append(eeg_test)
        all_noise_test.append(noise_test)

    SNRforEMG_test_dB = np.linspace(-5.0, 5.0, num=(11))
    SNRforEMG_test = np.sqrt(10 ** (0.1 * (SNRforEMG_test_dB)))

    for i in range(11):

        sn_test = []
        eeg_test = []
        noise_test = []

        for j in range(EEGforEMG_test.shape[0]):
            noise = preprocessing.scale(NOISEforEMG_test[j])
            EEG = preprocessing.scale(EEGforEMG_test[j])

            coe = get_rms(EEG, 0) / (get_rms(noise, 0) * SNRforEMG_test[i])
            noise = noise * coe
            signal_noise = EEG + noise

            sn_test.append(signal_noise)
            eeg_test.append(EEG)
            noise_test.append(noise)

        sn_test = np.array(sn_test)
        eeg_test = np.array(eeg_test)
        noise_test = np.array(noise_test)

        all_sn_test.append(sn_test)
        all_eeg_test.append(eeg_test)
        all_noise_test.append(noise_test)


    SNRforECG_test_dB = np.linspace(-5.0, 5.0, num=(11))
    SNRforECG_test = np.sqrt(10 ** (0.1 * (SNRforECG_test_dB)))

    for i in range(11):

        sn_test = []
        eeg_test = []
        noise_test = []

        for j in range(EEGforECG_test.shape[0]):
            noise = preprocessing.scale(NOISEforECG_test[j])
            EEG = preprocessing.scale(EEGforECG_test[j])

            coe = get_rms(EEG, 0) / (get_rms(noise, 0) * SNRforECG_test[i])
            noise = noise * coe
            signal_noise = EEG + noise

            sn_test.append(signal_noise)
            eeg_test.append(EEG)
            noise_test.append(noise)

        sn_test = np.array(sn_test)
        eeg_test = np.array(eeg_test)
        noise_test = np.array(noise_test)

        all_sn_test.append(sn_test)
        all_eeg_test.append(eeg_test)
        all_noise_test.append(noise_test)

    label_test = []
    for i in range(11):  # EOG
        label_test.append(np.zeros(EEGforEOG_test.shape[0]))
    for i in range(11):  # EMG
        label_test.append(np.ones(EEGforEMG_test.shape[0]))
    for i in range(11):  # ECG
        label_test.append(np.full(EEGforECG_test.shape[0], 2))

    label_test = np.concatenate(label_test, axis=0)

    X_train = np.array(sn_train)
    y_train = np.array(eeg_train)
    y_train_noise = np.array(noise_train)

    indices_for_train = np.random.permutation(X_train.shape[0])
    X_train = X_train[indices_for_train]
    y_train = y_train[indices_for_train]
    y_train_noise = y_train_noise[indices_for_train]
    label_train = label_train[indices_for_train]

    X_val = np.array(sn_val)
    y_val = np.array(eeg_val)
    y_val_noise = np.array(noise_val)

    X = np.concatenate(all_sn_test, axis=0)
    y = np.concatenate(all_eeg_test, axis=0)
    y_noise = np.concatenate(all_noise_test, axis=0)

    X_test = np.array(X)
    y_test = np.array(y)
    y_test_noise = np.array(y_noise)

    X_train = np.expand_dims(X_train, axis=1)
    y_train = np.expand_dims(y_train, axis=1)
    y_train_noise = np.expand_dims(y_train_noise, axis=1)

    X_val = np.expand_dims(X_val, axis=1)
    y_val = np.expand_dims(y_val, axis=1)
    y_val_noise = np.expand_dims(y_val_noise, axis=1)

    X_test = np.expand_dims(X_test, axis=1)
    y_test = np.expand_dims(y_test, axis=1)
    y_test_noise = np.expand_dims(y_test_noise, axis=1)

    np.save('.data/data_for_train/X_train.npy', X_train)
    np.save('.data/data_for_train/y_train.npy', y_train)
    np.save('.data/data_for_train/y_train_noise.npy', y_train_noise)
    np.save('.data/data_for_train/label_train.npy', label_train)

    np.save('.data/data_for_val/X_val.npy', X_val)
    np.save('.data/data_for_val/y_val.npy', y_val)
    np.save('.data/data_for_val/y_val_noise.npy', y_val_noise)
    np.save('.data/data_for_val/label_val.npy', label_val)

    np.save('.data/data_for_test/X_test.npy', X_test)
    np.save('.data/data_for_test/y_test.npy', y_test)
    np.save('.data/data_for_test/y_test_noise.npy', y_test_noise)
    np.save('.data/data_for_test/label_test.npy', label_test)

    Dataset = [X_train, y_train, y_train_noise, label_train, X_val, y_val, y_val_noise, label_val]

    print('Dataset ready to use.')

    return Dataset

if __name__ == '__main__':
    [X_train, y_train, y_train_noise, label_train, X_val, y_val, y_val_noise, label_val] = prepare_data(combin_num=11, train_per=0.8, noise_type='d4pm')