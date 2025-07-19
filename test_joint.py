import torch
import numpy as np
import yaml
from DDPM_joint import DDPM
from denoising_model_eegdnet_class import DualBranchDenoisingModel
from denoising_model_eegdnet_class_noise import DualBranchDenoisingModel_noise
from scipy.stats import pearsonr
import pandas as pd

def rrmse_per_sample(clean_signal, denoised_signal):
    mse = ((clean_signal - denoised_signal) ** 2).mean(dim=(1, 2))
    rms = (clean_signal ** 2).mean(dim=(1, 2))
    rrmse = torch.sqrt(mse) / torch.sqrt(rms)
    return rrmse  # shape: (340,)

def snr_per_sample(clean_signal, denoised_signal):
    signal_power = (clean_signal ** 2).sum(dim=-1)  # shape: [B, 1]
    noise_power = ((clean_signal - denoised_signal) ** 2).sum(dim=-1)  # shape: [B, 1]

    snr = 10 * torch.log10(signal_power / (noise_power + 1e-8))  # 防止除0
    return snr.squeeze(1)  # shape: [B]

def compute_cc_per_sample(clean_signal, denoised_signal):
    """CC"""
    # shape: [B, 1, 512] -> [B, 512]
    clean = clean_signal.squeeze(1)
    denoised = denoised_signal.squeeze(1)

    mean_clean = clean.mean(dim=1, keepdim=True)
    mean_denoised = denoised.mean(dim=1, keepdim=True)

    num = ((clean - mean_clean) * (denoised - mean_denoised)).sum(dim=1)
    denom = torch.sqrt(((clean - mean_clean) ** 2).sum(dim=1) * ((denoised - mean_denoised) ** 2).sum(dim=1))

    return num / denom  # shape: [B]

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    noise_type = 'd4pm'
    final_pth_eeg = "./check_points/EEG_" + noise_type + "/" + "model.pth"
    final_pth_noise = "./check_points/Artifacts_" + noise_type + "/" + "model.pth"
    path = "config/" + 'base.yaml'
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    file_location = './data/data_for_test/'  ############ change it to your own location #########
    if noise_type == 'd4pm':
        X_test = np.load(file_location + 'X_test.npy')
        y_test = np.load(file_location + 'y_test.npy')
        label_test = np.load(file_location + 'label_test.npy')

    # Taking EOG as an example
    print("X_test shape: ", X_test.shape)

    X_test_EOG = X_test[:41140, :, :]
    y_test_EOG = y_test[:41140, :, :]
    label_test_EOG = label_test[:41140]
    X_test_EOG = X_test_EOG.reshape(11, 3740, 1, 512)
    y_test_EOG = y_test_EOG.reshape(11, 3740, 1, 512)
    label_test_EOG = label_test_EOG.reshape(11, 3740)

    print("X_test shape: ", X_test.shape)

    base_model_eeg = DualBranchDenoisingModel(config['train']['feats']).to(device)
    base_model_noise = DualBranchDenoisingModel_noise(config['train']['feats']).to(device)

    model_eeg = DDPM(base_model_eeg, config, device)
    model_noise = DDPM(base_model_noise, config, device)

    model_eeg.load_state_dict(torch.load(final_pth_eeg, map_location=device))
    model_noise.load_state_dict(torch.load(final_pth_noise, map_location=device))

    # Set up Joint posterior sampling
    model_eeg.model_h = model_noise.model
    model_eeg.to(device)
    model_noise.to(device)
    model_eeg.eval()
    model_noise.eval()

    results = []
    for i in range(X_test_EOG.shape[0]):
        print("combin_num: ", i)

        X_test_3 = X_test_EOG[i, :50, :, :]
        y_test_3_eeg = y_test_EOG[i, :50, :, :]
        label_test_i_EOG = label_test_EOG[i, :50]

        X_test_tensor = torch.FloatTensor(X_test_3).to(device)
        y_test_tensor = torch.FloatTensor(y_test_3_eeg).to(device)
        label_test_tensor = torch.FloatTensor(label_test_i_EOG).to(device)

        with torch.no_grad():
            # Perform joint posterior sampling
            denoised_signal = model_eeg.joint_denoising(X_test_tensor, label_test_tensor, lambda_dc=0.5, gamma=1.0, eta=0.3, continous=False)
            # print(f"denoised_signal.shape: {denoised_signal.shape}")

            # RRMSE_temporal
            rrmse_temporal = rrmse_per_sample(y_test_tensor, denoised_signal)
            rrmse_temporal = rrmse_temporal.mean().item()

            # SNR
            snr_values = snr_per_sample(y_test_tensor, denoised_signal)
            snr_mean = snr_values.mean().item()

            cc_list = []
            p_list = []
            for j in range(y_test_tensor.shape[0]):
                y_clean_np = y_test_tensor.cpu().numpy()
                y_denoised_np = denoised_signal.cpu().numpy()
                # noisy_signal_np = X_test_tensor.cpu().numpy()

                cc, p = pearsonr(y_clean_np[j, 0, :], y_denoised_np[j, 0, :])
                cc_list.append(cc)
                p_list.append(p)

            cc_mean = np.mean(cc_list)
            p_mean = np.mean(p_list)

            # RRMSE_spectral
            Y_clean = torch.abs(torch.fft.fft(y_test_tensor, dim=-1))
            Y_denoised = torch.abs(torch.fft.fft(denoised_signal, dim=-1))
            rrmse_spectral = rrmse_per_sample(Y_clean, Y_denoised)
            rrmse_spectral = rrmse_spectral.mean().item()

        results.append([i, rrmse_temporal, rrmse_spectral, cc_mean, snr_mean, p_mean])
        print(results[i])

    # Pandas DataFrame
    df_results = pd.DataFrame(results,
                              columns=["combin_num", "avg_RRMSE_temporal", "avg_RRMSE_spectral", "avg_CC", "avg_snr", "avg_p_value"])

    print(df_results)  # 直接打印
    # Save as CSV file
    df_results.to_csv("denoising_results_for_EOG.csv", index=False)

if __name__ == '__main__':
    main()

