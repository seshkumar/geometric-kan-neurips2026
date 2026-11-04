import numpy as np
import torch
from src.neurips_normalization import standard_deviation
DATASETS = ['712_chscase_geyser1', '678_visualizing_environmental', '1027_ESL', '1030_ERA', '210_cloud', '230_machine_cpu', '665_sleuth_case2002', '561_cpu', '522_pm10', '547_no2', '666_rmftsa_ladata', '1028_SWD']

def load_pmlb_dataset(name, seed, test_frac=0.2, sd_ddof=1):
    from pmlb import fetch_data
    df = fetch_data(name)
    X = df.drop('target', axis=1).values.astype(np.float32)
    y = df['target'].values.astype(np.float32)
    n = len(y)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = max(int(n * test_frac), 10)
    n_train = n - n_test
    X_train, X_test = (X[idx[:n_train]], X[idx[n_train:]])
    y_train, y_test = (y[idx[:n_train]], y[idx[n_train:]])
    mu_x = X_train.mean(axis=0)
    std_x = standard_deviation(X_train, axis=0, ddof=sd_ddof)
    std_x[std_x < 1e-08] = 1.0
    X_train = (X_train - mu_x) / std_x
    X_test = (X_test - mu_x) / std_x
    mu_y = y_train.mean()
    std_y = standard_deviation(y_train, ddof=sd_ddof)
    if std_y < 1e-08:
        std_y = 1.0
    y_train_n = (y_train - mu_y) / std_y
    y_test_n = (y_test - mu_y) / std_y
    return (torch.tensor(X_train), torch.tensor(y_train_n), torch.tensor(X_test), torch.tensor(y_test_n), mu_y, std_y, y_test)
