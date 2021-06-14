import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from torch.utils.data import Dataset, DataLoader
import torch
from math import sin
from sklearn.model_selection import train_test_split


class ToyDataset(Dataset):
    def __init__(self, data, target, add_features):
        super().__init__()
        feature = ["x1", "x2"]
        feature.extend(add_features)
        self.data = data[feature].values
        self.target = target.values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = torch.tensor(self.data[index])
        y = torch.tensor(self.target[index])
        return x, y


def dataset_prepare(data, target, add_feature):
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
    train_data = ToyDataset(x_train, y_train, add_feature)
    valid_data = ToyDataset(x_valid, y_valid, add_feature)
    test_data = ToyDataset(x_test, y_test, add_feature)

    return train_data, valid_data, test_data


def data_load():
    raw_data = load_iris()
    targets = raw_data.target
    scaler = StandardScaler()
    standart = scaler.fit_transform(raw_data.data)
    pca = PCA(n_components=2).fit_transform(standart)

    features = pd.DataFrame(pca, columns=['x1', 'x2'])
    targets = pd.Series(targets)
    features['x1^2'] = features['x1']*features['x1']
    features['x2^2'] = features['x2']*features['x2']
    features['x1x2'] = features['x1']*features['x2']
    features['sin(x1)'] = features['x1'].apply(sin)
    features['sin(x2)'] = features['x2'].apply(sin)

    return features, targets


def results_collector(df, data, predicted):
    add = pd.DataFrame(data)
    add['results'] = pd.Series(predicted)
    df = df.append(add, ignore_index=True)

    return df

