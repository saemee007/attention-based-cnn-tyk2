
import pandas as pd
import torch.utils.data as data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold
import numpy as np
import os, glob
import joblib


def read_data(data_path): # json 또는 zip 파일 읽기
    data = None
    if data_path.endswith('.json'):
        try:
            data = pd.read_json(data_path, lines=True)
        except ValueError:
            data = pd.read_json(data_path)
    if data_path.endswith('.zip'):
        try:
            data = pd.read_json(data_path, compression='zip', lines=True)
        except ValueError:
            data = pd.read_json(data_path, compression='zip')
    return data


def train_validation_split(data_path): # train 80%/ test 20% 으로 split해서 json으로 저장
    if os.path.isdir(data_path): # 경로 생성
        train_path = os.path.join(data_path, 'train.json')
        val_path = os.path.join(data_path, 'val.json')
    else:
        train_path = data_path.split('.')[0] + '_' + 'train.json'
        val_path = data_path.split('.')[0] + '_' + 'val.json'
    if os.path.exists(train_path) and os.path.exists(val_path):
        # return read_data(train_path), read_data(val_path)
        return pd.read_json(train_path, lines=True), pd.read_json(val_path, lines=True)
    data = read_data(data_path)
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42) # 데이터 분할
    train_data.to_json(train_path, orient='records', lines=True) # 저장
    val_data.to_json(val_path, orient='records', lines=True) # 저장
    return train_data, val_data 


def train_cross_validation_split(data_path): # 5개 fold로 cross-validation set 생성
    dir_path = os.path.dirname(os.path.abspath(data_path))
    fold_dirs = glob.glob(os.path.join(dir_path, 'folds_*'))
    if len(fold_dirs) == 5:
        for fold_dir in fold_dirs:
            train_path = os.path.join(fold_dir, 'train.json')
            val_path = os.path.join(fold_dir, 'val.json')
            yield pd.read_json(train_path), pd.read_json(val_path)
    else:
        kfold = KFold(n_splits=5, shuffle=True, random_state=42) # 인덱스 분할
        data = read_data(data_path)
        for i, (train_ids, val_ids) in enumerate(kfold.split(X=data.drop('active', axis=1).values,
                                                             y=data['active'].values)):
            train_data = data.iloc[train_ids, :]
            val_data = data.iloc[val_ids, :]
            # os.makedirs(os.path.join(dir_path, 'folds_{}'.format(i)), exist_ok=True)
            # train_data.to_json(os.path.join(os.path.join(dir_path, 'folds_{}'.format(i)), 'train.json'))
            # val_data.to_json(os.path.join(os.path.join(dir_path, 'folds_{}'.format(i)), 'val.json'))

            yield train_data, val_data


class EGFRDataset(data.Dataset): # 데이터셋 Class (정규화 & 필요없는 col drops & label 분리 등..)
    def __init__(self, data, infer=False, train=True, hashcode=None):
        if isinstance(data, pd.DataFrame):
            self.data = data
        elif isinstance(data, str):
            self.data = read_data(data)
        self.NON_MORD_NAMES = ['smile_ft', 'id', 'subset', 'quinazoline', 'pyrimidine', 'smiles', 'active']
        self.infer = infer

        # Standardize mord features
        if not train: # 만약 학습 중이 아니라면 (valid or test)
            scl = joblib.load(f'data/trained_models/scaler_{hashcode}.pkl')
        else: # 학습 중이라면
            scl = StandardScaler()
            scl.fit(self.data.drop(columns=self.NON_MORD_NAMES).astype(np.float64))
        self.mord_ft = scl.transform(
            self.data.drop(columns=self.NON_MORD_NAMES).astype(np.float64)).tolist()
        self.non_mord_ft = self.data['smile_ft'].values.tolist()
        self.smiles = self.data['smiles'].values.tolist()
        self.label = self.data['active'].values.tolist()
        self.scaler = scl

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        if self.infer:
            return self.smiles[idx], self.mord_ft[idx], self.non_mord_ft[idx], self.label[idx]
        else:
            return self.mord_ft[idx], self.non_mord_ft[idx], self.label[idx]

    def get_dim(self, ft): # demension 구하기
        if ft == 'non_mord':
            return len(self.non_mord_ft[0])
        if ft == 'mord':
            return len(self.mord_ft[0])

    def get_smile_ft(self): # smiles feature 구하기
        return self.non_mord_ft







