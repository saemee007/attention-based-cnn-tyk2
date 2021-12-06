
import os
import pickle
import torch


def get_max_length(x): # max length 구하기
    return len(max(x, key=len))


def pad_sequence(seq): # 0으로 padding 하기
    def _pad(_it, _max_len):
        return [0] * (_max_len - len(_it)) + _it
    padded = [_pad(it, get_max_length(seq)) for it in seq]
    return padded


def custom_collate(batch):  # 타입 캐스팅(tensor 자료형으로 변환)
    transposed = zip(*batch)
    lst = []
    for samples in transposed:
        try:
            if isinstance(samples[0], str) or isinstance(samples[0], unicode):
                lst.append(samples)
        except NameError:
            if isinstance(samples[0], str):
                lst.append(samples)
        if isinstance(samples[0], int):
            lst.append(torch.LongTensor(samples))
        elif isinstance(samples[0], float):
            lst.append(torch.DoubleTensor(samples))
        elif isinstance(samples[0], list):
            lst.append(torch.LongTensor(pad_sequence(samples)))
    return lst


def create_dir(dir_name): # 디랙토리 생성
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def save_pickle(obj, path): # pickle 파일로 저장
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def read_pickle(path): # pickle 파일 읽기
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_model(model, model_dir_path, hash_code): # model 저장
    if not os.path.exists(model_dir_path):
        os.makedirs(model_dir_path)
    torch.save(model.state_dict(), "{}/model_{}_{}".format(model_dir_path, hash_code, "BEST"))
    print('Save done!')

