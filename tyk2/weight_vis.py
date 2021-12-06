
import torch
import argparse
import pandas as pd
from nets import UnitedNet
from dataset import EGFRDataset
from torch.utils.data import dataloader
import utils
from rdkit import Chem
from rdkit.Chem.Draw import SimilarityMaps
from feature import atom_flag
import os
import numpy as np
import ast #파이썬 응용 프로그램이 파이썬 추상 구문 문법의 트리를 처리하는 데 도움을 줌


def get_mol_importance(data_path, model_path, dir_path, device):
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)

    data = pd.read_json(data_path, lines=True)
    dataset = EGFRDataset(data, infer=True) # 정규화 & 필요없는 col drops
    loader = dataloader.DataLoader(dataset=dataset,
                                   batch_size=128,
                                   collate_fn=utils.custom_collate,
                                   shuffle=False)
    united_net = UnitedNet(dense_dim=dataset.get_dim('mord'), use_mat=True, dir_path=dir_path, infer=True).to(device) # 각 layer와 변수 정의
    united_net.load_state_dict(torch.load(model_path, map_location=device)) # 모델의 매개변수 불러옴
    united_net.eval() # dropout, batchnorm layer 등과 같은 train time과 달리 eval time에서는 사용하지 않아야 하는 layer 알아서 off시키는 함수

    for i, (smiles, mord_ft, non_mord_ft, label) in enumerate(loader):
        with torch.no_grad(): # locally disabling gradient computationㄴ
            mord_ft = mord_ft.float().to(device) # 재할당 필수, tensor의 장치 올리기 또는 변환(cpu or gpu) 
            non_mord_ft = non_mord_ft.view((-1, 1, 150, 42)).float().to(device) # reshape
            mat_ft = non_mord_ft.squeeze(1).float().to(device)
            o = united_net(non_mord_ft, mord_ft, mat_ft, smiles=smiles)
    print('Forward done !!!')


def weight_vis(smiles, weights, cm='jet', lines=10):
    m = Chem.MolFromSmiles(smiles)
    try: smi = Chem.MolToSmiles(m, kekuleSmiles=True, isomericSmiles=True, rootedAtAtom=int(n))
    except: smi = Chem.MolToSmiles(m, kekuleSmiles=True, isomericSmiles=True)
    smi = Chem.MolToSmiles(m) # smiles -> canonical smiles
    aod = ast.literal_eval(m.GetProp('_smilesAtomOutputOrder')) # smiles에 원자가 쓰여진 순서
    flg = atom_flag(smi,150) # maxlength = 150, 전체 smiles 중 weigth 추출할 원자 선택 (수소가 아닌 원자들의 weight 추출)
    extracted_wt = [weights[i] for i in range(len(flg)) if flg[i]]
    fig = SimilarityMaps.GetSimilarityMapFromWeights(m, extracted_wt, colorMap=cm, contourLines=lines) # 원자 weight 주어진 분자에 대한 유사성 맵을 생성
    return fig


def save_weight_vis(smiles_file, weight_file, save_dir):
    with open(smiles_file) as f:  
        smiles = f.read().splitlines() # smiles load & split

    weights = np.loadtxt(weight_file) # atom 별 weight load

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i in range(len(smiles)):
        filename = os.path.join(save_dir, str(i) + '.png')
        fig = weight_vis(smiles[i], weights[i])
        fig.savefig(filename, bbox_inches="tight", pad_inches=0) # 각 분자별 similarity map 저장
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', help='Input dataset',
                        dest='dataset', default='data/egfr_10_full_ft_pd_lines.json')
    parser.add_argument('-m', '--modelpath', help='Input dataset',
                        dest='modelpath', default='data/model')
    parser.add_argument('-dir', '--dirpath', help='Directory to save attention weights',
                        dest='dirpath', default='data/att_weight')
    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    get_mol_importance(args.dataset, args.modelpath, args.dirpath, args.device) # 데이터 load & preprocessing, att model 생성 & 학습
    save_weight_vis(os.path.join(args.dirpath, 'smiles.txt'), # data/att_weight/smiles.txt
                    os.path.join(args.dirpath, 'weight.txt'), # data/att_weight/weight.txt
                    os.path.join(args.dirpath, 'vis/') # data/att_weight/vis/
                    )


if __name__ == '__main__':
    main()
