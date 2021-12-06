import argparse
import torch
import torch.nn as nn
import tensorboard_logger
from nets import UnitedNet
from torch.utils.data import dataloader
from dataset import EGFRDataset, train_validation_split
import torch.optim as optim
from metrics import *
import utils
import matplotlib.pyplot as plt
import os
import warnings
import joblib
from sklearn.metrics import precision_recall_curve
plt.switch_backend('agg')
warnings.filterwarnings('ignore')


def train_validate_united(train_dataset,
                          val_dataset,
                          train_device,
                          val_device,
                          use_mat,
                          use_mord,
                          opt_type,
                          n_epoch,
                          batch_size,
                          metrics,
                          hash_code,
                          lr):
    train_loader = dataloader.DataLoader(dataset=train_dataset,
                                         batch_size=batch_size, # batch_size: 128
                                         collate_fn=utils.custom_collate, # 타입 캐스팅(tensor 자료형으로 변환)
                                         shuffle=False) # dataloader 함수는 자동 batching, multi-processing data loading, customize data loading order 등을 통해 효율적으로 데이터 load

    val_loader = dataloader.DataLoader(dataset=val_dataset,
                                       batch_size=batch_size, # batch_size: 128
                                       collate_fn=utils.custom_collate, # 타입 캐스팅(tensor 자료형으로 변환)
                                       shuffle=False) # dataloader 함수는 자동 batching, multi-processing data loading, customize data loading order 등을 통해 효율적으로 데이터 load

    tensorboard_logger.configure('logs/' + hash_code) # 이 경로에 자동으로 log 기록됨
 
    criterion = nn.BCELoss() # Binary Cross Entropy 오차
    united_net = UnitedNet(dense_dim=train_dataset.get_dim('mord'),
                           use_mat=use_mat, use_mord=use_mord).to(train_device) # 모델 생성 (CNN + Attension)

    if opt_type == 'sgd': # SGD를 optimizer로 사용할 경우
        opt = optim.SGD(united_net.parameters(),
                        lr=lr,
                        momentum=0.99) # learning rate과 momentum(관성)을 고려하여 속도 결정
    elif opt_type == 'adam':
        opt = optim.Adam(united_net.parameters(), # Adagrad + Momentum
                         lr=lr)

    min_loss = 100  # 최소 오차
    early_stop_count = 0 # early stopping 시작
    for e in range(n_epoch): # 각 epoch 마다
        train_losses = []
        val_losses = []
        train_outputs = []
        val_outputs = []
        train_labels = []
        val_labels = []
        print(e, '--', 'TRAINING ==============>')
        for i, (mord_ft, non_mord_ft, label) in enumerate(train_loader): # trainning data 학습
            united_net.train() # training 시작
            mord_ft = mord_ft.float().to(train_device) # molecular discriptors
            non_mord_ft = non_mord_ft.view((-1, 1, 150, 42)).float().to(train_device) # non molecular discriptors
            mat_ft = non_mord_ft.squeeze(1).float().to(train_device) # attention에 쓰일 matrix
            label = label.float().to(train_device) # target

            # Forward
            opt.zero_grad()
            outputs = united_net(non_mord_ft, mord_ft, mat_ft) # forward

            loss = criterion(outputs, label) # Binary Cross Entropy 오차
            train_losses.append(float(loss.item()))
            train_outputs.extend(outputs)
            train_labels.extend(label)

            # Parameters update
            loss.backward() # backword
            opt.step()

        # Validate after each epoch
        print('EPOCH', e, '--', 'VALIDATION ==============>')
        for i, (mord_ft, non_mord_ft, label) in enumerate(val_loader): 
            united_net.eval() # validation 시작 (dropout, batchnorm layer 등과 같은 train time과 달리 eval time에서는 사용하지 않아야 하는 layer 알아서 off)
            mord_ft = mord_ft.float().to(val_device) # molecular discriptors
            non_mord_ft = non_mord_ft.view((-1, 1, 150, 42)).float().to(val_device) # non molecular discriptors
            mat_ft = non_mord_ft.squeeze(1).float().to(train_device) # attention에 쓰일 matrix
            label = label.float().to(val_device) # target

            with torch.no_grad():
                outputs = united_net(non_mord_ft, mord_ft, mat_ft) # forward

                loss = criterion(outputs, label) # Binary Cross Entropy 오차
                val_losses.append(float(loss.item()))
                val_outputs.extend(outputs)
                val_labels.extend(label)

        train_outputs = torch.stack(train_outputs)
        val_outputs = torch.stack(val_outputs)
        train_labels = torch.stack(train_labels)
        val_labels = torch.stack(val_labels)
        tensorboard_logger.log_value('train_loss', sum(train_losses) / len(train_losses), e + 1)
        tensorboard_logger.log_value('val_loss', sum(val_losses) / len(val_losses), e + 1)
        print('{"metric": "train_loss", "value": %f, "epoch": %d}' % (sum(train_losses) / len(train_losses), e + 1))
        print('{"metric": "val_loss", "value": %f, "epoch": %d}' % (sum(val_losses) / len(val_losses), e + 1))
        for key in metrics.keys():
            train_metric = metrics[key](train_labels, train_outputs)
            val_metric = metrics[key](val_labels, val_outputs)
            print('{"metric": "%s", "value": %f, "epoch": %d}' % ('train_' + key, train_metric, e + 1))
            print('{"metric": "%s", "value": %f, "epoch": %d}' % ('val_' + key, val_metric, e + 1))
            tensorboard_logger.log_value('train_{}'.format(key),
                                         train_metric, e + 1)
            tensorboard_logger.log_value('val_{}'.format(key),
                                         val_metric, e + 1)
        loss_epoch = sum(val_losses) / len(val_losses)
        if loss_epoch < min_loss:
            early_stop_count = 0
            min_loss = loss_epoch
            utils.save_model(united_net, "data/trained_models", hash_code)
        else:
            early_stop_count += 1
            if early_stop_count > 30:
                print('Traning can not improve from epoch {}\tBest loss: {}'.format(e, min_loss))
                break

    train_metrics = {}
    val_metrics = {}
    for key in metrics.keys():
        train_metrics[key] = metrics[key](train_labels, train_outputs)
        val_metrics[key] = metrics[key](val_labels, val_outputs)

    return train_metrics, val_metrics


def predict(dataset, model_path, device='cpu'):
    loader = dataloader.DataLoader(dataset=dataset,
                                   batch_size=128,
                                   collate_fn=utils.custom_collate,
                                   shuffle=False)
    united_net = UnitedNet(dense_dim=dataset.get_dim('mord'), use_mat=True).to(device)
    united_net.load_state_dict(torch.load(model_path, map_location=device))
    # EVAL_MODE
    united_net.eval()
    probas = []
    for i, (mord_ft, non_mord_ft, label) in enumerate(loader):
        with torch.no_grad():
            mord_ft = mord_ft.float().to(device)
            non_mord_ft = non_mord_ft.view((-1, 1, 150, 42)).float().to(device)
            mat_ft = non_mord_ft.squeeze(1).float().to(device)
            # Forward to get smiles and equivalent weights
            proba = united_net(non_mord_ft, mord_ft, mat_ft)
            probas.append(proba)
    print('Forward done !!!')
    probas = np.concatenate(probas)
    return probas


def plot_roc_curve(y_true, y_pred, hashcode=''): # ROC curve 그림

    if not os.path.exists('vis/'):
        os.makedirs('vis/')

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1) 
    auc_roc = metrics.roc_auc_score(y_true, y_pred)
    print('AUC: {:4f}'.format(auc_roc))
    plt.plot(fpr, tpr)
    plt.savefig('vis/ROC_{}'.format(hashcode + '.png')) # 그림 저장
    plt.clf()  # Clear figure


def plot_precision_recall(y_true, y_pred, hashcode=''): # precision recall plot 그림

    if not os.path.exists('vis/'):
        os.makedirs('vis/')

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred)
    plt.plot(thresholds, precisions[:-1], label="Precision")
    plt.plot(thresholds, recalls[:-1], label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])
    plt.savefig('vis/PR_{}'.format(hashcode + '.png')) # 그림 저장
    plt.clf()  # Clear figure


def main():
    parser = argparse.ArgumentParser() # Run할 때, parameter 설정
    parser.add_argument('-d', '--dataset', help='Input dataset', dest='dataset',
                        default='data/egfr_10_full_ft_pd_lines.json')
    parser.add_argument('-e', '--epochs', help='Number of epochs', dest='epochs', default=500) # default epochs: 500 
    parser.add_argument('-b', '--batchsize', help='Batch size', dest='batchsize', default=128) # default batchsize: 128 
    parser.add_argument('-o', '--opt', help='Optimizer adam or sgd', dest='opt', default='adam')  # default optimizer: adam
    parser.add_argument('-g', '--gpu', help='Use GPU or Not?', action='store_true')  # default use gpu
    parser.add_argument('-c', '--hashcode', help='Hashcode for tf.events', dest='hashcode', default='TEST')
    parser.add_argument('-l', '--lr', help='Learning rate', dest='lr', default=1e-5, type=float) # default learning rate: 1e-5
    parser.add_argument('-k', '--mode', help='Train or predict ?', dest='mode', default='train')  # default mode: train
    parser.add_argument('-m', '--model_path', help='Trained model path', dest='model_path')
    parser.add_argument('-uma', '--use_mat',
                        default=True,
                        help='Use mat feature or not',
                        dest='use_mat',
                        type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('-umo', '--use_mord',
                        default=True,
                        help='Use mord feature or not', # False면 non_mord_ft만 사용, True면 non_mord_ft / mord_ft 둘다 사용 
                        dest='use_mord',
                        type=lambda x: (str(x).lower() == 'true'))
    args = parser.parse_args()

    train_data, val_data = train_validation_split(args.dataset) # train 80%/ test 20% 으로 split해서 json으로 저장
    train_dataset = EGFRDataset(train_data) # 데이터셋 Class (정규화 & 필요없는 col drops & label 분리 등..)
    scaler = train_dataset.scaler
    os.makedirs('./data/trained_models/', exist_ok=True)
    joblib.dump(scaler, os.path.join(f'./data/trained_models/scaler_{args.hashcode}.pkl'))
    val_dataset = EGFRDataset(val_data, train=False, hashcode=args.hashcode) # 데이터셋 Class (정규화 & 필요없는 col drops & label 분리 등..)
    if torch.cuda.is_available(): # cuda 사용가능하면 사용, 아니면 cpu 사용
        train_device = 'cuda'
        val_device = 'cuda'
    else:
        train_device = 'cpu'
        val_device = 'cpu'

    if args.mode == 'train': # 학습 모드일 경우
        train_validate_united(train_dataset,
                              val_dataset,
                              train_device,
                              val_device,
                              args.use_mat,
                              args.use_mord,
                              args.opt,
                              int(args.epochs),
                              int(args.batchsize),
                              {'sensitivity': sensitivity, 'specificity': specificity,
                               'accuracy': accuracy, 'mcc': mcc, 'auc': auc},
                              args.hashcode,
                              args.lr)
    elif args.mode == 'test': # 테스트 모드일 경우
        pred_dataset = EGFRDataset(args.dataset, train=False, hashcode=args.hashcode)  # 데이터셋 Class (정규화 & 필요없는 col drops & label 분리 등..)
        y_pred = predict(pred_dataset, args.model_path, train_device) # predict data 예측
        y_true = pred_dataset.label

        plot_roc_curve(y_true, y_pred, args.model_path.split('/')[-1]) # ROC curve 그림
        plot_precision_recall(y_true, y_pred, args.model_path.split('/')[-1]) # precision recall plot 그림
    else: 
        ValueError

if __name__ == '__main__':
    main()


