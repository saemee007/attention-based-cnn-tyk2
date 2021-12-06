import argparse
import torch
import torch.nn as nn
import tensorboard_logger
from nets import UnitedNet
from torch.utils.data import dataloader
from dataset import EGFRDataset, train_cross_validation_split
import torch.optim as optim
from metrics import *
import utils
import matplotlib.pyplot as plt
import os
import pandas as pd
import warnings
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
                          lr,
                          fold):
    train_loader = dataloader.DataLoader(dataset=train_dataset,
                                         batch_size=batch_size,
                                         collate_fn=utils.custom_collate,
                                         shuffle=False)

    val_loader = dataloader.DataLoader(dataset=val_dataset,
                                       batch_size=batch_size,
                                       collate_fn=utils.custom_collate,
                                       shuffle=False)

    criterion = nn.BCELoss()
    united_net = UnitedNet(dense_dim=train_dataset.get_dim('mord'),
                           use_mat=use_mat, use_mord=use_mord).to(train_device)

    if opt_type == 'sgd':
        opt = optim.SGD(united_net.parameters(),
                        lr=lr,
                        momentum=0.99)
    elif opt_type == 'adam':
        opt = optim.Adam(united_net.parameters(),
                         lr=lr)

    min_loss = 100  # arbitary large number
    early_stop_count = 0
    for e in range(n_epoch):
        train_losses = []
        val_losses = []
        train_outputs = []
        val_outputs = []
        train_labels = []
        val_labels = []
        print('FOLD', fold, '-- EPOCH', e, '-- TRAINING')
        for i, (mord_ft, non_mord_ft, label) in enumerate(train_loader):
            united_net.train()
            mord_ft = mord_ft.float().to(train_device)
            non_mord_ft = non_mord_ft.view((-1, 1, 150, 42)).float().to(train_device)
            mat_ft = non_mord_ft.squeeze(1).float().to(train_device)
            label = label.float().to(train_device)

            # Forward
            opt.zero_grad()
            outputs = united_net(non_mord_ft, mord_ft, mat_ft)

            loss = criterion(outputs, label)
            train_losses.append(float(loss.item()))
            train_outputs.extend(outputs)
            train_labels.extend(label)

            # Parameters update
            loss.backward()
            opt.step()

        # Validate after each epoch
        print('FOLD', fold, 'EPOCH', e, '--', 'VALIDATING')
        for i, (mord_ft, non_mord_ft, label) in enumerate(val_loader):
            united_net.eval()
            mord_ft = mord_ft.float().to(val_device)
            non_mord_ft = non_mord_ft.view((-1, 1, 150, 42)).float().to(val_device)
            mat_ft = non_mord_ft.squeeze(1).float().to(train_device)
            label = label.float().to(val_device)

            with torch.no_grad():
                outputs = united_net(non_mord_ft, mord_ft, mat_ft)

                loss = criterion(outputs, label)
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
            utils.save_model(united_net, "data/trained_models", hash_code + "_" + str(fold))
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


def predict(dataset, model_path, device='cpu', use_mat=True, use_mord=True):
    loader = dataloader.DataLoader(dataset=dataset,
                                   batch_size=128,
                                   collate_fn=utils.custom_collate,
                                   shuffle=False)
    united_net = UnitedNet(dense_dim=dataset.get_dim('mord'),
                           use_mat=use_mat, use_mord=use_mord).to(device)
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


def plot_roc_curve(y_true, y_pred, hashcode=''):

    if not os.path.exists('vis/'):
        os.makedirs('vis/')

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
    auc_roc = metrics.roc_auc_score(y_true, y_pred)
    print('AUC: {:4f}'.format(auc_roc))
    plt.plot(fpr, tpr)
    plt.savefig('vis/ROC_{}'.format(hashcode + '.png'))
    plt.clf()  # Clear figure


def plot_precision_recall(y_true, y_pred, hashcode=''):

    if not os.path.exists('vis/'):
        os.makedirs('vis/')

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred)
    plt.plot(thresholds, precisions[:-1], label="Precision")
    plt.plot(thresholds, recalls[:-1], label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])
    plt.savefig('vis/PR_{}'.format(hashcode + '.png'))
    plt.clf()  # Clear figure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', help='Input dataset', dest='dataset',
                        default='data/egfr_10_full_ft_pd_lines.json')
    parser.add_argument('-e', '--epochs', help='Number of epochs', dest='epochs', default=500)
    parser.add_argument('-b', '--batchsize', help='Batch size', dest='batchsize', default=128)
    parser.add_argument('-o', '--opt', help='Optimizer adam or sgd', dest='opt', default='adam')
    parser.add_argument('-g', '--gpu', help='Use GPU or Not?', action='store_true')
    parser.add_argument('-c', '--hashcode', help='Hashcode for tf.events', dest='hashcode', default='TEST')
    parser.add_argument('-l', '--lr', help='Learning rate', dest='lr', default=1e-5, type=float)
    parser.add_argument('-k', '--mode', help='Train or predict ?', dest='mode', default='train')
    parser.add_argument('-m', '--model_path', help='Trained model path', dest='model_path')
    parser.add_argument('-uma', '--use_mat',
                        default=True,
                        help='Use mat feature or not',
                        dest='use_mat',
                        type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('-umo', '--use_mord',
                        default=True,
                        help='Use mord feature or not',
                        dest='use_mord',
                        type=lambda x: (str(x).lower() == 'true'))
    args = parser.parse_args()

    if torch.cuda.is_available():
        train_device = 'cuda'
        val_device = 'cuda'
    else:
        train_device = 'cpu'
        val_device = 'cpu'

    tensorboard_logger.configure('logs/' + args.hashcode)

    metrics_cv_dict = {'sensitivity': sensitivity_cv,
                       'specificity': specificity_cv,
                       'accuracy': accuracy_cv,
                       'mcc': mcc_cv,
                       'auc': auc_cv}

    train_cv_metrics = []
    val_cv_metrics = []
    best_cv = []
    fold = 0
    for train_data, val_data in train_cross_validation_split(args.dataset):
        fold += 1
        train_dataset = EGFRDataset(train_data)
        val_dataset = EGFRDataset(val_data)
        train_metrics, val_metrics = train_validate_united(train_dataset,
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
                                                           args.lr,
                                                           fold)
        train_cv_metrics.append(train_metrics)
        val_cv_metrics.append(val_metrics)

        # VALIDATE in BEST MODEL
        filename = "data/trained_models/model_" + args.hashcode + '_' + str(fold) + "_BEST"
        y_pred = predict(val_dataset, filename, use_mat=args.use_mat, use_mord=args.use_mord)
        y_true = val_dataset.label
        bestcv = {}
        for metric, metric_fun in metrics_cv_dict.items():
            bestcv[metric] = metric_fun(y_true, y_pred)

        best_cv.append(bestcv)

    train_metrics_df = pd.DataFrame(train_cv_metrics)
    val_metrics_df = pd.DataFrame(val_cv_metrics)
    best_cv_df = pd.DataFrame(best_cv)
    cv_metrics_df = pd.DataFrame([train_metrics_df.mean(),
                                  val_metrics_df.mean(),
                                  best_cv_df.mean()],
                                 index=['train', 'val', 'load'])
    print(cv_metrics_df[['sensitivity', 'specificity', 'accuracy', 'mcc', 'auc']])


if __name__ == '__main__':
    main()


