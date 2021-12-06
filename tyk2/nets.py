
import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class UnitedNet(nn.Module):
    def __init__(self, dense_dim, use_mord=True, use_mat=True, infer=False, dir_path=None, vis_thresh=0.2): # layer 정의
        super(UnitedNet, self).__init__()
        self.use_mord = use_mord
        self.use_mat = use_mat
        self.infer = infer
        self.vis_thresh = vis_thresh
        self.dir_path = dir_path # data/att_weight
        if self.dir_path:
            self.smile_out_f = open(os.path.join(self.dir_path, 'smiles.txt'), 'w')
            self.weight_f = open(os.path.join(self.dir_path, 'weight.txt'), 'w')

        # PARAMS FOR CNN NET
        # Convolutionals
        self.conv_conv1 = nn.Conv2d(1, 6, kernel_size=3)
        self.conv_pool = nn.MaxPool2d(2, 2)
        self.conv_conv2 = nn.Conv2d(6, 16, kernel_size=3)

        # Fully connected
        self.conv_fc = nn.Linear(16 * 9 * 36, 150)

        # Batch norms
        self.conv_batch_norm1 = nn.BatchNorm2d(6)
        self.conv_batch_norm2 = nn.BatchNorm2d(16)

        # PARAMS FOR DENSE NET
        # Fully connected
        if self.use_mord:
            self.dense_fc1 = nn.Linear(dense_dim, 512)
            self.dense_fc2 = nn.Linear(512, 128)
            self.dense_fc3 = nn.Linear(128, 64)

            # Batch norms
            self.dense_batch_norm1 = nn.BatchNorm1d(512)
            self.dense_batch_norm2 = nn.BatchNorm1d(128)
            self.dense_batch_norm3 = nn.BatchNorm1d(64)

            # Dropouts
            self.dense_dropout = nn.Dropout()

        # PARAMS FOR ATTENTION NET
        if self.use_mat:
            self.att_fc = nn.Linear(256, 1)
        else:
            self.comb_fc_alt = nn.Linear(128, 1)

        # PARAMS FOR COMBINED NET
        if self.use_mord:
            self.comb_fc = nn.Linear(214, 1)
        else:
            self.comb_fc = nn.Linear(150, 1)

    def forward(self, x_non_mord, x_mord, x_mat, smiles=None):
        # FORWARD CNN
        x_non_mord = self.conv_conv1(x_non_mord)
        x_non_mord = self.conv_batch_norm1(x_non_mord)
        x_non_mord = F.relu(x_non_mord)
        x_non_mord = self.conv_pool(x_non_mord)

        x_non_mord = self.conv_conv2(x_non_mord)
        x_non_mord = self.conv_batch_norm2(x_non_mord)
        x_non_mord = F.relu(x_non_mord)
        x_non_mord = self.conv_pool(x_non_mord)

        x_non_mord = x_non_mord.view(x_non_mord.size(0), -1)
        if self.use_mat:
            x_non_mord = torch.sigmoid(self.conv_fc(x_non_mord))
        else:
            x_non_mord = F.relu(self.conv_fc(x_non_mord))

        # FORWARD DENSE
        if self.use_mord:
            x_mord = F.relu(self.dense_fc1(x_mord))
            x_mord = self.dense_batch_norm1(x_mord)
            x_mord = self.dense_dropout(x_mord)

            x_mord = F.relu(self.dense_fc2(x_mord))
            x_mord = self.dense_batch_norm2(x_mord)
            x_mord = self.dense_dropout(x_mord)

            x_mord = F.relu(self.dense_fc3(x_mord))
            x_mord = self.dense_batch_norm3(x_mord)
            x_mord = self.dense_dropout(x_mord)

        # FORWARD ATTENTION
        if self.use_mat:
            x_mat = torch.bmm(x_mat.permute(0, 2, 1), x_non_mord.unsqueeze(-1)).squeeze(-1) # batch matrix-matrix product / unsqueeze(-1): 마지막 차원 1 증가, squeeze(-1): 마지막 차원 1 감소
            x_mat = torch.cat([x_mat, x_non_mord], dim=1)

            if self.use_mord:
                x_comb = torch.cat([x_mat, x_mord], dim=1)
                probs = torch.sigmoid(self.att_fc(x_comb))
                if self.infer:
                    if not smiles:
                        raise ValueError('Please input smiles')
                    alphas = x_comb.cpu().detach().numpy().tolist()
                    alphas = ["\t".join([str(round(elem, 4)) for elem in seq]) for seq in alphas]
                    prob_list = probs.cpu().detach().numpy().tolist()
                    for smile, alpha, prob in zip(smiles, alphas, prob_list):
                        if prob[0] > self.vis_thresh:
                            self.weight_f.write(alpha + '\n')
                            self.smile_out_f.write(smile + '\n')
                return probs
            else:
                return torch.sigmoid(self.comb_fc(x_mat))
        else:
            if self.use_mord:
                x_comb = torch.cat([x_non_mord, x_mord], dim=1) # concat
            else:
                x_comb = x_non_mord
            return torch.sigmoid(self.comb_fc(x_comb))

    def __del__(self):
        print('Closing files ...')
        if hasattr(self, 'weight_f'):
            self.weight_f.close()
        if hasattr(self, 'smile_out_f'):
            self.smile_out_f.close()


