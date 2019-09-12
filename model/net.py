"""Defines the neural network, loss function and metrics"""

import numpy as np
import torch
import torch.nn as nn

CHED = 8  # Character Embedding Dimension

IK1 = 1  # Input 1 Kernel Size
IK2 = 3  # Input 2 Kernel Size
IK3 = 5  # Input 3 Kernel Size
IK4 = 7  # Input 4 Kernel Size

ICO1 = 50  # Input 1 Convolution Output Size
ICO2 = 50  # Input 2 Convolution Output Size
ICO3 = 75  # Input 3 Convolution Output Size
ICO4 = 75  # Input 4 Convolution Output Size

DCI1 = 250  # Dilated Convolution 1 Input Dimension
DCI2 = 500  # Dilated Convolution 2 Input Dimension
DCI3 = 750  # Dilated Convolution 3 Input Dimension
DCI4 = 1000  # Dilated Convolution 4 Input Dimension
DCI5 = 1250  # Dilated Convolution 5 Input Dimension
DCI6 = 1500  # Dilated Convolution 6 Input Dimension

DCO1 = 250  # Dilated Convolution 1 Output Dimension
DCO2 = 250  # Dilated Convolution 2 Output Dimension
DCO3 = 250  # Dilated Convolution 3 Output Dimension
DCO4 = 250  # Dilated Convolution 4 Output Dimension
DCO5 = 250  # Dilated Convolution 5 Output Dimension
DCO6 = 250  # Dilated Convolution 6 Output Dimension

IP2 = 1  # Input 2 Padding Size
IP3 = 2  # Input 3 Padding Size
IP4 = 3  # Input 4 Padding Size

OK = 3  # Output 1 Kernel Size
OP = 1  # Output 1 Kernel Size

DK = 3  # Dilation kernel size
DS = 2  # Dilation Size
DP = 2  # Dilation Padding
SD = 1  # SoftMax dimension
DCO = 250


class Merger(nn.Module):
    """Merges two tensors"""

    def __init__(self):
        super(Merger, self).__init__()

    def forward(self, *input_tensor, dim=1):
        return torch.cat(input_tensor, dim)


class Merger_2(nn.Module):
    """Merges two tensors"""

    def __init__(self):
        super(Merger_2, self).__init__()

    def forward(self, x, y, dim=2):
        x = x.view(x.size()[0], x.size()[1], 1)
        y = y.view(y.size()[0], y.size()[1], 1)
        return torch.cat((x, y), dim)


class Flatten(nn.Module):
    """Flattens Tensor"""

    def forward(self, x):
        return x.view(x.size()[0], -1)


class MerchantNet(nn.Module):
    """
    Convolutional Neural Network with 8 layers.
    Used for merchant string detection in transaction strings.
    """

    def __init__(self):
        super(MerchantNet, self).__init__()

        self.conv_1 = nn.Conv1d(CHED, ICO1, kernel_size=IK1)
        self.conv_2 = nn.Conv1d(CHED, ICO2, kernel_size=IK2, padding=IP2)
        self.conv_3 = nn.Conv1d(CHED, ICO3, kernel_size=IK3, padding=IP3)
        self.conv_4 = nn.Conv1d(CHED, ICO4, kernel_size=IK4, padding=IP4)

        self.merge_1 = Merger()

        self.bnorm_1 = nn.BatchNorm1d(DCO)
        self.act_1 = nn.ReLU()

        self.tr_conv_1 = nn.Conv1d(DCI1, DCO1, kernel_size=DK, dilation=DS, padding=DP)
        self.bnorm_2 = nn.BatchNorm1d(DCO1)
        self.act_2 = nn.ReLU()

        self.merge_2 = Merger()

        self.tr_conv_2 = nn.Conv1d(DCI2, DCO2, kernel_size=DK, dilation=DS, padding=DP)
        self.bnorm_3 = nn.BatchNorm1d(DCO2)
        self.act_3 = nn.ReLU()

        self.merge_3 = Merger()

        self.tr_conv_3 = nn.Conv1d(DCI3, DCO3, kernel_size=DK, dilation=DS, padding=DP)
        self.bnorm_4 = nn.BatchNorm1d(DCO3)
        self.act_4 = nn.ReLU()

        self.merge_4 = Merger()

        self.tr_conv_4 = nn.Conv1d(DCI4, DCO4, kernel_size=DK, dilation=DS, padding=DP)
        self.bnorm_5 = nn.BatchNorm1d(DCO4)
        self.act_5 = nn.ReLU()

        self.merge_5 = Merger()

        self.tr_conv_5 = nn.Conv1d(DCI5, DCO5, kernel_size=DK, dilation=DS, padding=DP)
        self.bnorm_6 = nn.BatchNorm1d(DCO5)
        self.act_6 = nn.ReLU()

        self.merge_6 = Merger()

        self.tr_conv_6 = nn.Conv1d(DCI6, DCO6, kernel_size=DK, dilation=DS, padding=DP)
        self.bnorm_7 = nn.BatchNorm1d(DCO6)
        self.act_7 = nn.ReLU()

        self.merge_7 = Merger()

        self.conv_5 = nn.Conv1d(DCI6, 1, kernel_size=OK, padding=OP)
        self.conv_6 = nn.Conv1d(DCI6, 1, kernel_size=OK, padding=OP)

        self.flatten_1 = Flatten()
        self.flatten_2 = Flatten()

        self.start_idx = nn.Softmax(SD)
        self.end_idx = nn.Softmax(SD)

        self.out_merge = Merger_2()

    def forward(self, input):
        """
        Takes One 2 Dimensional input: Character Embedding Dimension * Character Sequence Length

        Args:
            input: (Variable) contains a batch of sentences, of dimension (batch_size x max_seq_len x char_emb_dim),
                    where seq_len is the length of the longest sentence in the batch.

        Returns:
            indices_res: (Variable) dimension batch_size * 2 with start and end indices of contained target substring
                        of each sentence.
        """

        conv_1_res = self.conv_1(input)
        conv_2_res = self.conv_2(input)
        conv_3_res = self.conv_3(input)
        conv_4_res = self.conv_4(input)

        merge_1_res = self.merge_1(conv_1_res, conv_2_res, conv_3_res, conv_4_res)

        bnorm_1_res = self.bnorm_1(merge_1_res)
        act_1_res = self.act_1(bnorm_1_res)

        tr_conv_1_res = self.tr_conv_1(act_1_res)
        bnorm_2_res = self.bnorm_2(tr_conv_1_res)
        act_2_res = self.act_2(bnorm_2_res)

        merge_2_res = self.merge_2(act_1_res, act_2_res)

        tr_conv_2_res = self.tr_conv_2(merge_2_res)
        bnorm_3_res = self.bnorm_3(tr_conv_2_res)
        act_3_res = self.act_3(bnorm_3_res)

        merge_3_res = self.merge_3(act_3_res, merge_2_res)

        tr_conv_3_res = self.tr_conv_3(merge_3_res)
        bnorm_4_res = self.bnorm_4(tr_conv_3_res)
        act_4_res = self.act_4(bnorm_4_res)

        merge_4_res = self.merge_4(act_4_res, merge_3_res)

        tr_conv_4_res = self.tr_conv_4(merge_4_res)
        bnorm_5_res = self.bnorm_5(tr_conv_4_res)
        act_5_res = self.act_5(bnorm_5_res)

        merge_5_res = self.merge_5(act_5_res, merge_4_res)

        tr_conv_5_res = self.tr_conv_5(merge_5_res)
        bnorm_6_res = self.bnorm_6(tr_conv_5_res)
        act_6_res = self.act_6(bnorm_6_res)

        merge_6_res = self.merge_6(act_6_res, merge_5_res)

        tr_conv_6_res = self.tr_conv_6(merge_6_res)
        bnorm_7_res = self.bnorm_7(tr_conv_6_res)
        act_7_res = self.act_7(bnorm_7_res)

        merge_7_res = self.merge_7(act_2_res, act_3_res, act_4_res, act_5_res, act_6_res, act_7_res)

        conv_5_res = self.conv_5(merge_7_res)
        conv_6_res = self.conv_6(merge_7_res)

        flatten_1_res = self.flatten_1(conv_5_res)
        flatten_2_res = self.flatten_2(conv_6_res)

        start_idx_res = self.start_idx(flatten_1_res)
        end_idx_res = self.end_idx(flatten_2_res)

        indices_res = self.out_merge(start_idx_res, end_idx_res)

        return indices_res


def accuracy(outputs, indices):
    """
    Compute the accuracy, given the outputs and indices.

    Args:
        outputs: (np.ndarray) dimension batch_size * seq_len * 2 - softmax output of the model
        indices: (np.ndarray) dimension batch_size x seq_len * 2 - given indices.

    Returns:
        acc: (float) accuracy in [0,1]
    """

    # reshape indices to give a flat vector of length batch_size*seq_len*2
    indices = indices.reshape(indices.shape[0], -1)

    # np.argmax gives us the class predicted for each token by the model
    outputs = np.argmax(outputs, axis=1)

    # count each samples with correctly predicted both indices
    corrects = ((outputs == indices).sum(axis=1) == 2)

    # calculate overal part of correctly predicted pair of indices
    acc = corrects.sum() / corrects.shape[0]
    return acc


def get_indices_and_confidences(outputs, indices):
    """
    Get nn prediction results with real indices, predicted indices and confidences of prediction.
    Args:
        outputs: (np.ndarray) dimension batch_size * seq_len * 2 - softmax output of the model
        indices: (np.ndarray) dimension batch_size x seq_len * 2 - given indices.

    Returns:
        results: (list) lists of indices, predicted_indices and confidences of batch

    """

    # reshape indices to give a flat vector of length batch_size*seq_len*2
    indices = indices.reshape(indices.shape[0], -1)

    # np.argmax gives us the class predicted for each token by the model
    predicted_indices = np.argmax(outputs, axis=1)
    confidences = np.max(outputs, axis=1)

    results = [[indices, predicted_indices, confidences]]

    return results


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'accuracy': accuracy,
    # could add more metrics such as accuracy for each token type
}
