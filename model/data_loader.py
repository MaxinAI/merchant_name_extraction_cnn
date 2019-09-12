import os
import random

import numpy as np
import torch
from torch.autograd import Variable

import utils


class DataLoader(object):
    """
    Handles all aspects of the data. Stores the dataset_params.
    """

    def __init__(self, data_dir, params):
        """
        Loads dataset_params. It also modifies params and appends dataset_params to params.

        Args:
            data_dir: (string) directory containing the dataset
            params: (Params) hyperparameters of the training process.
        """

        json_path = os.path.join(data_dir, 'dataset_params.json')
        assert os.path.isfile(json_path), f'No json file found at {json_path}, run build_vocab.py'
        self.dataset_params = utils.Params(json_path)

    @staticmethod
    def character_embedding(text, max_len=300, emb_dim=8):
        """
        Embeds character string with the use of (emb_dim)-bit binary values of each character.

        Args:
            text: (string) text to embed
            max_len: (int) maximum length of text that will be encoded. Padding will be done with zeros.
            emb_dim:

        Returns:
            str_array: (ndarray) 2 dimensional numpy array containing embedded text of shape emb_dim*max_len

        """

        # cut long text with maximum accepted length
        if len(text) > max_len:
            text = text[:max_len]

        str_array = np.zeros((emb_dim, max(len(text), max_len)), dtype=np.int32).tolist()

        for index, char in enumerate(text):
            str_binary = format(ord(char), 'b').zfill(emb_dim)[::-1]
            str_binary = str_binary[:emb_dim]
            for str_index, str_char in enumerate(str_binary, 0):
                str_array[str_index][index] = int(str_char)

        padding_str_binary = '0' * emb_dim
        for index in range(len(text), max_len):
            for str_index, str_char in enumerate(padding_str_binary, 0):
                str_array[str_index][index] = int(str_char)

        return str_array

    def load_sentences_indices(self, sentences_file, indices_file, d):
        """
        Loads sentences and labels from their corresponding files. Finds merchants in sentences and
        stores indices in dict d.

        Args:
            sentences_file: (string) file with sentences with tokens space-separated
            indices_file: (string) file with NER tags for the sentences in labels_file
            d: (dict) a dictionary in which the loaded data is stored

        """

        sentences = []
        real_sentences = []
        indices = []

        with open(sentences_file, encoding='utf-8') as f:
            for sentence in f.read().splitlines():
                if len(sentence) > 0:
                    real_sentences.append(sentence)
                    embedded_sentence = self.character_embedding(sentence)
                    sentences.append(embedded_sentence)

        with open(indices_file, encoding='utf-8') as f:
            for line in f.read().splitlines():
                if len(line) > 0:
                    start_idx, end_idx = line.split()
                    indices.append((int(start_idx), int(end_idx)))

        # checks to ensure there is a indices for each sentence
        assert len(indices) == len(sentences)
        for i in range(len(indices)):
            assert len(indices[i]) == 2

        # storing sentences and indices in dict d
        d['sentences'] = real_sentences
        d['data'] = sentences
        d['indices'] = indices
        d['size'] = len(sentences)

    def load_data(self, types, data_dir):
        """
         Loads the data for each type in types from data_dir.

        Args:
            types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
            data_dir:  (string) directory containing the dataset

        Returns:
            data: (dict) contains the data dictionary splitted into train/val/test with [data,indices,size] inside

        """

        data = {}

        for split in types:
            sentences_file = os.path.join(data_dir, split, "sentences.txt")
            indices_file = os.path.join(data_dir, split, "indices.txt")
            data[split] = {}
            self.load_sentences_indices(sentences_file, indices_file, data[split])

        return data

    def data_iterator(self, data, params, shuffle=False):
        """
        Returns a generator that yields batches data with indices. Batch size is params.batch_size. Expires after one
        pass over the data.

        Args:
            data: (dict) contains data which has keys 'data', 'indices' and 'size'
            params: (Params) hyperparameters of the training process.
            shuffle: (bool) whether the data should be shuffled

        Returns:
            batch_data: (Variable) dimension batch_size x seq_len with the sentence data
            batch_indices: (Variable) dimension batch_size x seq_len with the corresponding indices
        """

        # make a list that decides the order in which we go over the data- this avoids explicit shuffling of data
        order = list(range(data['size']))
        if shuffle:
            random.seed(230)
            random.shuffle(order)

        # one pass over data
        for i in range((data['size'] + 1) // params.batch_size):
            # fetch sentences and indices
            batch_data = [data['data'][idx] for idx in order[i * params.batch_size:(i + 1) * params.batch_size]]
            batch_indices = [data['indices'][idx] for idx in order[i * params.batch_size:(i + 1) * params.batch_size]]

            # since all data are indices, we convert them to torch LongTensors
            batch_data, batch_indices = torch.FloatTensor(batch_data), torch.LongTensor(batch_indices)

            # shift tensors to GPU if available
            if params.cuda:
                batch_data, batch_indices = batch_data.cuda(), batch_indices.cuda()

            # convert them to Variables to record operations in the computational graph
            batch_data, batch_indices = Variable(batch_data), Variable(batch_indices)

            yield batch_data, batch_indices
