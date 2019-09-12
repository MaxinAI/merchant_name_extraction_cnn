import os

import model.net as net
import utils
from flask_api_helpers import *
from model.net_helpers import *


class MerchantExtractor(object):
    """
    Class that handles trained nn model predictions. It gets data directory, model directory and model restore file
    name like: ['best', 'last']
    """

    def __init__(self, data_dir, model_dir, restore_file, batch_size):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.restore_file = restore_file
        self.batch_size = batch_size

        json_path = os.path.join(self.model_dir, 'params.json')
        assert os.path.isfile(json_path), f'No json configuration file found at {json_path}'
        self.params = utils.Params(json_path)

        # use GPU if available
        self.params.cuda = torch.cuda.is_available()  # use GPU is available

        # Set the random seed for reproducible experiments
        torch.manual_seed(230)
        if self.params.cuda:
            torch.cuda.manual_seed(230)

        self.model = net.MerchantNet().cuda() if self.params.cuda else net.MerchantNet()

        utils.load_checkpoint(os.path.join(self.model_dir, self.restore_file + '.pth.tar'), self.model)

    def get_merchant(self, text):
        """
        Takes transaction string and finds merchant name inside. It uses trained nn model to predict start and end
        indices of merchant inside.
        Args:
            text: (str) transaction string

        Returns:
            results: (dict) contains original string, merchant string and probabilities/confidences of start and end
            indices of predicted merchant string.

        """
        if text is None or (not isinstance(text, str) and not isinstance(text, list)):
            results = {'originalString': '', 'merchantString': '',
                       'beginIndexConf': '', 'endIndexConf': ''}
            return results

        if isinstance(text, str):
            text = [text]

        all_results = []
        num_batches = int(len(text) / self.batch_size) + int(len(text) % self.batch_size > 0)
        for batch_idx in range(num_batches):

            sentences = text[self.batch_size * batch_idx:self.batch_size * (batch_idx + 1)]

            curr_batch_size = len(sentences)

            data = get_dataset(sentences)

            data_iterator = get_data_iterator(data, self.params)

            predictions = prediction(self.model, data_iterator)

            preds_and_confs = get_predictions_and_confidences(predictions, curr_batch_size)

            indices, confs = preds_and_confs[0][0], preds_and_confs[0][1]

            for index, confidence, sentence in zip(indices, confs, sentences):
                sentence, index = self.correct_merchant_indices(sentence, index)
                result = {'originalString': str(sentence), 'merchantString': str(sentence[index[0]:index[1]]),
                          'beginIndexConf': float(confidence[0]), 'endIndexConf': float(confidence[1])}

                all_results.append(result)

        return all_results

    def correct_merchant_indices(self, transaction, indices):
        """
        Correct the indices of detected merchant if it splits space separated tokens in two which can cause detected
        merchants not to be fully included.
        Args:
            transaction: (str) transaction string
            indices: (tuple) start and end indices of predicted merchant string inside the transaction

        Returns:
        transaction: (str) same transaction string as input
        indices: (tuple) corrected start and end indices of predicted merchant string inside the transaction
        """
        start, end = indices
        if end <= start:
            return transaction, indices

        idx = transaction.rfind(' ', 0, start)
        if idx != -1:
            start = int(idx + 1)
        else:
            start = 0

        idx = transaction.find(' ', end, -1)
        if idx != -1:
            end = idx
        else:
            end = len(transaction) - 1
        indices = (start, end)

        return transaction, indices
