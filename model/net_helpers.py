import torch
from torch.autograd import Variable


def get_data_iterator(data, params):
    """
    Returns a generator that yields batch of data without indices. Batch size is params.batch_size. Expires after one
    pass over the data.

    Args:
        data: (dict) contains data which has keys 'data' and 'size'
        params: (Params) hyperparameters of the training process.

    Returns:
        batch_data: (Variable) dimension batch_size x seq_len with the sentence data
    """

    # fetch sentences and indices
    batch_data = data['data'][:]

    # convert data to float tensors
    batch_data = torch.FloatTensor(batch_data)

    # shift tensors to GPU if available
    if params.cuda:
        batch_data = batch_data.cuda()

    # convert them to Variables to record operations in the computational graph
    batch_data = Variable(batch_data)

    yield batch_data
