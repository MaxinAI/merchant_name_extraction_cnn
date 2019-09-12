import numpy as np

from model.data_loader import DataLoader


def prediction(model, data_iterator):
    """
    Evaluate Model on Given Data.

    Args:
        model: (torch.nn.Module) the neural network
        data_iterator: (generator) a generator that generates batches of data without indices

    """
    # set model to evaluation mode
    model.eval()

    # fetch single line
    data_batch = next(data_iterator)

    # compute model output
    output_batch = model(data_batch)

    # extract data from torch Variable, move to cpu, convert to numpy arrays
    output_batch = output_batch.data.cpu().numpy()

    return output_batch


def predict_indices(outputs, batch_size):
    """
    Show predicted indices given outputs from the model.

    Args:
        outputs: (Variable) dimension batch_size*seq_len x num_tags - log softmax output of the model
        batch_size: number of sentences in batch to predict indices

    Returns:
        indices: True indices of predicted sentences of numpy int64 data type

    """
    indices = np.argmax(outputs, axis=1)

    converted_indices = indices.reshape((batch_size, -1))

    return converted_indices


def get_indices_confidences(outputs, batch_size):
    """
    Show confidences of predicted indices given outputs from the model.

    Args:
        outputs: (Variable) dimension batch_size*seq_len x num_tags - log softmax output of the model
        batch_size: number of sentences in batch to predict indices

    Returns:
        indices: True indices of predicted sentences of numpy int64 data type

    """
    confidences = np.max(outputs, axis=1)

    converted_confidences = confidences.reshape((batch_size, -1))

    return converted_confidences


def get_predictions_and_confidences(outputs, batch_size):
    """
    Predict indices given transaction string and its confidences.

    Args:
        outputs: (Variable) dimension batch_size*seq_len x num_tags - log softmax output of the model
        batch_size: number of sentences in batch to predict indices

    Returns:
        results: (list) list of predicted indices amd their confidences

    """
    indices = predict_indices(outputs, batch_size)
    confidences = get_indices_confidences(outputs, batch_size)
    results = [[indices, confidences]]

    return results


def get_dataset(sentences):
    """
    Prepare dataset for dataloader fron list of sentences.

    Args:
        sentences: (list) list of sentences

    Returns:
        dataset: (dict) with keys ["data", "size"] containing prepated dataset of given sentences.
    """

    data = []
    dataset = {}

    for sentence in sentences:
        sentence = sentence.upper()
        embedded_sentence = DataLoader.character_embedding(sentence)
        data.append(embedded_sentence)

    dataset['data'] = data
    dataset['size'] = len(data)

    return dataset
