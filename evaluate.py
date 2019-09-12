"""Evaluates the model"""

import logging
import os

import argparse
import numpy as np
import pandas as pd
import torch

import model.net as net
import utils
from model.data_loader import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data', help="Directory containing the dataset")
parser.add_argument('--test_data_dir', default='test', help="Directory containing the dataset for testing")
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")
parser.add_argument('--results_file', default='results.csv', help='File name to csv file of results')


def evaluate(model, loss_fn, data_iterator, metrics, params, num_steps, training=False):
    """
    Evaluates model on given data set.

    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: (function) a function that takes batch_output and batch_labels and computes the loss for the batch
        data_iterator: (generator) a generator that generates batches of data and labels
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
        training: (bool) stop calculation of evaluations for training speedup

    Returns:
        metrics_mean: (float)

    """

    # set model to evaluation mode
    model.eval()

    # summary for current eval loop
    summ = []

    full_results = []
    # compute metrics over the dataset
    for idx in range(num_steps):
        # fetch the next evaluation batch
        data_batch, labels_batch = next(data_iterator)

        # compute model output
        output_batch = model(data_batch)
        loss = loss_fn(output_batch, labels_batch)

        # extract data from torch Variable, move to cpu, convert to numpy arrays
        output_batch = output_batch.data.cpu().numpy()
        labels_batch = labels_batch.data.cpu().numpy()

        # compute all metrics on this batch
        summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                         for metric in metrics}
        if not training:
            full_results.extend(net.get_indices_and_confidences(output_batch, labels_batch))

        summary_batch['loss'] = loss.data[0]
        summ.append(summary_batch)
        if not training:
            print(f'Step: {idx+1}/{num_steps} Batch_Size: {data_batch.shape[0]}')

    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join(f'{k}: {v:05.3f}' for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)
    return metrics_mean, full_results


def save_results_to_files(data, results, results_file):
    """
    Save results to csv file. Results include Transaction strings, real indices, predicted indices, confidences of
    predicted indices, predicted merchants and real merchants.
    Args:
        data: (dict) data file with keys ['sentences', 'data'. 'indices', 'size']
        results: (list) all list of per batch results from model evaluation on given data set.
        results_file: (str) filename where to save csv results

    """
    # here we have list of batches which is inconvenient for iterating through samples
    # all the samples will be saved in whole list which will be easy for iteration
    flat_res = []
    for batch_res in results:
        indices, predicted_indices, confidences = batch_res[0], batch_res[1], batch_res[2]
        for a, b, c in zip(indices, predicted_indices, confidences):
            flat_res.append((a, b, c))

    sents = data['sentences']

    # define empty dictionary with headers to create pandas dataframe using it
    results_dict = {'Raw_Transaction_String': [], 'Start_Index_Real': [], 'End_Index_Real': [],
                    'Start_Index_Predicted': [],
                    'End_Index_Predicted': [], 'Start_Index_Predicted_Confidence': [],
                    'End_Index_Predicted_Confidence': [],
                    'Merchant_Predicted': [], 'Merchant_Real': []}

    # process each sample on one iteration
    for trans, res in zip(sents, flat_res):
        ind, pred, conf = res
        results_dict['Raw_Transaction_String'].append(trans)
        results_dict['Start_Index_Real'].append(str(ind[0]))
        results_dict['End_Index_Real'].append(str(ind[1]))
        results_dict['Start_Index_Predicted'].append(str(pred[0]))
        results_dict['End_Index_Predicted'].append(str(pred[1]))
        results_dict['Start_Index_Predicted_Confidence'].append(str(conf[0]))
        results_dict['End_Index_Predicted_Confidence'].append(str(conf[1]))
        results_dict['Merchant_Predicted'].append(trans[pred[0]:pred[1]])
        results_dict['Merchant_Real'].append(trans[ind[0]:ind[1]])

    # create dataframe from dictionary and then save it to csv file
    df = pd.DataFrame.from_dict(results_dict)

    df.to_csv(results_file)


if __name__ == '__main__':
    """
    Evaluate the model on the test set.
    """
    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), f'No json configuration file found at {json_path}'
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()  # use GPU is available

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda: torch.cuda.manual_seed(230)

    # Get the logger
    utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")

    # load data
    data_loader = DataLoader(args.data_dir, params)
    data = data_loader.load_data([args.test_data_dir], args.data_dir)
    test_data = data[args.test_data_dir]
    print(test_data['size'])

    # specify the test set size
    params.test_size = test_data['size']
    test_data_iterator = data_loader.data_iterator(test_data, params)

    logging.info("- done.")

    # Define the model
    model = net.MerchantNet().cuda() if params.cuda else net.MerchantNet().cpu()

    loss_fn = torch.nn.CrossEntropyLoss()
    metrics = net.metrics

    logging.info("Starting evaluation")

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(args.model_dir, args.restore_file + '.pth.tar'), model)

    # Evaluate
    num_steps = (params.test_size + 1) // params.batch_size
    test_metrics, results = evaluate(model, loss_fn, test_data_iterator, metrics, params, num_steps)

    save_results_to_files(test_data, results, args.results_file)

    save_path = os.path.join(args.model_dir, f'metrics_test_{args.restore_file}.json')
    utils.save_dict_to_json(test_metrics, save_path)
