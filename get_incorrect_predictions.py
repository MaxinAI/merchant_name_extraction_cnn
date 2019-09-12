import pandas as pd
import argparse
import os
from merchant_extractor import MerchantExtractor
from model.data_loader import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data', help="directory containing the dataset")
parser.add_argument('--splits', default=['train', 'val', 'test','final_test_set'], help="dataset split names to use for testing")
parser.add_argument('--output_data', default='bad_samples.csv',
                    help="output test results data csv file containing samples on which model predicted incorrectly")
parser.add_argument('--model_dir', default='experiments/base_model', help="directory containing params.json")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")
parser.add_argument('--batch_size', default=32, type=int,
                    help='size of the batch for processing several transaction strings together with nn model')

if __name__ == '__main__':
    args = parser.parse_args()

    data_loader = DataLoader(args.data_dir, None)
    data = data_loader.load_data(args.splits, args.data_dir)
    print(f'- data loaded from data splits {str(args.splits)}...')

    full_data = {'sentences': [], 'data': [], 'indices': []}
    for split in args.splits:
        full_data['sentences'].extend(data[split]['sentences'])
        full_data['data'].extend(data[split]['data'])
        full_data['indices'].extend(data[split]['indices'])

    string_cleaner = MerchantExtractor(args.data_dir, args.model_dir, args.restore_file, args.batch_size)

    responses = string_cleaner.get_merchant(full_data['sentences'])
    print(f'- predictions are made...')

    bad_samples = {'transaction': [], 'merchant': [], 'model_prediction': [], 'start_idx_conf': [], 'end_idx_conf': []}

    for response, indices in zip(responses, full_data['indices']):
        if response['cleanedString'] != response['originalString'][indices[0]:indices[1]]:
            bad_samples['transaction'].append(response['originalString'])
            bad_samples['merchant'].append(response['originalString'][indices[0]:indices[1]])
            bad_samples['model_prediction'].append(response['cleanedString'])
            bad_samples['start_idx_conf'].append(response['beginIndexConf'])
            bad_samples['end_idx_conf'].append(response['endIndexConf'])

    print(f'- incorrectly predicted samples are chosen...')

    bad_samples_df = pd.DataFrame.from_dict(bad_samples)

    bad_samples_df.to_csv(os.path.join(args.data_dir, args.output_data))
    print(f'- data saved to {args.output_data} file...')
