"""Read, split and save the Merchant dataset for our model"""

import argparse
import csv
import json
import os
import random
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data', help="CSV file containing Dataset")
parser.add_argument('--augment_mult', type=int, default=0, help="Number of copies to generate for each sample ")
parser.add_argument('--max_data_size', type=int, default=10000, help="Maximum number of samples to load from csv")


def find_all(a_str, sub, overlapping=False):
    """
    Finds all occurencies of substrings (Non Overlapping) in given string

    Args:
        a_str: (str) input string
        sub: (str) substring
        overlapping: (boolean) overlapping substring search

    Yields:
        (int) start index of substring in a given string

    """
    start = 0
    increment = 1

    if not overlapping:
        increment = len(sub)

    while True:
        start = a_str.find(sub, start)
        if start == -1:
            return
        yield start
        start += increment


def get_merchant_indices_in_sentence(sentence, merchant):
    """
    Given transaction string and merchant string, returns start end end indices of merchant in transaction string.

    Args:
        sentence: (string) transaction string. (example: )
        merchant: (string) merchant name. (example: )

    Returns:
        sentence: (string) converted to upper strings
        start: (int) merchant string start index
        end: (int) merchant string end index

    Examples:
        sentences, start, end = get_sentence_indices("Target 00014423 WATERTOWN MA","target")

        sentences: "TARGET 00014423 WATERTOWN MA"
        start: 0
        end: 6
    """

    sentence = sentence.upper()
    merchant = merchant.upper()

    start = -1
    end = -1

    idx = sentence.find(merchant)
    if idx != -1:
        start = idx
        end = start + len(merchant)

    return sentence, start, end


def load_dataset(path_csv, num_samples):
    """
    Loads dataset into memory from csv file

    Args:
        path_csv: (str) csv file path
        num_samples: (int) how many samples from dataset we need to load

    Returns:
        dataset: (list) containing list of tuples like:(sentence, st_idx, end_idx)

    """

    use_python3 = sys.version_info[0] >= 3
    with (open(path_csv, encoding='utf-8') if use_python3 else open(path_csv)) as f:
        csv_file = csv.reader(f, delimiter=',')
        dataset = []

        for idx, row in enumerate(csv_file):
            if idx == num_samples:
                break
            if idx == 0:
                continue
            _, sentence, st_idx, end_idx = row
            st_idx = int(st_idx)
            end_idx = int(end_idx)
            if 0 < len(sentence) < 300:
                dataset.append((sentence, st_idx, end_idx))

    return dataset


def slide_merchant_in_transaction(trans_1):
    """
    Slides merchant substring in whole transaction string between space separated words.

    Args:
        trans_1: (tuple) like:(transaction, start_idx, end_idx)

    Returns:
        transactions: (list) list of generated tuples like:(transaction, start_idx, end_idx)

    """
    (transaction, start, end) = trans_1
    transactions = []

    tmp_transaction = " ".join([transaction[:start].strip(), transaction[end:].strip()])

    spaces_idx = list(find_all(tmp_transaction, " "))

    # remove index of merchant separator space in original string to avoid copies
    if start in spaces_idx:
        spaces_idx.remove(start)

    for idx in spaces_idx:
        new_transaction = " ".join([tmp_transaction[:idx], transaction[start:end], tmp_transaction[idx:]])
        st_idx = idx + 1
        end_idx = idx + + end - start + 1
        transactions.append((new_transaction, st_idx, end_idx))

    return transactions


def swap_merchants_in_transactions(trans_1, trans_2):
    """
    Given two samples like (transaction, start_idx, end_idx) and it will exchange merchant names between these samples.

    Args:
        trans_1: (tuple) like:(transaction, start_idx, end_idx)
        trans_2: (tuple) like:(transaction, start_idx, end_idx)

    Returns:
        (tuple): like: (transaction, start_idx, end_idx)
        (tuple): like: (transaction, start_idx, end_idx)

    """

    (transaction_1, start_1, end_1) = trans_1
    (transaction_2, start_2, end_2) = trans_2

    transaction_3 = transaction_1[:start_1] + transaction_2[start_2: end_2] + transaction_1[end_1:]
    start_3, end_3 = start_1, start_1 + end_2 - start_2

    transaction_4 = transaction_2[:start_2] + transaction_1[start_1: end_1] + transaction_2[end_2:]
    start_4, end_4 = start_2, start_2 + end_1 - start_1

    return (transaction_3, start_3, end_3), (transaction_4, start_4, end_4)


def pad_transactions(*args):
    """
     Given several samples like (transaction, start_idx, end_idx), it will choose top 2 with higher number of start_idx
     and exchange non merchant prefix and suffix for padding each other.

    Args:
        *args: *(tuple) given multiple samples like (transaction, start_idx, end_idx)

    Returns:
        (tuple): like: (transaction, start_idx, end_idx)
        (tuple): like: (transaction, start_idx, end_idx)

    Example:
        '''
        In[1]: pad_transaction((("USAA.COM PMT - THANK YOU SAN ANTONIO TX", 15, 24),
                                ("APL* ITUNES.COM/BILL 866-712-7753 CA", 5, 11)))

        Out[1]:
        (('USAA.COM PMT - APL* ITUNES.COM/BILL 866-712-7753 CA SAN ANTONIO TX', 20, 26),
        ('APL* USAA.COM PMT - THANK YOU SAN ANTONIO TX.COM/BILL 866-712-7753 CA', 20, 29))
        '''

    """

    trans_list = list(*args)
    trans_list = list(set(trans_list))
    trans_list.sort(key=lambda x: int(x[1]), reverse=True)

    trans_1 = trans_list[0]
    trans_2 = trans_list[1]

    (transaction_1, start_1, end_1) = trans_1
    (transaction_2, start_2, end_2) = trans_2

    pad_1 = None
    pad_2 = None

    if start_1 > 0:
        transaction_3 = str(transaction_1[:start_1] + transaction_2 + transaction_1[end_1:])
        start_3, end_3 = start_1 + start_2, start_1 + end_2
        pad_1 = (transaction_3, start_3, end_3)
    if start_2 > 0:
        transaction_4 = str(transaction_2[:start_2] + transaction_1 + transaction_2[end_2:])
        start_4, end_4 = start_2 + start_1, start_2 + end_1
        pad_2 = (transaction_4, start_4, end_4)

    return pad_1, pad_2


def augment_dataset(dataset, augment_mult):
    """
    For each sample in dataset it will do swap and pad operations in augment_mult times.

    Args:
        dataset: (list) list of tuples like: [(transaction, start_idx, end_idx), ..... , ]
        augment_mult: (int) number of augmentation sequence to be done for each sample in dataset.

    Returns:
        dataset: (list) list of tuples like: [(transaction, start_idx, end_idx), ..... , ]
                            passed dataset appended with augmented samples .
    """
    augmented_dataset = []
    size = len(dataset)
    for idx in range(size):
        for i in range(augment_mult):

            slided_transactions = slide_merchant_in_transaction(dataset[idx])
            for slided_transaction in slided_transactions:
                augmented_dataset.append(slided_transaction)

            # swap merchant strings between transactions
            random_idx = random.randint(0, size - 1)
            augmented_transactions = swap_merchants_in_transactions(dataset[idx], dataset[random_idx])

            for augmented_transaction in augmented_transactions:
                if augmented_transaction is not None:
                    augmented_dataset.append(augmented_transaction)

            # pad first given transaction with another transactions content without merchant name
            # use just 32 examples for this operation because dataset contains 74% of transactions where merchant name
            # is at position 0.
            pad_indices = [random.randint(0, size - 1) for x in range(31)]
            pad_indices.append(idx)
            padded_transactions = pad_transactions((dataset[i] for i in pad_indices))

            for padded_transaction in padded_transactions:
                if padded_transaction is not None:
                    augmented_dataset.append(padded_transaction)

    dataset += augmented_dataset

    return dataset


def shuffle_dataset(dataset):
    """
    Shuffle samples in dataset.

    Args:
        dataset: (list) list of tuples like: [(transaction, start_idx, end_idx), ..... , ]

    Returns:
        dataset: (list) list of tuples like: [,....., (transaction, start_idx, end_idx), ..... , ]

    """

    shuffled_indices = list(range(len(dataset)))
    random.shuffle(shuffled_indices)
    dataset = [dataset[index] for index in shuffled_indices]

    return dataset


def save_dataset(dataset, save_dir):
    """
    Writes sentences.txt and indices.txt files in save_dir from dataset

    Args:
        dataset: (list) list of tuples like: [(transaction, start_idx, end_idx), ..... , ]
        save_dir: (string) directory where dataset will be saved.

    """

    print(f'Saving in {save_dir}...')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir, 'sentences.txt'), 'w', encoding='utf-8') as file_sentences:
        with open(os.path.join(save_dir, 'indices.txt'), 'w', encoding='utf-8') as file_indices:
            for sentence, st_index, end_index in dataset:
                file_sentences.write(f'{sentence}\n')
                file_indices.write(f'{str(st_index)} {str(end_index)}\n')

    print("- done.")


def save_dict_to_json(d, json_path):
    """
    Saves dictionary to json file

    Args:
        d: (dict)
        json_path: (string) path to json file

    """
    with open(json_path, 'w') as f:
        d = {k: v for k, v in d.items()}
        json.dump(d, f, indent=4)


if __name__ == "__main__":
    args = parser.parse_args()
    data_dir = args.data_dir
    max_data_size = args.max_data_size
    augment_mult = args.augment_mult
    augment = (augment_mult > 0)
    dataset_path = os.path.join(data_dir, 'dataset.csv')

    # Check that the dataset exists (you need to make sure you haven't downloaded the `ner.csv`)
    msg = f'{dataset_path} file not found. Make sure you have downloaded the right dataset'
    assert os.path.isfile(dataset_path), msg

    # Load the dataset into memory
    print("Loading dataset into memory...")
    dataset = load_dataset(dataset_path, max_data_size)
    print("- done.")

    if augment:
        print(f'Doing augmentation operations {augment_mult} times for each sample')
        augment_dataset(dataset, augment_mult)
        print('- Done.')

    # shuffle dataset
    dataset = shuffle_dataset(dataset)
    dataset = dataset[:max_data_size]

    # Split the dataset into train, val and split (dummy split with no shuffle)
    train_dataset = dataset[:int(0.7 * len(dataset))]
    val_dataset = dataset[int(0.7 * len(dataset)): int(0.85 * len(dataset))]
    test_dataset = dataset[int(0.85 * len(dataset)):]

    # Save the datasets to files
    save_dataset(train_dataset, 'data/train')
    save_dataset(val_dataset, 'data/val')
    save_dataset(test_dataset, 'data/test')

    # Save datasets properties in json file
    sizes = {
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'test_size': len(test_dataset),
    }

    save_dict_to_json(sizes, os.path.join(args.data_dir, 'dataset_params.json'))
