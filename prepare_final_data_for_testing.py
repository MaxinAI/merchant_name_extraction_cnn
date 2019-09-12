import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data', help="Directory containing the data set")
parser.add_argument('--test_data_dir', default='data/final_test_set',
                    help="Directory containing the data set for final testing of model")


def get_data_set(filename):
    with open(os.path.join(filename, 'sentences.txt')) as sent:
        sentences = [line.rstrip('\n') for line in sent]

    indices = []
    with open(os.path.join(filename, 'indices.txt')) as ind:
        for line in ind.readlines():
            start, end = line.split()
            indices.append((int(start), int(end)))

    merchants = set()
    for sentence, index in zip(sentences, indices):
        merchants.add(sentence[index[0]:index[1]])

    return sentences, indices, merchants


def get_data_set_with_merchants(sentences, indices, merchants):
    chosen_sentences = []
    chosen_indices = []

    for sentence, index in zip(sentences, indices):
        for merchant in merchants:
            if sentence.find(merchant) != -1 and sentence not in chosen_sentences:
                chosen_sentences.append(sentence)
                chosen_indices.append(index)

    return chosen_sentences, chosen_indices


def save_line_separated_data(data, file, tuples=False):
    with open(file, 'w') as f:
        if not tuples:
            for line in data:
                f.write(line + '\n')
        else:
            for line in data:
                f.write(' '.join(str(i) for i in line) + '\n')


if __name__ == '__main__':
    args = parser.parse_args()
    data_dir = args.data_dir
    test_data_dir = args.test_data_dir

    if not os.path.exists(data_dir):
        raise FileNotFoundError

    test = get_data_set(os.path.join(data_dir, 'test'))
    val = get_data_set(os.path.join(data_dir, 'val'))
    train = get_data_set(os.path.join(data_dir, 'train'))

    rare_merchants = list(test[2] - set(list(val[2]) + list(train[2])))

    rare_sentences, rare_indices = get_data_set_with_merchants(test[0], test[1], rare_merchants)

    if not os.path.exists(test_data_dir):
        os.mkdir(test_data_dir)

    save_line_separated_data(rare_merchants, os.path.join(test_data_dir, 'merchants.txt'))
    save_line_separated_data(rare_sentences, os.path.join(test_data_dir, 'sentences.txt'))
    save_line_separated_data(rare_indices, os.path.join(test_data_dir, 'indices.txt'), True)
