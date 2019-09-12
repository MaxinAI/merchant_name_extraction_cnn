# Merchant Detection with Pytorch

**Authors**: Anzor Gozalishvili, Levan Tsinadze

## Requirements

We recommend using python3 and a virtual env.
Or alternalively You can use Conda environment.

```
virtualenv -p python3 .env
source .env/bin/activate
pip install -r requirements.txt
```

When you're done working on the project, deactivate the virtual environment with `deactivate`.

## Task

Given a transaction string, predict the indices of merchant name inside a transaction.

```
given transaction: CASTLE ACADEMY LLC 303-663-7300 CO
get merchant indices: 0 14
predicted merchant: CASTLE ACADEMY
```

## Quickstart (~10 min)

__Download the dataset__ `dataset.csv` data/ directory.

__Build the dataset__ Run the following script
```
python build_dataset.py
```
It will extract the sentences and labels from the dataset, split it into
train/val/test and save it in a convenient format for training the model.

*Debug* If you get some errors, check that you downloaded the right file and
saved it in the right directory. If you have issues with encoding, try running
the script with python 2.7.

__Your first experiment__ We created a `base_model` directory under
    the `experiments` directory. It contains a file `params.json` which sets the
    hyperparameters for the experiment. It looks like

```json
{
  "learning_rate": 1e-3,
  "batch_size": 20,
  "num_epochs": 10,
  "save_summary_steps": 100
}

```
For every new experiment, you will need to create a new directory under
`experiments` with a `params.json` file.

__Train__ your experiment. Simply run
```
python train.py --data_dir data/ --model_dir experiments/base_model --restore_file best
```
It will instantiate a model and train it on the training set following the
hyperparameters specified in `params.json`. It will also evaluate some metrics
on the development set.Model will be initialised from scratch if you won't pass
restore_file parameter which can be 'best' of 'last' (saved model pth file names)

__Your first hyperparameters search__ We created a new directory
    `learning_rate` in `experiments` for you. Now, run
```
python search_hyperparams.py --data_dir data/ --parent_dir experiments/learning_rate
```
It will train and evaluate a model with different values of learning rate defined
in `search_hyperparams.py` and create a new directory for each experiment under
`experiments/learning_rate/`.

__Display the results__ of the hyperparameters search in a nice format
```
python synthesize_results.py --parent_dir experiments/learning_rate
```

__Evaluation on the test set__ Once you've run many experiments and selected
    your best model and hyperparameters based on the performance on the development
    set, you can finally evaluate the performance of your model on the test set.
    Run
```
python evaluate.py --data_dir data/ --test_data_dir test --model_dir experiments/base_model --restore_file best --results_file results.csv
```

__Gererate Final Test Set__ If you need to test your model on the data samples that hasn't been
either in train in validation set.
```
python prepare_final_data_for_testing.py --data_dir data --test_data_dir data/final_test
```

__Start Flask API service__ After training you model you can use flask api service for production
```
python flask_api.py --data_dir data --host 0.0.0.0 --port 5000 query_key text --model_dir experiments/base_model --restore_file best
```

__Using FLask API__ You can use Postman to send GET requests
```
http://0.0.0.0:5000?text=12324 MOTOROLA CARD PURCHASE XXXXX4834 DEBIT PIN PURCHASE
```

and you will get such response:
```json
{
"originalString": "12324 MOTOROLA CARD PURCHASE XXXXX4834 DEBIT PIN PURCHASE",
"cleanedString": "MOTOROLA",
"beginIndexConf": 0.9999987483024597,
"endIndexConf": 0.5950449109077454
}
```


## Guidelines for more advanced use

We recommend reading through `train.py` to get a high-level overview of the training loop steps:
- loading the hyperparameters for the experiment (the `params.json`)
- loading the training and validation data
- creating the model, loss_fn and metrics
- training the model for a given number of epochs by calling `train_and_evaluate(...)`

You can then go through `model/data_loader.py` to understand the following steps:
- creating the sentences/indices datasets from the text files
- how the `data_iterator` creates a batch of data and indices

Once you get the high-level idea, depending on your dataset, you might want to modify
- `model/model.py` to change the neural network, loss function and metrics
- `model/data_loader.py` to suit the data loader to your specific needs
- `train.py` for changing the optimizer
- `train.py` and `evaluate.py` for some changes in the model or input require changes here

Once you get something working for your dataset, feel free to edit any part of the code to suit your own needs.

## Resources

- [PyTorch documentation](http://pytorch.org/docs/0.3.0/)
- [Tutorials](http://pytorch.org/tutorials/)
- [PyTorch warm-up](https://github.com/jcjohnson/pytorch-examples)
