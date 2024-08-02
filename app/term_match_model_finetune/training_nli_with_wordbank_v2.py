"""
The system trains BERT (or any other transformer model like RoBERTa, DistilBERT etc.) on the SNLI + MultiNLI (AllNLI) dataset
with MultipleNegativesRankingLoss. Entailnments are poisitive pairs and the contradiction on AllNLI dataset is added as a hard negative.
At every 10% training steps, the model is evaluated on the STS benchmark dataset

Usage:
python training_nli_v2.py

OR
python training_nli_v2.py pretrained_transformer_model_name
"""
import pandas as pd
import numpy as np
import math
from training_args import ModelArguments, DataTrainingArguments
from transformers import HfArgumentParser, TrainingArguments
from sentence_transformers import models, losses, datasets
from sentence_transformers import LoggingHandler, SentenceTransformer, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, BinaryClassificationEvaluator
import logging
from datetime import datetime
import sys
import os
import gzip
import csv
import random

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])
    #### /print debug information to stdout
    
    model_name = model_args.model_name
    train_batch_size = training_args.per_device_train_batch_size
    max_seq_length = data_args.max_seq_length
    num_epochs = int(training_args.num_train_epochs)
    learning_rate = training_args.learning_rate
    do_train = training_args.do_train
    do_test = training_args.do_eval
    task_name = data_args.task_name
    dataset_name = data_args.dataset_name
    
    boc_dataset_folder = 'data/sentence_pairs'
    list_of_files = os.listdir(boc_dataset_folder)
    if list_of_files:
        latest_file = max(list_of_files, key=os.path.getctime)
        boc_dataset_path = os.path.join(boc_dataset_folder, latest_file)
        model_save_path = training_args.output_dir + re.sub('.csv', '', latest_file)
    else:
        boc_dataset_path = data_args.train_file
        model_save_path = training_args.output_dir
    df_dataset = pd.read_csv(boc_dataset_path)

    '''
    EmbeddingSimilarityEvaluator: Measure Spearman and Pearson rank correlation between cosine score and gold labels
    BinaryClassificationEvaluator: Measure accuracy with cosine similarity as well as euclidean and Manhattan distance of identifying similar and dissimilar sentences
    '''
    evaluator = EmbeddingSimilarityEvaluator # BinaryClassificationEvaluator

    # Here we define our SentenceTransformer model
    word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    # Check if dataset exsist. If not, download and extract  it
    # nli_dataset_path = 'data/AllNLI.tsv.gz'
    # sts_dataset_path = 'data/stsbenchmark.tsv.gz'

    # if not os.path.exists(nli_dataset_path):
    #     util.http_get('https://sbert.net/datasets/AllNLI.tsv.gz', nli_dataset_path)

    # if not os.path.exists(sts_dataset_path):
    #     util.http_get('https://sbert.net/datasets/stsbenchmark.tsv.gz', sts_dataset_path)


    # Read the BoC sentence pair file and create the training dataset
    logging.info("Read BoC sentence pair train dataset")

    def add_to_samples(sent1, sent2, label):
        if sent1 not in train_data:
            train_data[sent1] = {'contradiction': set(), 'entailment': set(), 'neutral': set()}
        train_data[sent1][label].add(sent2)
        
    df_dataset['ts_text'] = df_dataset['ts_text'].astype(str)
    df_dataset['fa_text'] = df_dataset['fa_text'].astype(str)
    df_dataset['sentence1'] = df_dataset['ts_text']
    df_dataset['sentence2'] = df_dataset['fa_text']

    # df_train = df_dataset[df_dataset.split=='train']
    df_train = df_dataset
    print('length of training dataset: ', len(df_train))
    train_samples = []
    if task_name == 'nli':
        train_data = {}
        for idx, row in df_train.iterrows():
            sent1 = row['sentence1'].strip()
            sent2 = row['sentence2'].strip()

            add_to_samples(sent1, sent2, row['label'])
            add_to_samples(sent2, sent1, row['label']) 
        for sent1, others in train_data.items():
            if len(others['entailment']) > 0 and len(others['contradiction']) > 0:
                train_samples.append(InputExample(texts=[sent1, random.choice(list(others['entailment']))]))
                train_samples.append(InputExample(texts=[random.choice(list(others['entailment'])), sent1]))
    elif task_name == 'sts':
        for idx, row in df_train.iterrows():
            # score = random.uniform(0.999, 1)
            score = row['score']
            train_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))

    # Special data loader that avoid duplicates within a batch
    train_dataloader = datasets.NoDuplicatesDataLoader(train_samples, batch_size=train_batch_size)
    
    
    #Read STSbenchmark dataset and use it as development set
    # logging.info("Read STSbenchmark dev dataset")
    logging.info("Read dev dataset")
    df_dev = df_dataset[df_dataset.split=='test']
    print('length of dev dataset: ', len(df_dev))
    dev_samples = []
  
    for idx, row in df_dev.iterrows():
        # score = random.uniform(0.999, 1)
        score = row['score']
        dev_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))
    
    dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, batch_size=train_batch_size, name='boc-dev')
    # Configure the training
    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
    logging.info("Warmup-steps: {}".format(warmup_steps))


    logging.info("Train samples: {}".format(len(train_samples)))
    if do_train:
        # Special data loader that avoid duplicates within a batch
        train_dataloader = datasets.NoDuplicatesDataLoader(train_samples, batch_size=train_batch_size)
        # Our training loss
        if task_name == 'nli':
            train_loss = losses.MultipleNegativesRankingLoss(model)
        elif task_name == 'sts':
            train_loss = losses.CosineSimilarityLoss(model=model)

        # dev_evaluator = evaluator.from_input_examples(dev_samples, batch_size=train_batch_size, name='boc-dev')

        # Configure the training
        warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
        logging.info("Warmup-steps: {}".format(warmup_steps))


        # Train the model
        model.fit(train_objectives=[(train_dataloader, train_loss)],
                evaluator=dev_evaluator,
                epochs=num_epochs,
                optimizer_params={'lr':learning_rate},
                evaluation_steps=int(len(train_dataloader)*0.1),
                warmup_steps=warmup_steps,
                output_path=model_save_path,
                # use_amp=True
                use_amp=False          #Set to True, if your GPU supports FP16 operations
                )


    ##############################################################################
    #
    # Load the stored model and evaluate its performance on BOC test dataset
    #
    ##############################################################################

    # if do_test:
    #     model = SentenceTransformer(model_save_path)
    #     # model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    #     test_evaluator = evaluator.from_input_examples(test_samples, batch_size=train_batch_size, name='boc-test')
    #     test_evaluator(model, output_path=model_save_path)


    #     ##############################################################################
    #     #
    #     # Load the stored model and evaluate its performance on STS benchmark dataset
    #     #
    #     ##############################################################################

    #     sts_test_samples = []
    #     with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
    #         reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    #         for row in reader:
    #             if row['split'] == 'test':
    #                 score = float(row['score']) / 5.0 #Normalize score to range 0 ... 1
    #                 if score >= 0.5:
    #                     score = 1
    #                 if score < 0.5:
    #                     score = 0    
    #                 sts_test_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))

    #     model = SentenceTransformer(model_save_path)
    #     test_evaluator = evaluator.from_input_examples(sts_test_samples, batch_size=train_batch_size, name='sts-test')
    #     test_evaluator(model, output_path=model_save_path)
    

if __name__ == "__main__":
    main()