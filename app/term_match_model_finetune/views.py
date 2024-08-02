import pandas as pd
import re
import asyncio
from datetime import datetime
from rest_framework import status
from rest_framework.response import Response
from rest_framework.renderers import JSONRenderer
from rest_framework.decorators import authentication_classes, permission_classes
from adrf.views import APIView
from django.core.exceptions import BadRequest
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi

from app.get_sent_pairs.views import generate_term_match_sent_pairs
from app.term_match_model_finetune.models import GetModelFinetune
from app.term_match_model_finetune.serializers import GetModelFinetuneSerializer
from asgiref.sync import sync_to_async
import logging

import glob
import os
from config import SENT_BERT_MODEL_PATH, SENT_EMBEDDINGS_BATCH_SIZE
import random

# list_of_files = glob.glob('data') # * means all if need specific format then *.csv
# latest_file = max(list_of_files, key=os.path.getctime) # e.g. sentence_pairs_2024-03-26_2024-03-28.csv

logger = logging.getLogger(__name__)

sample_dict = {
    'updateStartDate': '2024-03-26',
    'updateEndDate': '2024-03-28'
}

request_schema_dict = {
    'updateStartDate': openapi.Schema(
        type=openapi.TYPE_STRING,
        description='*REQUIRED*. updateStartDate, format: yyyy-mm-dd',
        default=sample_dict['updateStartDate']
    ),
    'updateEndDate': openapi.Schema(
        type=openapi.TYPE_STRING,
        description='*REQUIRED*. updateEndDate, format: yyyy-mm-dd',
        default=sample_dict['updateEndDate']
    )
}

response_schema_dict = {
    "200": openapi.Response(
        description="Success",
        examples={
            "application/json": {
                "message": f"term matching model is fine-tuned with TS-FA sentence pairs dataset retrieved from updated start date {str(sample_dict['updateStartDate'])} to updated end date {str(sample_dict['updateEndDate'])}",
                "numSentPairs": 1234,
                "numManuallyAddedPairs": 546,
                "numSystemSuggestedPairs": 688,
                "outputFineTuneModelPath": f'models/term_match_finetune_model_{sample_dict["updateStartDate"]}_{sample_dict["updateEndDate"]}_sentpairs_200',
                'fineTuneStatus': 'success',
                'duration(min.)': 60
            }
        }
    ),
    "400": openapi.Response(
        description="Error: Bad Request",
        examples={
            "application/json": {
                'finetune_term_match_status': 'failed'
            }
        }
    )
}


def add_section2text(row, doc_type):
    '''
    @param row: pandas.Series object with indices ['text', 'section', 'fa_text', 'fa_section', 'fa_sub_section', 'parent_caption', 'parent_list']
    @param doc_type: either 'TS' or 'FA'
    @return: return a concatenated string of useful features (section, sub-section, caption, list item and main text) as a complete content
    @rtype: str
    '''
    if doc_type == 'FA':
        fa_section = row['faSection']
        fa_sub_section = row['faSubSection']
        text = row['faText']

        # erase section / sub-section when text_element is section / sub-section because text standalone is already a section / sub-section
        # remove leading digit in FA section text, eg. 21.2 Financial Covanents -> Financial Covanents
        text = re.sub('^\d+\.*\d*\.*\d*', '', text)

        # erase section / sub-section that contains word "definition" or "interpretation" because it is meaningless
        if fa_section and re.search('definition|interpretation', fa_section, re.IGNORECASE):
            fa_section = None
        if fa_sub_section and re.search('definition|interpretation', fa_sub_section, re.IGNORECASE):
            fa_sub_section = None

        if fa_sub_section and fa_section and text:
            return fa_section + ' - ' + fa_sub_section + ': ' + text
        elif fa_sub_section and not fa_section and text:
            return fa_sub_section + ': ' + text
        elif not fa_sub_section and fa_section and text:
            return fa_section + ': ' + text
        else:
            return text

    elif doc_type == 'TS':
        section = row['tsTerm']
        text = row['tsText']

        # erase section that contains word "documentation" because it is meaningless
        if section and re.search('documentation', section, re.IGNORECASE):
            section = None
        # if both section and text not null
        if section and text:
            return section + ': ' + text
        else:
            return text


def term_match_finetune(sent_pairs_json, updateStartDate, updateEndDate):
    import pandas as pd
    import numpy as np
    import math
    from app.term_match_model_finetune.training_args import ModelArguments, DataTrainingArguments
    from transformers import HfArgumentParser, TrainingArguments
    from sentence_transformers import models, losses, datasets
    from sentence_transformers import LoggingHandler, SentenceTransformer, util, InputExample
    from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, BinaryClassificationEvaluator
    from datetime import datetime
    import sys
    import os
    import gzip
    import csv
    import random
    import json
    import time

    start = time.time()

    #### Just some code to print debug information to stdout
    # logging.basicConfig(format='%(asctime)s - %(message)s',
    #                     datefmt='%Y-%m-%d %H:%M:%S',
    #                     level=logging.INFO,
    #                     handlers=[LoggingHandler()])
    #### /print debug information to stdout
    ## todo: add json files
    with open('app/term_match_model_finetune/training_config.json', 'r') as f:
        train_config = json.load(f)
    num_of_sent_pairs = len([i for i in sent_pairs_json if i['label'] == 'entailment'])
    num_of_manual_added = len([i for i in sent_pairs_json if i['isManualContent']])
    num_of_system_correct_suggested = len([i for i in sent_pairs_json if not i['isManualContent']])
    model_name_or_path = SENT_BERT_MODEL_PATH  # train_config["model_name"]
    train_batch_size = SENT_EMBEDDINGS_BATCH_SIZE  # train_config["batch_size"]
    max_seq_length = train_config["max_seq_length"]
    num_epochs = train_config["num_train_epochs"]
    learning_rate = train_config["learning_rate"]
    # boc_dataset_path = train_config["train_file"]
    # dataset_name = train_config["dataset_name"]
    # model_save_path = train_config["model_save_path"]
    model_save_path = f"models/term_match_finetune_model_{str(updateStartDate)}_{str(updateEndDate)}_sentpairs_{str(num_of_sent_pairs)}"

    '''
    EmbeddingSimilarityEvaluator: Measure Spearman and Pearson rank correlation between cosine score and gold labels
    BinaryClassificationEvaluator: Measure accuracy with cosine similarity as well as euclidean and Manhattan distance of identifying similar and dissimilar sentences
    '''
    evaluator = EmbeddingSimilarityEvaluator  # BinaryClassificationEvaluator

    # Here we define our SentenceTransformer model
    # model_name_or_path = "thenlper/gte-base"
    word_embedding_model = models.Transformer(model_name_or_path, max_seq_length=max_seq_length)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    # model = SentenceTransformer(model_name_or_path)

    # Read the BoC sentence pair file and create the training dataset
    logger.info(
        f"Read BoC train dataset with {num_of_sent_pairs} sentence-pairs reviewed from {str(updateStartDate)} to {str(updateEndDate)}")

    def add_to_samples(sent1, sent2, label):
        if sent1 not in train_data:
            train_data[sent1] = {'contradiction': set(), 'entailment': set(), 'neutral': set()}
        train_data[sent1][label].add(sent2)

    # df_dataset = pd.read_csv(boc_dataset_path)
    # use the sentence pairs queried by /api/getSentPairs API with update start date and update end date provided
    ''' sample sent_pairs_json: [
        {
            "updateTime": "2024-03-27T19:26:53.000",
            "id": 17469,
            "taskId": 1,
            "indexId": 3,
            "textBlockId": 11,
            "pageId": "2",
            "phraseId": 0,
            "tsTerm": "Borrower",
            "tsText": "Borrower:",
            "identifier": "Sched_12【BANKING (EXPOSURE LIMITS) RULES】",
            "faSection": "BANKING (EXPOSURE LIMITS) RULES",
            "faSubSection": null,
            "faText": "[BORROWER]",
            "isManualContent": false,
            "label": "entailment"
        },
        {
            "updateTime": "2024-03-27T19:26:53.000",
            "id": 17469,
            "taskId": 1,
            "indexId": 3,
            "textBlockId": 11,
            "pageId": "2",
            "phraseId": 0,
            "tsTerm": "Borrower",
            "tsText": "Borrower:",
            "identifier": "Cl_1.1-Borrowings_(a)【DEFINITIONS AND INTERPRETATION - Definitions】",
            "faSection": "DEFINITIONS AND INTERPRETATION",
            "faSubSection": "Definitions",
            "faText": "moneys borrowed and debit balances at banks or other financial institutions;",
            "isManualContent": false,
            "label": "contradiction"
        }
        ]
     '''
    df_dataset = pd.DataFrame(sent_pairs_json)
    logger.info("no of entailment: {}".format(len(df_dataset[df_dataset.label == 'entailment'])))
    logger.info('no of contradiction: {}'.format(len(df_dataset[df_dataset.label == 'contradiction'])))

    df_dataset['score'] = df_dataset['label'].apply(
        lambda i: random.uniform(0.999, 1) if i == 'entailment' else 0.3
    )
    df_dataset['sentence1'] = df_dataset.apply(lambda row: add_section2text(row, 'TS'),
                                               axis=1)  # sentence1 is tsTerm: tsText
    df_dataset['sentence2'] = df_dataset.apply(lambda row: add_section2text(row, 'FA'),
                                               axis=1)  # sentence2 is faSection - faSubSection: faText

    # df_train = df_dataset[df_dataset.split=='train']
    df_train = df_dataset
    logger.info("length of training dataset: {}".format(len(df_train)))
    train_samples = []
    train_data = {}
    for idx, row in df_train.iterrows():
        sent1 = row['sentence1'].strip()
        sent2 = row['sentence2'].strip()
        label = row['label']
        if sent1 not in train_data:
            train_data[sent1] = {'contradiction': set(), 'entailment': set(), 'neutral': set()}
        train_data[sent1][label].add(sent2)

        if sent2 not in train_data:
            train_data[sent2] = {'contradiction': set(), 'entailment': set(), 'neutral': set()}
        train_data[sent2][label].add(sent1)
        # add_to_samples(sent1, sent2, row['label'])
        # add_to_samples(sent2, sent1, row['label'])

    logger.info("train_data: {}".format(len(train_data)))
    for sent1, others in train_data.items():
        if len(others['entailment']) > 0 and len(others['contradiction']) > 0:
            train_samples.append(InputExample(texts=[sent1, random.choice(list(others['entailment']))]))
            train_samples.append(InputExample(texts=[random.choice(list(others['entailment'])), sent1]))
    train_samples = train_samples * 100
    train_dataloader = datasets.NoDuplicatesDataLoader(train_samples, batch_size=train_batch_size)

    logger.info("Read dev dataset")
    # df_dev = df_dataset[df_dataset.split=='test']
    df_dev = df_dataset
    dev_samples = []

    for idx, row in df_dev.iterrows():
        score = row['score']
        dev_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))

    dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, batch_size=train_batch_size,
                                                                     name='boc-dev')
    # Configure the training
    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
    logger.info("Warmup-steps: {}".format(warmup_steps))
    logger.info("Train samples: {}".format(len(train_samples)))
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # Train the model
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              evaluator=dev_evaluator,
              epochs=num_epochs,
              optimizer_params={'lr': learning_rate},
              evaluation_steps=int(len(train_dataloader) * 0.1),
              warmup_steps=warmup_steps,
              output_path=model_save_path,
              # use_amp=True
              use_amp=False  # Set to True, if your GPU supports FP16 operations
              )

    end = time.time()
    duration = end - start

    results = {
        "message": f"term matching model is fine-tuned with TS-FA sentence pairs dataset retrieved from updated start date {str(updateStartDate)} to updated end date {str(updateEndDate)}",
        "numSentPairs": num_of_sent_pairs,
        "numManuallyAddedPairs": num_of_manual_added,
        "numSystemSuggestedPairs": num_of_system_correct_suggested,
        "outputFineTuneModelPath": model_save_path,
        'fineTuneStatus': 'success',
        'duration(min.)': round(duration / 60, 2)

    }
    return results


class GetModelFinetuneView(APIView):
    queryset = GetModelFinetune.objects.all()
    serializer_class = GetModelFinetuneSerializer

    @authentication_classes([])
    @permission_classes([])
    @swagger_auto_schema(
        operation_summary='fine-tune term match models with retrieved sentence pairs from database within a range of reviewed dates',
        operation_description='After user made their feedback on term matching results, the term match model can be fine-tuned with the collected term match results.',
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties=request_schema_dict
        ),
        responses=response_schema_dict
    )
    async def post(self, request):
        data = request.data
        try:
            updateStartDate = data.get("updateStartDate")
        except:
            raise BadRequest(
                f'updateStartDate is compulsory field name. Please check if updateStartDate is provided in the request')
        try:
            updateEndDate = data.get("updateEndDate")
        except:
            raise BadRequest(
                f'updateEndDate is compulsory field name. Please check if updateEndDate is provided in the request')

        sent_pairs_json = await generate_term_match_sent_pairs(data)
        sent_pairs_data = sent_pairs_json.data
        results = await sync_to_async(term_match_finetune)(sent_pairs_data, updateStartDate, updateEndDate)
        response = Response(results, status=status.HTTP_200_OK)
        response.accepted_renderer = JSONRenderer()
        response.accepted_media_type = "application/json"
        response.renderer_context = {}
        response.render()
        return response
