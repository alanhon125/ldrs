import os
import tqdm
import pandas as pd
from PIL import Image
import re
import numpy as np
import json
import torch
import gc

from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoModelForTokenClassification
)
from datasets import Features, Sequence, ClassLabel, Value, Array2D, Array3D
from datasets import Dataset, load_dataset

from config import *

import pyarrow
import logging

logger = logging.getLogger(__name__)
# logging.basicConfig(
#     filename=LOG_FILEPATH,
#     filemode='a',
#     format='【%(asctime)s】【%(filename)s:%(lineno)d】【%(levelname)-8s】%(message)s',
#     level=os.environ.get("LOGLEVEL", "INFO"),
#     datefmt='%Y-%m-%d %H:%M:%S'
#     )

pyarrow.PyExtensionType.set_auto_load(True)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # Explicitly setting seed to make sure that models created in two processes
    # start from same random weights and biases.
    torch.manual_seed(42)


def cleanup():
    try:
        dist.destroy_process_group()
    except:
        pass

def remove_duplicates(x):
  return list(dict.fromkeys(x))

def filter_invalid_char(my_str):
    # To remove invalid characters from a string
    import re
    my_str = re.sub(' +', ' ', my_str)
    my_str = re.sub('', '', my_str)
    my_str = re.sub(' / ', '/', my_str)
    my_str = re.sub(r' ([\.,;]) ', r'\1 ', my_str)
    my_str = re.sub(r' ([\)\]])', r'\1', my_str)
    my_str = re.sub(r' ([\(\[]) ', r' \1', my_str)
    my_str = re.sub(r'([\u4e00-\u9fff]+)', '', my_str)  # Chinese
    my_str = re.sub(r'([\u2580—\u259f]+)', '•', my_str)  # Block Elements
    my_str = re.sub(r'([\u25a0-\u25ff]+)', '•', my_str)  # geometric shape
    my_str = re.sub(r'([\ue000—\uf8ff]+)', '•', my_str)  # Private Use Area
    my_str = re.sub(r'([§]+)', '▪', my_str)
    my_str = re.sub(r'([Ø]+)','•', my_str)
    my_str = re.sub(r'([\uf06e]+)', '•', my_str)
    my_str = re.sub(r'“', r'"', my_str)
    my_str = re.sub(r'”', r'"', my_str)
    my_str = re.sub(r'’', r"'", my_str)
    my_str = re.sub(r'‘', r"'", my_str)
    # my_str = re.sub(r'([\uff01-\uff5e]+)', '', my_str) # fullwidth ascii variants
    # my_str = re.sub(r'([\u2018\u2019\u201a-\u201d]+)', '', my_str)  # Quotation marks and apostrophe
    return my_str.strip()

def token_classification(data, model_path, fname, gpu_ids="0,1,2,3", batch_size = 16, document_type='esgReport'):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # GPU IDs will be ordered by pci bus IDs
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids # gpu_ids
    os.environ["TOKENIZERS_PARALLELISM"] = "false" # disable the parallelism to avoid any hidden deadlock that would be hard to debug

    filename = fname + '.pdf'
    filename_dict = {'filename': filename}
    gpu_ids = gpu_ids.split(',')

    use_cuda = torch.cuda.is_available()
    device = torch.device(f'cuda:{gpu_ids[0]}' if use_cuda else 'cpu')

    try:
        model_version = re.match(r'\S+(v\d)',model_path).groups()[0]
    except:
        model_version = None


    # accept data input either as local path, dataset repository on the Hub, dictionary or pandas Dataframe object
    if isinstance(data, str):
        if data.endswith('.json'):
            datasets = load_dataset('json', data_files=[data])
            datasets = datasets['train']
        elif data.endswith('.csv'):
            datasets = load_dataset('csv', data_files=[data])
            datasets = datasets['train']
        else:
            try:
                datasets = load_dataset(data)
                datasets = datasets['train']
            except:
                raise Exception("The data path must be end with .json, .csv or a valid dataset repository on the Hub")
    elif isinstance(data, dict):
        datasets = Dataset.from_dict(data)
    elif isinstance(data, list):
        datasets = Dataset.from_pandas(pd.DataFrame(data=data))
    elif isinstance(data, pd.DataFrame):
        
        datasets = Dataset.from_pandas(data)

    column_names = datasets.column_names
    # features = datasets.features
    remove_columns = column_names

    datasets_dict = datasets.to_dict()
    input_dict = {
             'id':[datasets_dict['id'][idx] for idx, sublist in enumerate(datasets_dict['tokens']) for i in sublist],
             'tokens': [filter_invalid_char(i) for sublist in datasets_dict['tokens'] for i in sublist],
             'bboxes':[tuple(i) for sublist in datasets_dict['bboxes'] for i in sublist],
             'ner_tags':[i for sublist in datasets_dict['ner_tags'] for i in sublist],
             'image_path':[datasets_dict['image_path'][idx] for idx, sublist in enumerate(datasets_dict['tokens']) for i in sublist]
        }
    
    # In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the unique labels.
    def get_label_list(labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        labels = list(unique_labels)
        labels = [l for l in labels if l]
        labels.sort()
        return labels

    label_column_name = "ner_tags"
    # labels = get_label_list(dataset[label_column_name])
    if document_type == 'esgReport':
        labels = ['caption', 'figure', 'footer', 'list', 'paragraph', 'reference', 'section', 'table', 'title']
    elif document_type in ['agreement','termSheet']:
        labels = ['caption', 'footer', 'list', 'paragraph', 'reference', 'section', 'table', 'title']
    else:
        labels = ['abstract','author','caption','date','equation','figure', 'footer', 'list', 'paragraph', 'reference', 'section', 'table', 'title']
    id2label = {v: k for v, k in enumerate(labels)}
    label2id = {k: v for v, k in enumerate(labels)}
    num_labels = len(labels)

    config = AutoConfig.from_pretrained(
        model_path,
        num_labels=num_labels,
        input_size=224
    )
    model = AutoModelForTokenClassification.from_pretrained(
        model_path,
        config=config
    )
    processor = AutoProcessor.from_pretrained(
        model_path,
        use_fast=True,
        add_prefix_space=True,
        apply_ocr=False
    )

    # we need to define custom features
    if model_version == 'v2' or model_version is None:
        features = Features({
            'image': Array3D(dtype="int64", shape=(3, 224, 224)),
            'input_ids': Sequence(feature=Value(dtype='int64')),
            'attention_mask': Sequence(Value(dtype='int64')),
            'token_type_ids': Sequence(Value(dtype='int64')),
            'bbox': Array2D(dtype="int64", shape=(512, 4)),
            'labels': Sequence(ClassLabel(names=labels)),
            'offset_mapping': Array2D(dtype="int64", shape=(512, 2)),
            'id': Sequence(feature=Value(dtype="int64"))
        })
    elif model_version == 'v3':
        features = Features({
            'pixel_values': Array3D(dtype="float32", shape=(3, 224, 224)),
            'input_ids': Sequence(feature=Value(dtype='int64')),
            'attention_mask': Sequence(Value(dtype='int64')),
            'bbox': Array2D(dtype="int64", shape=(512, 4)),
            'labels': Sequence(feature=Value(dtype='int64')),
            'offset_mapping': Array2D(dtype="int64", shape=(512, 2)),
            'id': Sequence(feature=Value(dtype="int64"))
        })

    def preprocess_data(examples):

        images = [Image.open(path).convert("RGB") for path in examples['image_path']]
        words = examples['tokens']
        boxes = examples['bboxes']
        word_labels = examples['ner_tags']

        word_labels_ids = [[label2id[i] for i in k] for k in word_labels]
        label_lengths = [len(k) for k in word_labels]

        doc_ids = [[int(id)]*label_lengths[i] for i, id in enumerate(examples['id'])]

        encoded_inputs = processor(images,
                                    words,
                                    boxes=boxes,
                                    word_labels=word_labels_ids,
                                    padding="max_length",
                                    max_length=512,
                                    stride = 128,
                                    return_offsets_mapping=True,
                                    return_overflowing_tokens=True,
                                    truncation=True)

        encoding_for_doc_ids = processor(images,
                                    words,
                                    boxes=boxes,
                                    word_labels=doc_ids,
                                    padding="max_length",
                                    max_length=512,
                                    stride = 128,
                                    return_offsets_mapping=True,
                                    return_overflowing_tokens=True,
                                    truncation=True)

        overflow_to_sample_mapping = encoded_inputs.pop('overflow_to_sample_mapping')
        encoded_inputs['id'] = encoding_for_doc_ids['labels']

        # change the shape of pixel values
        x = []
        for i in range(0, len(encoded_inputs['pixel_values'])):
            x.append(encoded_inputs['pixel_values'][i])
        x = np.stack(x)
        encoded_inputs['pixel_values'] = x

        # for k,v in encoded_inputs.items():
        #     print(k,np.array(v).shape)

        return encoded_inputs

    input_dataset = datasets.map(preprocess_data,
                                 batched=True,
                                 remove_columns=remove_columns,
                                 features=features)

    # Finally, let's set the format to PyTorch, and place everything on the GPU:

    input_dataset.set_format(type="torch", device=device)

    # Next, we create corresponding dataloaders.

    inference_dataloader = DataLoader(input_dataset, batch_size=batch_size, shuffle=False)

    if use_cuda:
        world_size = 1
        rank = 0
        # n = len(gpu_ids) // world_size
        # device_ids = list(range(rank * n, (rank + 1) * n))
        device_ids = [int(i) for i in gpu_ids]
        setup(rank, world_size)
        model = model.to(device)
        model = DDP(model, device_ids=None) # , device_ids=device_ids
        # model = DataParallel(model,device_ids=device_ids).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    else:
        model.to(device)
        
    # Evaluation
    # put model in evaluation mode
    model.eval()
    all_predictions = []
    all_true = []
    all_bboxes = []
    all_tokens = []
    all_ids = []

    t = tqdm.tqdm(inference_dataloader, desc='LayoutLM model inference')
    logger.info(f'model used: {os.path.basename(model_path)}')

    image_column_name = "image_path"
    text_column_name = "tokens"
    boxes_column_name = "bboxes"
    label_column_name = "ner_tags"

    for batch_idx, batch in enumerate(t):
        with torch.no_grad():
            page_ids = batch.pop('id')
            page_ids = page_ids.squeeze().tolist()

            offset_mapping = batch.pop('offset_mapping')
            offset_mapping = offset_mapping.squeeze().tolist()

            if (len(offset_mapping) == 512):
                is_subwords = [np.array(offset_mapping)[:,0] != 0]
            else:
                is_subwords = [np.array(i)[:,0] != 0 for i in offset_mapping]

            for k,v in batch.items():
                batch[k] = v.to(device)

            # forward pass
            outputs = model(**batch)

            # predictions
            predictions = outputs.logits.argmax(-1).squeeze().tolist()
            token_boxes = batch['bbox'].squeeze().tolist()
            labels = batch['labels'].squeeze().tolist()
            input_ids = batch['input_ids'].squeeze().tolist()

            if (len(token_boxes) == 512):
                predictions = [predictions]
                token_boxes = [token_boxes]
                labels = [labels]
                input_ids = [input_ids]
                page_ids = [page_ids]

            def indices(lst, item):
                return [i for i, x in enumerate(lst) if x == item]

            def trim_list_by_indices(lst, indices):
                return [x for i, x in enumerate(lst) if i not in indices]

            flat_tokens = []
            flat_predictions = []
            flat_labels = []
            flat_boxes = []
            flat_page_ids = []

            for i, ids in enumerate(input_ids):
                prev_token = ''
                for j, id in enumerate(ids):
                    p = predictions[i][j]
                    l = labels[i][j]
                    box = token_boxes[i][j]
                    page_id = page_ids[i][j]

                    if is_subwords[i][j] and flat_tokens:
                        flat_tokens.pop(-1)
                        token = prev_token + processor.tokenizer.decode(id)
                        token = token.strip()
                        flat_tokens.append(token)
                    else:
                        token = processor.tokenizer.decode(id).strip()
                    if l != -100:
                        flat_tokens.append(token)
                        flat_predictions.append(id2label[p])
                        flat_labels.append(id2label[l])
                        flat_boxes.append(box)
                        flat_page_ids.append(page_id)
                    prev_token = token

            # remove weird character that cannot decode correctly
            weird_char_indices = indices(flat_tokens, '�')
            weird_char_indices2 = [flat_tokens.index(t) for t in flat_tokens if t.startswith('<s>')]
            list_remove_indices = weird_char_indices + weird_char_indices2
            list_remove_indices.sort()
            sanitized_tokens = trim_list_by_indices(flat_tokens, list_remove_indices) # remove weird character � in the list
            sanitized_tokens = list(map(lambda x: str.replace(x, "�", ""), sanitized_tokens)) # remove weird character � in each string
            sanitized_predictions = trim_list_by_indices(flat_predictions, list_remove_indices)
            sanitized_labels = trim_list_by_indices(flat_labels, list_remove_indices)
            sanitized_boxes = trim_list_by_indices(flat_boxes, list_remove_indices)
            sanitized_page_ids = trim_list_by_indices(flat_page_ids, list_remove_indices)

            all_ids.extend(sanitized_page_ids)
            all_predictions.extend(sanitized_predictions)
            all_true.extend(sanitized_labels)
            all_tokens.extend(sanitized_tokens)
            all_bboxes.extend(sanitized_boxes)

            del batch
            gc.collect()
            torch.cuda.empty_cache()
            
    cleanup()

    all_bboxes = [tuple(x) for x in all_bboxes] # turn all list of bboxes into tuple of bboxes
    all_res = list(zip(all_ids,all_predictions,all_true,all_tokens,all_bboxes))
    # all_res = remove_duplicates(all_res)
    # all_res = sorted(all_res, key=lambda x:(x[0],x[-1][1],x[-1][0],x[-1][3],x[-1][2])) # sort by page id and then vertically and then horizontally by position
    all_res = list(zip(*all_res))
    all_ids, all_predictions, all_true, all_tokens, all_bboxes = all_res

    key = tuple(zip(all_ids,all_bboxes))
    input_key = tuple(zip(input_dict['id'],input_dict['bboxes']))
    map_prediction = dict(zip(key,all_predictions))
    
    output_dict = {
        "id": input_dict['id'],
        "tokens": input_dict['tokens'],
        "bboxes": input_dict['bboxes'],
        "ner_tags": input_dict['ner_tags'],
        "predictions": [map_prediction.get(i) for idx,i in enumerate(input_key)]
    }
    
    idx_none_pred = [idx for idx,i in enumerate(output_dict["predictions"]) if i is None]
    for i in idx_none_pred:
        logger.warning(f"Missing model prediction at id= {output_dict['id'][i]}, tokens= {output_dict['tokens'][i]}, bboxes= {output_dict['bboxes'][i]}, ner_tags= {output_dict['ner_tags'][i]}")

    # filter out the tokens (or characters) doesn't decode sucessfully from model output
    output_dict['id'] = [i for idx, i in enumerate(output_dict['id']) if idx not in idx_none_pred]
    output_dict['tokens'] = [i for idx, i in enumerate(output_dict['tokens']) if idx not in idx_none_pred]
    output_dict['bboxes'] = [i for idx, i in enumerate(output_dict['bboxes']) if idx not in idx_none_pred]
    output_dict['ner_tags'] = [i for idx, i in enumerate(output_dict['ner_tags']) if idx not in idx_none_pred]
    output_dict['predictions'] = [i for idx, i in enumerate(output_dict['predictions']) if idx not in idx_none_pred]
    
    assert len(output_dict['tokens'])==len([i for i in output_dict['predictions'] if i]), f'Input length is {len(output_dict["tokens"])} while output length is {len([i for i in output_dict["predictions"] if i])}'
    
    # with open("{}{}{}".format(OUTPUT_LAYOUTLM_OUTPUT_DIR, fname, '.json'), 'w') as out:
    #     json.dump(output_dict, out, ensure_ascii=False)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    result = list(zip(output_dict['tokens'], output_dict['bboxes'], output_dict['id'], output_dict['ner_tags'], output_dict['predictions']))
    keys = ['token', 'bbox', 'page_id', 'rule_tag', 'tag']
    result = [dict(zip(keys,i)) for i in result]

    return result

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Fine-tuned Layoutlmv2 with ESG reports or agreement inference')

    parser.add_argument(
        '--data',
        required=True,
        help='Input data in pandas DataFrame or dictionary',
    )

    parser.add_argument(
        '--model_path',
        default='models/checkpoints/layoutlm/layoutlmv3_large_500k_docbank_epoch_1_lr_1e-5_1407_esg_epoch_2000_lr_1e-5',
        type=str,
        required=False,
        help='The directory store fine-tuned model checkpoint and log file',
    )

    parser.add_argument(
        '--fname',
        type=str,
        required=True,
        help='The filename of the document',
    )

    parser.add_argument(
        '--gpu_ids',
        default='0,1,2,3',
        type=str,
        required=False,
        help='The GPU IDs utilize for model inference',
    )

    parser.add_argument(
        '--batch_size',
        default=16,
        type=int,
        required=False,
        help='Batch size of model inference',
    )

    parser.add_argument(
        '--local_rank',
        default=0,
        type=int,
        required=False,
        help='Batch size of model inference',
    )

    parser.add_argument(
        "--document_type",
        default='agreement',
        type=str,
        required=False,
        help="Document type that pdf belongs to. Either 'esgReport' , 'agreement' or 'termSheet'",
    )

    args = parser.parse_args()
    all_predictions = token_classification(args.data, args.model_path, args.fname, gpu_ids=args.gpu_ids, batch_size = args.batch_size, document_type=args.document_type)
