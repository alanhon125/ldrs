'''
Keyphrase Boundary Infilling with Replacement (KBIR) fine-tunes it on the Inspec dataset, A dataset for benchmarking keyphrase extraction
and generation techniques from abstracts of English scientific papers.
'''

from transformers import (
    TokenClassificationPipeline,
    AutoModelForTokenClassification,
    AutoTokenizer,
)
from transformers.pipelines import AggregationStrategy
import numpy as np
import torch
from torch.nn import DataParallel

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:1' if use_cuda else 'cpu')
# device_ids = [0,1,2,3]
# Define keyphrase extraction pipeline
class KeyphraseExtractionPipeline(TokenClassificationPipeline):
    def __init__(self, model, *args, **kwargs):
        super().__init__(
            model=AutoModelForTokenClassification.from_pretrained(model),
            tokenizer=AutoTokenizer.from_pretrained(model),
            device=device,
            *args,
            **kwargs
        )

    def postprocess(self, all_outputs):
        try:
            results = super().postprocess(
                all_outputs=all_outputs,
                aggregation_strategy=AggregationStrategy.SIMPLE,
            )
        except:
            results = super().postprocess(
                model_outputs=all_outputs,
                aggregation_strategy=AggregationStrategy.FIRST,
            )
        return np.unique([result.get("word").strip() for result in results])
