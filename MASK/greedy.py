
import numpy as np
import torch
from transformers import BertJapaneseTokenizer, BertForMaskedLM
from mask_predict import predict_mask_topk


def greedy_prediction(text, tokenizer, bert_mlm):

  for _ in range(text.count('[MASK]')):
    text = predict_mask_topk(text, tokenizer, bert_mlm, 1)[0][0]
  return text