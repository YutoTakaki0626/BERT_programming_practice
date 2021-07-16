import numpy as np
import torch
from transformers import BertJapaneseTokenizer, BertForMaskedLM
from mask_predict import predict_mask_topk

def beam_search(text, tokenizer, bert_mlm, num_topk):
  num_mask = text.count('[MASK]')
  text_topk = [text]
  scores_topk = np.array([0])
  for _ in range(num_mask):
    text_candidates = []
    score_candidates = []
    for text_mask, score in zip(text_topk, scores_topk):
      text_topk_inner, scores_topk_inner = predict_mask_topk(
          text_mask, tokenizer, bert_mlm, num_topk
      )
      text_candidates.extend(text_topk_inner)
      score_candidates.append(score+scores_topk_inner)

    # 合計スコアの高いものを選ぶ
    score_candidates = np.hstack(score_candidates)
    idx_list = score_candidates.argsort()[::-1][:num_topk]
    text_topk = [ text_candidates[idx] for idx in idx_list ]
    scores_topk = score_candidates[idx_list]

  return text_topk, scores_topk