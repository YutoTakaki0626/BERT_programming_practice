
import numpy as np
import torch
from transformers import BertJapaneseTokenizer, BertForMaskedLM

def predict_mask_topk(text, tokenizer, bert_mlm, num_topk):
    """
    文章中の最初の[MASK]をスコアの上位のトークンに置き換える。
    """

    # 文章を符号化し、BERTで分類スコアを得る。
    input_ids = tokenizer.encode(text, return_tensors='pt')
    input_ids = input_ids.cuda()
    with torch.no_grad():
      output = bert_mlm(input_ids=input_ids)
    scores = output.logits

    # スコアが上位のトークンとスコアを求める
    mask_position = input_ids[0].tolist().index(4)
    topk = scores[0, mask_position].topk(num_topk)
    # tokens 
    ids_topk = topk.indices
    tokens_topk = tokenizer.convert_ids_to_tokens(ids_topk)


    # scores
    scores_topk = topk.values.cpu().numpy()

    # text
    text_topk = []
    for token in tokens_topk:
      token = token.replace('##', '')
      text_topk.append(text.replace('[MASK]', token, 1))

    return text_topk, scores_topk