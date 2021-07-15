import torch
from tqdm import tqdm

def create_dataset_for_loader(tokenizer, dataset, max_length):
    """
    データセットをデータローダに入力可能な形式にする。
    """
    dataset_for_loader = []
    for sample in tqdm(dataset):
        wrong_text = sample['wrong_text']
        correct_text = sample['correct_text']
        encoding = tokenizer.encode_plus_tagged(
            wrong_text, correct_text, max_length=max_length
        )
        encoding = { k: torch.tensor(v) for k, v in encoding.items() }
        dataset_for_loader.append(encoding)
    return dataset_for_loader

