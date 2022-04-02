import torch
import os
import json
from tqdm import tqdm
import argparse

def read_json(filepath):
    with open(filepath, 'r', encoding="utf8") as file:
        return json.load(file)


def save_pretrained(model, path):
    # 保存模型，先利用os模块创建文件夹，后利用torch.save()写入模型文件
    os.makedirs(path, exist_ok=True)
    torch.save(model, os.path.join(path, 'model.pth'))


def load_sentence_mrpc(data_path, train_ratio=0.8, dev_ratio=0.5):
    all_data = []
    statis = {"pos": 0, "neg": 0}
    for dct in tqdm(read_json(data_path)):
        label = 1 if dct['label'] != 0 else 0
        if label == 0:
            statis['pos'] += 1
        else:
            statis['neg'] += 1
        all_data.append((label, dct['premise'], dct['hypothesis']))
    import random
    random.shuffle(all_data)
    length = len(all_data)
    train_len = int(length * train_ratio)
    dev_len = int((length-train_len)*dev_ratio)
    train_data = all_data[:train_len]
    dev_data = all_data[train_len:train_len+dev_len]
    test_data = all_data[train_len+dev_len:]
    print("statistic about this dataset is :\n", statis)
    return train_data, dev_data, test_data


if __name__ == "__main__":
    data_path = r"E:\code_backup\textual_entailment\xnli.jsonl"
    load_sentence_mrpc(data_path)
