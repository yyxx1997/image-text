import json
import os
import random

from torch.utils.data import Dataset
from dataset.utils import pre_caption


class te_train_dataset(Dataset):
    def __init__(self, ann_file, max_words=30):        
        all_data =  json.load(open(ann_file,'r'))
        self.max_words = max_words
        self.ann = []
        for dct in all_data:
            dct['label']= 1 if dct['label'] != 0 else 0
            self.ann.append(dct) 
        
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        
        ann = self.ann[index]
        premise = pre_caption(ann['premise'], self.max_words)
        hypo = pre_caption(ann['hypothesis'], self.max_words)
        label = ann['label']
        return premise, hypo , label
    
    

class te_eval_dataset(Dataset):
    def __init__(self, ann_file, max_words=30):        
        all_data =  json.load(open(ann_file,'r'))
        self.max_words = max_words
        self.ann = []
        for dct in all_data:
            dct['label']= 1 if dct['label'] != 0 else 0
            self.ann.append(dct) 
        
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        
        ann = self.ann[index]
        premise = pre_caption(ann['premise'], self.max_words)
        hypo = pre_caption(ann['hypothesis'], self.max_words)
        label = ann['label']
        return premise, hypo , label
      
        
            

    
