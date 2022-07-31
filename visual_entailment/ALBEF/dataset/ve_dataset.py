import json
import os
from torch.utils.data import Dataset
from PIL import Image
from dataset.utils import pre_caption
import torch


class ve_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):        
        self.ann = json.load(open(ann_file,'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.labels = {'entailment':1,'neutral':0,'contradiction':0}
        
    def __len__(self):
        return len(self.ann)
    

    def __getitem__(self, index):    
        '''
        Input agnostic model
        '''
        ann = self.ann[index]
        image_bool = []
        if "image" in ann.keys():
            if "COCO" in ann["image"]:
                image_path = "/data1/ach/coco2014/{}".format(ann["image"])
                # /SISDC_GPFS/Home_SE/zqcao-suda/ach/ALBEF/configs/VE.yaml
            # elif 'jpg' in ann["image"]:
            #     image_path = os.path.join(self.image_root,ann['image'])     
            else:
                image_path = os.path.join(self.image_root,'%s.jpg'%ann['image'])      
            image = Image.open(image_path).convert('RGB')   
            image = self.transform(image)
            image_bool.append(1)
        else:
            image = Image.open('/data1/ach/image-text/visual_entailment/ALBEF/ve_caption_new/test.jpg').convert('RGB')   
            image = self.transform(image)
            image_bool.append(0)
        caption_bool = []
        if "caption" in ann.keys():
            # if len(ann["caption"]) ==
            captions = '[SEP]'.join([pre_caption(caption, self.max_words) for caption in ann['caption']])
            caption_bool.append(1)
        else:
            captions = ""
            caption_bool.append(0)
        sentence = pre_caption(ann['sentence'], self.max_words)

        return image, captions, sentence, self.labels[ann['label']],image_bool,caption_bool
    