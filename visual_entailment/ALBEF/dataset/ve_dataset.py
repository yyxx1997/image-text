import json
import os
from torch.utils.data import Dataset
from PIL import Image
from dataset.utils import pre_caption


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
        
        ann = self.ann[index]
        if '.jpg' in ann['image']:
            image_path = os.path.join(self.image_root,ann['image'])
        else:
            image_path = os.path.join(self.image_root,'%s.jpg'%ann['image'])        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)          

        captions = '[SEP]'.join([pre_caption(caption, self.max_words) for caption in ann['caption']])
        sentence = pre_caption(ann['sentence'], self.max_words)

        return image, captions, sentence, self.labels[ann['label']]

class ve_inference_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):        
        self.wait_infer = json.load(open(ann_file,'r'))
        self.texts=[]
        self.images=[]
        self.img2txt={}
        self.ann=[]
        text_id=0
        image_id=0
        for image_path,dct in self.wait_infer.items():
            self.images.append(image_path)
            self.img2txt[image_id]=[]
            goldens=dct['goldens']
            topks=dct['topks']
            for t in topks:
                if t not in goldens:
                    body={
                        "image":image_id,
                        "text":text_id
                    }
                    self.ann.append(body)
                    self.texts.append(t)
                    self.img2txt[image_id].append(text_id)
                    text_id+=1
            image_id+=1
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        
    def __len__(self):
        return len(self.ann)
    

    def __getitem__(self, index):    
        
        ann = self.ann[index]
        image_id = ann['image']
        text_id = ann['text']
        image_name = self.images[image_id]
        hypothesis = self.texts[text_id]
        captions = self.wait_infer[image_name]['goldens']
        image_path = os.path.join(self.image_root,image_name)
      
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)          

        premise = '[SEP]'.join([pre_caption(caption, self.max_words) for caption in captions])
        hypo = pre_caption(hypothesis, self.max_words)

        return image, premise, hypo , image_id, text_id