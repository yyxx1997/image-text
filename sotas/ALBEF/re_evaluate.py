import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader

from models.model_retrieval import ALBEF
from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer
import utils
from dataset import create_loader
from tqdm import tqdm
from torchvision import transforms
import json
import os
import random

from torch.utils.data import Dataset

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from dataset.utils import pre_caption

normalize = transforms.Normalize(
    (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

class re_eval_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):        
        self.ann = json.load(open(ann_file,'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words 
        
        self.text = []
        self.origin_text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        
        txt_id = 0
        for img_id, ann in enumerate(self.ann):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann['caption']):
                self.origin_text.append(caption)
                self.text.append(pre_caption(caption,self.max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1
                                    
    def __len__(self):
        return len(self.image)
    
    def __getitem__(self, index):    
        
        image_path = os.path.join(self.image_root, self.ann[index]['image'])        
        image = Image.open(image_path).convert('RGB')    
        image = self.transform(image)  

        return image, index

@torch.no_grad()
def evaluation_only_i2t(model, data_loader, tokenizer, device, config):
    # get topk entailment texts of image
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Get Entailments:'
    print('Computing features for evaluation...')
    print("pid is ", os.getpid())
    model.eval()
    texts = data_loader.dataset.text
    origin_texts=data_loader.dataset.origin_text
    num_text = len(texts)
    text_bs = 256
    text_feats = []
    text_embeds = []
    text_atts = []
    for i in tqdm(range(0, num_text, text_bs), desc="Loading text and get text features..."):
        text = texts[i: min(num_text, i+text_bs)]
        text_input = tokenizer(text, padding='max_length', truncation=True,
                               max_length=30, return_tensors="pt").to(device)
        text_output = model.text_encoder(
            text_input.input_ids, attention_mask=text_input.attention_mask, mode='text')
        text_feat = text_output.last_hidden_state
        text_embed = F.normalize(model.text_proj(text_feat[:, 0, :]))
        text_embeds.append(text_embed.to('cpu'))
        text_feats.append(text_feat.to('cpu'))
        text_atts.append(text_input.attention_mask.to('cpu'))
    text_embeds = torch.cat(text_embeds, dim=0)
    text_feats = torch.cat(text_feats, dim=0)
    text_atts = torch.cat(text_atts, dim=0)
    print(len(text_embeds))
    image_feats = []
    image_embeds = []
    image_ids = []
    img2txt = data_loader.dataset.img2txt
    count = 2
    for image, img_id in tqdm(data_loader, desc="Loading image and get image features..."):
        image = image.to(device)
        image_feat = model.visual_encoder(image)
        image_embed = model.vision_proj(image_feat[:, 0, :])
        image_embed = F.normalize(image_embed, dim=-1)

        image_feats.append(image_feat.to('cpu'))
        image_embeds.append(image_embed.to('cpu'))

        image_ids.extend(img_id.tolist())
        # count-=1
        # if count<0:
        #     break

    image_feats = torch.cat(image_feats, dim=0)
    image_embeds = torch.cat(image_embeds, dim=0)
    sims_matrix = image_embeds @ text_embeds.t()
    score_matrix_i2t = torch.full(
        (image_embeds.shape[0], len(texts)), -100.0).to(device)

    num_tasks = utils.get_world_size()
    rank = utils.get_rank()
    step = sims_matrix.size(0)//num_tasks + 1
    start = rank*step
    end = min(sims_matrix.size(0), start+step)

    for i, sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)):
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)

        encoder_output = image_feats[start +
                                     i].repeat(config['k_test'], 1, 1).to(device)
        encoder_att = torch.ones(encoder_output.size()[
                                 :-1], dtype=torch.long).to(device)
        output = model.text_encoder(encoder_embeds=text_feats[topk_idx].to(device),
                                    attention_mask=text_atts[topk_idx].to(
                                        device),
                                    encoder_hidden_states=encoder_output,
                                    encoder_attention_mask=encoder_att,
                                    return_dict=True,
                                    mode='fusion'
                                    )
        score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
        score_matrix_i2t[start+i, topk_idx] = score

    if args.distributed:
        dist.barrier()
        torch.distributed.all_reduce(
            score_matrix_i2t, op=torch.distributed.ReduceOp.SUM)

    result = {}
    check_entail = config['k_test']
    scores_i2t = score_matrix_i2t.cpu().numpy()
    #Images->Text
    for img in range(scores_i2t.shape[0]):
        contents = []
        for wait_check in np.argsort(scores_i2t[img])[::-1][:check_entail]:
            contents.append(int(wait_check))
        result[image_ids[img]] = contents

    topk_result = {}
    for image_id, txt_ids in result.items():
        topk_result[data_loader.dataset.image[image_id]] = {
            "goldens": [origin_texts[txtid] for txtid in img2txt[image_id]], 
            "topks": [origin_texts[txtid] for txtid in txt_ids]
            }
    

    #Images->Text
    pres = np.zeros((scores_i2t.shape[0], 10))
    ranks = np.zeros(scores_i2t.shape[0])
    golden_total=0
    for index, score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        goldens = img2txt[index]
        golden_total+=len(goldens)
        for i in goldens:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        pres[index] = np.cumsum(np.in1d(inds[:10], goldens))

    # Compute metrics

    pr5 = 100.0 * np.sum(pres[:, 4]) / golden_total
    pr10 = 100.0 * np.sum(pres[:, 9]) / golden_total

    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    tr_mean = (tr1 + tr5 + tr10) / 3
    pr_mean = (pr5 + pr10)/2

    eval_result = {'txt_r1': tr1,
                   'txt_r5': tr5,
                   'txt_r10': tr10,
                   'txt_r_mean': tr_mean,
                   'txt_pr5': pr5,
                   'txt_pr10': pr10,
                   'txt_pr_mean': pr_mean,
                   }

    return eval_result,topk_result

@torch.no_grad()
def evaluation_only_i2t_rebuild(model, data_loader, tokenizer, device, config):
    # get topk entailment texts of image
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Get Entailments:'
    print('Computing features for evaluation...')
    print("pid is ", os.getpid())
    model.eval()
    texts = data_loader.dataset.text_origin
    num_text = len(texts)
    text_bs = 256
    text_feats = []
    text_embeds = []
    text_atts = []
    for i in tqdm(range(0, num_text, text_bs), desc="Loading text and get text features..."):
        text = texts[i: min(num_text, i+text_bs)]
        text_input = tokenizer(text, padding='max_length', truncation=True,
                               max_length=30, return_tensors="pt").to(device)
        text_output = model.text_encoder(
            text_input.input_ids, attention_mask=text_input.attention_mask, mode='text')
        text_feat = text_output.last_hidden_state
        text_embed = F.normalize(model.text_proj(text_feat[:, 0, :]))
        text_embeds.append(text_embed.to('cpu'))
        text_feats.append(text_feat.to('cpu'))
        text_atts.append(text_input.attention_mask.to('cpu'))
    text_embeds = torch.cat(text_embeds, dim=0)
    text_feats = torch.cat(text_feats, dim=0)
    text_atts = torch.cat(text_atts, dim=0)
    print(len(text_embeds))
    image_feats = []
    image_embeds = []
    image_ids = []
    img2txt = data_loader.dataset.img2txt
    for image, img_id in tqdm(data_loader, desc="Loading image and get image features..."):
        image = image.to(device)
        image_feat = model.visual_encoder(image)
        image_embed = model.vision_proj(image_feat[:, 0, :])
        image_embed = F.normalize(image_embed, dim=-1)

        image_feats.append(image_feat.to('cpu'))
        image_embeds.append(image_embed.to('cpu'))

        image_ids.extend(img_id.tolist())

    image_feats = torch.cat(image_feats, dim=0)
    image_embeds = torch.cat(image_embeds, dim=0)
    sims_matrix = image_embeds @ text_embeds.t()
    score_matrix_i2t = torch.full(
        (image_embeds.shape[0], len(texts)), -100.0).to(device)

    num_tasks = utils.get_world_size()
    rank = utils.get_rank()
    step = sims_matrix.size(0)//num_tasks + 1
    start = rank*step
    end = min(sims_matrix.size(0), start+step)

    for i, sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)):
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)

        encoder_output = image_feats[start +
                                     i].repeat(config['k_test'], 1, 1).to(device)
        encoder_att = torch.ones(encoder_output.size()[
                                 :-1], dtype=torch.long).to(device)
        output = model.text_encoder(encoder_embeds=text_feats[topk_idx].to(device),
                                    attention_mask=text_atts[topk_idx].to(
                                        device),
                                    encoder_hidden_states=encoder_output,
                                    encoder_attention_mask=encoder_att,
                                    return_dict=True,
                                    mode='fusion'
                                    )
        score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
        score_matrix_i2t[start+i, topk_idx] = score

    if args.distributed:
        dist.barrier()
        torch.distributed.all_reduce(
            score_matrix_i2t, op=torch.distributed.ReduceOp.SUM)

    result = {}
    check_entail = config['k_test']
    scores_i2t = score_matrix_i2t.cpu().numpy()
    #Images->Text
    for img in range(scores_i2t.shape[0]):
        contents = []
        for wait_check in np.argsort(scores_i2t[img])[::-1][:check_entail]:
            contents.append(int(wait_check))
        result[image_ids[img]] = contents
    

    #Images->Text
    pres = np.zeros((scores_i2t.shape[0], 10))
    ranks = np.zeros(scores_i2t.shape[0])
    golden_total=0
    for index, score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        goldens = img2txt[index]
        golden_total+=len(goldens)
        for i in goldens:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        pres[index] = np.cumsum(np.in1d(inds[:10], goldens))

    # Compute metrics

    pr5 = 100.0 * np.sum(pres[:, 5]) / golden_total
    pr10 = 100.0 * np.sum(pres[:, 9]) / golden_total

    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    tr_mean = (tr1 + tr5 + tr10) / 3
    pr_mean = (pr5 + pr10)/2

    eval_result = {'txt_r1': tr1,
                   'txt_r5': tr5,
                   'txt_r10': tr10,
                   'txt_r_mean': tr_mean,
                   'txt_pr5': pr5,
                   'txt_pr10': pr10,
                   'txt_pr_mean': pr_mean,
                   }

    return eval_result

def main(args, config):
    utils.init_distributed_mode(args)
    config['dist'] = args.distributed
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset ####
    print("Creating retrieval dataset")
    test_transform = transforms.Compose([
        transforms.Resize(
            (config['image_res'], config['image_res']), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])
    val_dataset = re_eval_dataset(config['test_file'], test_transform, config['image_root'])

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
    samplers = [None]
    val_loader, = create_loader([val_dataset], samplers,
                               batch_size=[config['batch_size_test']],
                               num_workers=[4],
                               is_trains=[False],
                               collate_fns=[None])

    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)

    #### Model ####
    print("Creating model")
    model = ALBEF(config=config, text_encoder=args.text_encoder,
                  tokenizer=tokenizer)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint['model']

        # reshape positional embedding to accomodate for image resolution change
        pos_embed_reshaped = interpolate_pos_embed(
            state_dict['visual_encoder.pos_embed'], model.visual_encoder)
        state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
        m_pos_embed_reshaped = interpolate_pos_embed(
            state_dict['visual_encoder_m.pos_embed'], model.visual_encoder_m)
        state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped

        for key in list(state_dict.keys()):
            if 'bert' in key:
                encoder_key = key.replace('bert.', '')
                state_dict[encoder_key] = state_dict[key]
                del state_dict[key]
        msg = model.load_state_dict(state_dict, strict=False)

        print('load checkpoint from %s' % args.checkpoint)
        print(msg)

    model = model.to(device)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    print("Start Evaluating")
    start_time = time.time()

    eval_result,topk_result = evaluation_only_i2t(model_without_ddp, val_loader, tokenizer, device, config)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluating time {}'.format(total_time_str))

    if utils.is_main_process():
        with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
            f.write(str(eval_result)+'\n')
        with open('{}/topk_result_{}.json'.format(args.output_dir,config['k_test']),'w',encoding="utf8") as file:
            file.write(json.dumps(topk_result,ensure_ascii=False,indent=4))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Retrieval_flickr.yaml')
    parser.add_argument('--output_dir', default='output/Retrieval_flickr')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    print(config)
    print(args)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    main(args, config)
