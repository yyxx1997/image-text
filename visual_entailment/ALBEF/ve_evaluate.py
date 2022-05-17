import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
import json
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.model_ve import ALBEF
from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer

import utils
from dataset import create_dataset, create_sampler, create_loader
from scheduler import create_scheduler
from optim import create_optimizer

@torch.no_grad()
def evaluate_entail_rate(model, data_loader, tokenizer, device, config):
    # test
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'
    print_freq = 50
    predictions = []
    image_id_total = []
    for i, (images, caption, text, image_ids, text_ids) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        images = images.to(device, non_blocking=True)
        image_ids = image_ids.to(device, non_blocking=True)
        text_ids = text_ids.to(device, non_blocking=True)
        text_inputs = tokenizer(list(zip(caption, text)), padding='longest', return_tensors="pt").to(device)
        hypo_inputs = tokenizer(text, padding='longest',return_tensors="pt").to(device)
        prediction = model(images, text_inputs, hypo_inputs, train=False)

        prediction = utils.concat_all_gather(prediction, config['dist']).to('cpu')
        image_ids = utils.concat_all_gather(image_ids, config['dist']).to('cpu')

        predictions.append(prediction)
        image_id_total.append(image_ids)
    
    img2txt = data_loader.dataset.img2txt
    predictions = torch.cat(predictions)
    _, pred_class = predictions.max(1)
    image_id_total = torch.cat(image_id_total)

    t10 = 0
    t30 = 0 
    t50 = 0
    t_all = 0
    total = 0
    for img_id in img2txt.keys():
        idx = torch.where(image_id_total == img_id)[0]
        pred_image_id = pred_class[idx]
        topk_entail_number = torch.cumsum(pred_image_id,dim=-1)
        t10 += topk_entail_number[9].item()
        t30 += topk_entail_number[29].item()
        t50 += topk_entail_number[49].item()
        t_all += topk_entail_number[-1].item()
        total += 1 
    
    t10 /= total*10
    t30 /= total*30
    t50 /= total*50
    t_all /= total*128

    eval_result = {
        "t10":t10,
        "t30":t30,
        "t50":t50,
        "t_all":t_all
    }

    return eval_result

@torch.no_grad()
def evaluate_infer(model, data_loader, tokenizer, device, config):
    # test
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'
    print_freq = 50
    predictions = []
    image_id_total = []
    text_id_total = []
    for i, (images, caption, text, image_ids, text_ids) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        images = images.to(device, non_blocking=True)
        image_ids = image_ids.to(device, non_blocking=True)
        text_ids = text_ids.to(device, non_blocking=True)
        text_inputs = tokenizer(list(zip(caption, text)), padding='longest', return_tensors="pt").to(device)
        hypo_inputs = tokenizer(text, padding='longest',return_tensors="pt").to(device)
        prediction = model(images, text_inputs, hypo_inputs, train=False)

        prediction = utils.concat_all_gather(prediction, config['dist']).to('cpu')
        image_ids = utils.concat_all_gather(image_ids, config['dist']).to('cpu')
        text_ids = utils.concat_all_gather(text_ids, config['dist']).to('cpu')

        predictions.append(prediction)
        image_id_total.append(image_ids)
        text_id_total.append(text_ids)

    predictions = torch.cat(predictions)
    _, pred_class = predictions.max(1)
    image_id_total = torch.cat(image_id_total).tolist()
    text_id_total = torch.cat(text_id_total).tolist()
    pred_class = pred_class.tolist()

    entailments={}
    images=data_loader.dataset.images
    texts=data_loader.dataset.texts

    for image_id,text_id,prediction in zip(image_id_total,text_id_total,pred_class):
        if prediction != 1:
            continue
        image = images[image_id]
        text = texts[text_id]
        if image not in entailments.keys():
            entailments[image]={
                "goldens":data_loader.dataset.wait_infer[image]['goldens'],
                "entailments":[]
            }
        else:
            entailments[image]['entailments'].append(text)

    return entailments


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
    print("Creating dataset")
    datasets = create_dataset('ve_eval', config)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler([datasets], [False], num_tasks, global_rank)
    else:
        samplers = [None]

    test_loader = DataLoader(
            datasets,
            batch_size=config['batch_size_test'],
            num_workers=4,
            pin_memory=True,
            sampler=samplers[0],
            shuffle=False,
            collate_fn=None
        )

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

        if not args.evaluate:
            if config['distill']:
                m_pos_embed_reshaped = interpolate_pos_embed(
                    state_dict['visual_encoder_m.pos_embed'], model.visual_encoder_m)
                state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped

            for key in list(state_dict.keys()):
                if 'te_bert' in key:
                    continue
                if 'bert' in key:
                    new_key = key.replace('bert.', '')
                    state_dict[new_key] = state_dict[key]
                    del state_dict[key]

        msg = model.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % args.checkpoint)
        print(msg)
    if args.te_checkpoint:
        te_check = args.te_checkpoint
        checkpoint = torch.load(te_check, map_location='cpu')
        state_dict = checkpoint['model']
        for key in list(state_dict.keys()):
            if 'bert' in key:
                new_key = key.replace('bert.', '')
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        msg = model.te_bert.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % te_check)
        print(msg)

    model = model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu])
        model_without_ddp = model.module
    if args.evaluate:
        result = evaluate_entail_rate(model, test_loader, tokenizer, device, config)
    else:
        result = evaluate_infer(model, test_loader, tokenizer, device, config)

    if utils.is_main_process():
        with open(os.path.join(args.output_dir, "result.txt"), "w",encoding="utf8") as f:
            f.write(json.dumps(result,ensure_ascii=False,indent=4))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/VE.yaml')
    parser.add_argument('--output_dir', default='output/VE')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--te_checkpoint', default='')
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

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    main(args, config)
