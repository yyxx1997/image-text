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
from dataset import create_dataset, create_sampler, create_loader
from scheduler import create_scheduler
from optim import create_optimizer

def modi_opt(optimizer, decay_rate):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay_rate

def reset_opt(optimizer, origin_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = origin_lr

def train(model, data_loader, optimizer, tokenizer, epoch, device, scheduler, config):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.8f}'))
    metric_logger.add_meter('lr_low', utils.SmoothedValue(window_size=1, fmt='{value:.8f}'))
    metric_logger.add_meter('loss_origin', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_entail', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    lr_decay = config['lowlr']
    total_step = epoch * len(data_loader)
    
    for i,(image, text, idx, extra_entail) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device,non_blocking=True)   
        idx = idx.to(device,non_blocking=True)   
        text_input_origin = tokenizer(text, padding='longest', max_length=50, return_tensors="pt").to(device)  
        text_input_entail = tokenizer(extra_entail, padding='longest', max_length=50, return_tensors="pt").to(device)  

        if epoch>0 or not config['warm_up']:
            alpha = config['alpha']
        else:
            alpha = config['alpha']*min(1,i/len(data_loader))

        loss_ita, loss_itm = model(image, text_input_origin,alpha=alpha, idx=idx)                  
        loss_origin = loss_ita + loss_itm
        optimizer.zero_grad()
        loss_origin.backward()
        optimizer.step()  

        origin_lr = optimizer.param_groups[0]['lr']
        metric_logger.update(lr=origin_lr)

        # 使用低学习率蕴含
        if lr_decay > 0:
            loss_ita, loss_itm = model(image, text_input_entail,alpha=alpha, idx=idx)                  
            loss_entail = loss_ita + loss_itm
            modi_opt(optimizer,lr_decay)
            optimizer.zero_grad()
            loss_entail.backward()
            optimizer.step()    
        else:
            loss_entail = 0
        
        metric_logger.update(loss_origin=loss_origin)
        metric_logger.update(loss_entail=loss_entail)
        metric_logger.update(lr_low=optimizer.param_groups[0]["lr"])
        reset_opt(optimizer, origin_lr)

        scheduler.step(total_step+i)      
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}  



@torch.no_grad()
def evaluation(model, data_loader, tokenizer, device, config):
    # test
    model.eval() 
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'    
    
    print('Computing features for evaluation...')
    start_time = time.time()  

    texts = data_loader.dataset.text
    images = data_loader.dataset.image
    img2txt = data_loader.dataset.img2txt
    txt2img = data_loader.dataset.txt2img
    num_text = len(texts)
    text_bs = 256
    text_feats = []
    text_embeds = []  
    text_atts = []
    
    for i in range(0, num_text, text_bs):
        text = texts[i: min(num_text, i+text_bs)]
        text_input = tokenizer(text, padding='max_length', truncation=True, max_length=50, return_tensors="pt").to(device) 
        text_output = model.text_encoder(text_input.input_ids, attention_mask = text_input.attention_mask, mode='text')  
        text_feat = text_output.last_hidden_state
        text_embed = F.normalize(model.text_proj(text_feat[:,0,:]))
        text_embeds.append(text_embed.cpu())   
        text_feats.append(text_feat.cpu())
        text_atts.append(text_input.attention_mask.cpu())
    text_embeds = torch.cat(text_embeds,dim=0)
    text_feats = torch.cat(text_feats,dim=0)
    text_atts = torch.cat(text_atts,dim=0)
    
    image_feats = []
    image_embeds = []
    image_ids = []

    for image, img_id in data_loader: 
        image = image.to(device) 
        image_feat = model.visual_encoder(image)        
        image_embed = model.vision_proj(image_feat[:,0,:])            
        image_embed = F.normalize(image_embed,dim=-1)      
        
        image_feats.append(image_feat.cpu())
        image_embeds.append(image_embed.cpu())
        image_ids.extend(img_id.tolist())
     
    image_feats = torch.cat(image_feats,dim=0)
    image_embeds = torch.cat(image_embeds,dim=0)
    
    sims_matrix = image_embeds @ text_embeds.t()
    score_matrix_i2t = torch.full((len(data_loader.dataset.image),len(texts)),-100.0).to(device)
    
    num_tasks = utils.get_world_size()
    rank = utils.get_rank() 
    step = sims_matrix.size(0)//num_tasks + 1
    start = rank*step
    end = min(sims_matrix.size(0),start+step)

    for i,sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 100, header)): 
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)

        encoder_output = image_feats[start+i].repeat(config['k_test'],1,1).to(device)
        encoder_att = torch.ones(encoder_output.size()[:-1],dtype=torch.long).to(device)
        output = model.text_encoder(encoder_embeds = text_feats[topk_idx].to(device), 
                                    attention_mask = text_atts[topk_idx].to(device),
                                    encoder_hidden_states = encoder_output,
                                    encoder_attention_mask = encoder_att,                             
                                    return_dict = True,
                                    mode = 'fusion'
                                   )
        score = model.itm_head(output.last_hidden_state[:,0,:])[:,1]
        score_matrix_i2t[start+i,topk_idx] = score
        
    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full((len(texts),len(data_loader.dataset.image)),-100.0).to(device)
    
    step = sims_matrix.size(0)//num_tasks + 1
    start = rank*step
    end = min(sims_matrix.size(0),start+step)    
    
    for i,sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 100, header)): 
        
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
        encoder_output = image_feats[topk_idx].to(device)
        encoder_att = torch.ones(encoder_output.size()[:-1],dtype=torch.long).to(device)
        output = model.text_encoder(encoder_embeds = text_feats[start+i].repeat(config['k_test'],1,1).to(device), 
                                    attention_mask = text_atts[start+i].repeat(config['k_test'],1).to(device),
                                    encoder_hidden_states = encoder_output,
                                    encoder_attention_mask = encoder_att,                             
                                    return_dict = True,
                                    mode = 'fusion'
                                   )
        score = model.itm_head(output.last_hidden_state[:,0,:])[:,1]
        score_matrix_t2i[start+i,topk_idx] = score

    if args.distributed:
        dist.barrier()   
        torch.distributed.all_reduce(score_matrix_i2t, op=torch.distributed.ReduceOp.SUM) 
        torch.distributed.all_reduce(score_matrix_t2i, op=torch.distributed.ReduceOp.SUM)        
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str)) 

    result = {}
    topk_upper = config['k_test']
    scores_i2t = score_matrix_i2t.cpu().numpy()
    scores_t2i = score_matrix_t2i.cpu().numpy()
    # Get topk retrieval results
    for img in range(scores_i2t.shape[0]):
        contents = []
        for wait_check in np.argsort(scores_i2t[img])[::-1][:topk_upper]:
            contents.append(int(wait_check))
        result[image_ids[img]] = contents

    topk_result = {}
    origin_text = data_loader.dataset.origin_text
    for image_id, txt_ids in result.items():
        topk_result[images[image_id]] = {
            "goldens": [origin_text[txtid] for txtid in img2txt[image_id]], 
            "topks": [origin_text[txtid] for txtid in txt_ids]
            }
    # Images->Text
    ranks = np.zeros(scores_i2t.shape[0])
    for index, score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        goldens = img2txt[index]
        for i in goldens:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    # Compute metrics

    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    # Text->Images 
    ranks = np.zeros(scores_t2i.shape[0])
    for index, score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        goldens = txt2img[index]
        for i in goldens:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)        

    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    eval_result = {'txt_r1': tr1,
                   'txt_r5': tr5,
                   'txt_r10': tr10,
                   'txt_r_mean': tr_mean,
                   'img_r1': ir1,
                   'img_r5': ir5,
                   'img_r10': ir10,
                   'img_r_mean': ir_mean,
                   'r_mean': r_mean
                   }
    print("eval_result is:\n",eval_result)
    return eval_result, topk_result




def main(args, config):
    utils.init_distributed_mode(args)    
    config['dist'] = args.distributed
    config['lowlr'] = args.lowlr
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset #### 
    print("Creating retrieval dataset")
    train_dataset, val_dataset, test_dataset = create_dataset('re_entail_lr_split', config)  

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler([train_dataset], [True], num_tasks, global_rank) + [None, None]
    else:
        samplers = [None, None, None]
    
    train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset],samplers,
                                                          batch_size=[config['batch_size_train']]+[config['batch_size_test']]*2,
                                                          num_workers=[4,4,4],
                                                          is_trains=[True, False, False], 
                                                          collate_fns=[None,None,None])   
       
    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)

    #### Model #### 
    print("Creating model")
    model = ALBEF(config=config, text_encoder=args.text_encoder, tokenizer=tokenizer)
    
    if args.checkpoint:    
        checkpoint = torch.load(args.checkpoint, map_location='cpu') 
        state_dict = checkpoint['model']
        
        # reshape positional embedding to accomodate for image resolution change
        pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder)         
        state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
        m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],model.visual_encoder_m)   
        state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped 
        
        for key in list(state_dict.keys()):
            if 'bert' in key:
                encoder_key = key.replace('bert.','')         
                state_dict[encoder_key] = state_dict[key] 
                del state_dict[key]                
        msg = model.load_state_dict(state_dict,strict=False)  
        
        print('load checkpoint from %s'%args.checkpoint)
        print(msg)  
        
    
    model = model.to(device)   
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module   
    
    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    arg_sche.steps_per_epoch = len(train_loader)
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer) 
    
    max_epoch = config['schedular']['epochs']
    best = 0
    best_epoch = 0

    if args.eval_before_train:
        val_result,val_topk = evaluation(model_without_ddp, val_loader, tokenizer, device, config)
        test_result,test_topk = evaluation(model_without_ddp, test_loader, tokenizer, device, config)

    print("Start training")
    start_time = time.time()    
    for epoch in range(0, max_epoch):
        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
            train_stats = train(model, train_loader, optimizer, tokenizer, epoch, device, lr_scheduler, config)  
            
        val_result,val_topk = evaluation(model_without_ddp, val_loader, tokenizer, device, config)
        test_result,test_topk = evaluation(model_without_ddp, test_loader, tokenizer, device, config)
    
        if utils.is_main_process():  
            
            if args.evaluate:                
                log_stats = {**{f'val_{k}': v for k, v in val_result.items()},
                             **{f'test_{k}': v for k, v in test_result.items()},                  
                             'epoch': epoch,
                            }
                with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                    f.write(json.dumps(log_stats, ensure_ascii=False, indent=4) + "\n")   
            else:
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             **{f'val_{k}': v for k, v in val_result.items()},
                             **{f'test_{k}': v for k, v in test_result.items()},                  
                             'epoch': epoch,
                            }
                with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                    f.write(json.dumps(log_stats, ensure_ascii=False, indent=4) + "\n")   
                
                save_obj = {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'config': config,
                        'epoch': epoch,
                    }
                torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint-{}.pth'.format(epoch))) 

                if val_result['r_mean']>best:
                    torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))  
                    best = val_result['r_mean']    
                    best_epoch = epoch
                    with open(os.path.join(args.output_dir, "top{}_result_{}_val.json".format(config['k_test'], best_epoch)), "w") as f:
                        f.write(json.dumps(val_topk,ensure_ascii=False,indent=4))
                    with open(os.path.join(args.output_dir, "top{}_result_{}_test.json".format(config['k_test'], best_epoch)), "w") as f:
                        f.write(json.dumps(test_topk,ensure_ascii=False,indent=4))
                    
        if args.evaluate: 
            break
           
        if utils.is_dist_avail_and_initialized():
            dist.barrier()     
        torch.cuda.empty_cache()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 

    if utils.is_main_process():   
        with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
            f.write("best epoch: %d"%best_epoch)               

            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()     
    parser.add_argument('--config', default='./configs/Retrieval_flickr.yaml')
    parser.add_argument('--output_dir', default='output/Retrieval_flickr')        
    parser.add_argument('--checkpoint', default='')   
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--eval_before_train', action='store_true')
    parser.add_argument('--lowlr', default=0.3, type=float)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)
