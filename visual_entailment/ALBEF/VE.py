import argparse
import os
from regex import P
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

def train(model, data_loader, optimizer, val_loader, test_loader,tokenizer, epoch, warmup_steps, device, scheduler, config, args, best):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_ve', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_te', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_joint', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50   
    step_size = 100
    warmup_iterations = warmup_steps*step_size
    num = 0   
    for i,(images, caption, text, targets,image_bool,text_bool) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        images= images.to(device,non_blocking=True)
        targets = targets.to(device,non_blocking=True)
        if caption is not None:
            text_inputs = tokenizer(list(zip(caption,text)), padding='longest', return_tensors="pt").to(device)
        else:
            text_inputs = None 
        
        hypo_inputs = tokenizer(text, padding='longest', return_tensors="pt").to(device)

        if epoch>0 or not config['warm_up']:
            alpha = config['alpha']
        else:
            alpha = config['alpha']*min(1,i/len(data_loader))

        loss_ve, loss_te, loss_joint = model(images, text_inputs, hypo_inputs, targets=targets, train=True, alpha=alpha,image_bool=image_bool,text_bool=text_bool)    
        loss = loss_te + loss_ve + loss_joint
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
               
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(loss_ve=loss_ve.item())
        metric_logger.update(loss_te=loss_te.item())
        metric_logger.update(loss_joint=loss_joint.item())
        if epoch==0 and i%step_size==0 and i<=warmup_iterations: 
            scheduler.step(i//step_size)
        num+=1
        if num % 500 ==0:
            val_stats = evaluate(model, val_loader, tokenizer, device, config)
            test_stats = evaluate(model, test_loader, tokenizer, device, config)

            if utils.is_main_process():  
                    log_stats = {**{f'val_{k}': v for k, v in val_stats.items()},
                                **{f'test_{k}': v for k, v in test_stats.items()},
                                'epoch': epoch,
                                }

                    with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                        f.write(json.dumps(log_stats) + "\n")                
                    save_obj = {
                            'model': model.module.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': scheduler.state_dict(),
                            'config': config,
                            'epoch': epoch,
                        }
                    #torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint-{}.pth'.format(epoch))) 
                    if float(val_stats['accuracy'])>best:
                        
                        torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth')) 
                        best = float(val_stats['accuracy'])
                        # best_epoch = epoch
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}, best    

@torch.no_grad()
def evaluate(model, data_loader, tokenizer, device, config):
    # test
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'
    print_freq = 50
    predictions=[]
    goldens=[]
    for i,(images, caption, text, targets,image_bool,text_bool) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        images, targets = images.to(device,non_blocking=True), targets.to(device,non_blocking=True)     
        text_inputs = tokenizer(list(zip(caption,text)), padding='longest', return_tensors="pt").to(device)       
        hypo_inputs = tokenizer(text, padding='longest', return_tensors="pt").to(device) 
        prediction = model(images, text_inputs, hypo_inputs, targets=targets, train=False,image_bool=image_bool,text_bool=text_bool)  
        prediction = utils.concat_all_gather(prediction,config['dist']).to('cpu')   
        targets = utils.concat_all_gather(targets,config['dist']).to('cpu')            
        predictions.append(prediction)
        goldens.append(targets)

    predictions=torch.cat(predictions)
    goldens=torch.cat(goldens)
    _, pred_class = predictions.max(1)
    accuracy = (goldens==pred_class).sum() / goldens.size(0)
    precision = goldens[pred_class==1].sum() / (pred_class==1).sum()
    recall = pred_class[goldens==1].sum() / goldens.sum()
    F1 = 2 * precision * recall / (precision + recall)
    print("evaluation dataset size is ", goldens.size(0))
    print("Averaged stats accuracy:", accuracy)     
    print("Averaged stats precision:", precision)    
    print("Averaged stats recall:", recall)
    print("Averaged stats F1:", F1)
    eval_result =  {
        'accuracy': accuracy.item(),
        'precision':precision.item(),
        'recall':recall.item(),
        'F1':F1.item()
        }
    print("Averaged stats: acc", eval_result["accuracy"])     
    return eval_result
    
def main(args, config):
    utils.init_distributed_mode(args)    
    config['dist']=args.distributed
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset #### 
    print("Creating dataset")
    datasets = create_dataset('ve', config) 
    
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler(datasets, [True, False, False], num_tasks, global_rank)         
    else:
        samplers = [None, None, None]

    train_loader, val_loader, test_loader = create_loader(datasets,samplers,
                                                          batch_size=[config['batch_size_train']]+[config['batch_size_test']]*2,
                                                          num_workers=[4,4,4],is_trains=[True,False,False], 
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
        
        if not args.evaluate:
            if config['distill']:
                m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],model.visual_encoder_m)   
                state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped 

            for key in list(state_dict.keys()):   
                if 'te_bert' in key:
                    continue       
                if 'bert' in key:
                    new_key = key.replace('bert.','')
                    state_dict[new_key] = state_dict[key] 
                    del state_dict[key]
                
        msg = model.load_state_dict(state_dict,strict=False)
        print('load checkpoint from %s'%args.checkpoint)
        print(msg)
    if args.te_checkpoint:
        te_check = args.te_checkpoint
        checkpoint = torch.load(te_check, map_location='cpu')
        state_dict = checkpoint['model']
        for key in list(state_dict.keys()):                
            if 'bert' in key:
                new_key = key.replace('bert.','')
                state_dict[new_key] = state_dict[key] 
                del state_dict[key]
        msg = model.te_bert.load_state_dict(state_dict,strict=False)
        print('load checkpoint from %s'%te_check)
        print(msg)

    model = model.to(device)   
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],find_unused_parameters=True)
        # model = model.module
        model_without_ddp = model.module    
    
    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)  
    # optimizer.load_state_dict(checkpoint['optimizer'])
    # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']
    best = 0
    best_epoch = 0
    
    print("Start training")
    start_time = time.time()
    val_stats = evaluate(model, val_loader, tokenizer, device, config)
    # test_stats = evaluate(model, test_loader, tokenizer, device, config)
    for epoch in range(0, max_epoch):
        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
            train_stats,best = train(model, train_loader, optimizer,val_loader,test_loader,tokenizer, epoch, warmup_steps, device, lr_scheduler, config, args, best)  
            
        val_stats = evaluate(model, val_loader, tokenizer, device, config)
        test_stats = evaluate(model, test_loader, tokenizer, device, config)

        if utils.is_main_process():  
            if args.evaluate:
                log_stats = {**{f'val_{k}': v for k, v in val_stats.items()},
                             **{f'test_{k}': v for k, v in test_stats.items()},
                             'epoch': epoch,
                            }

                with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")                
            else:    
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             **{f'val_{k}': v for k, v in val_stats.items()},
                             **{f'test_{k}': v for k, v in test_stats.items()},
                             'epoch': epoch,
                            }

                with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")
                save_obj = {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'config': config,
                        'epoch': epoch,
                    }
                torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint-{}.pth'.format(epoch))) 
                if float(val_stats['accuracy'])>best:
                    
                    torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth')) 
                    best = float(val_stats['accuracy'])
                    best_epoch = epoch
        
        if args.evaluate:
            break
        lr_scheduler.step(epoch+warmup_steps+1)  
        dist.barrier()   
                
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 
    
    if utils.is_main_process():   
        with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
            f.write("best epoch: %d"%best_epoch)         
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='/data1/ach/image-text/visual_entailment/ALBEF/configs/debug.yaml')
    parser.add_argument('--output_dir', default='output/VE')  
    parser.add_argument('--checkpoint', default='/data1/ach/project/ALBEF/output/pre.pth')   
    parser.add_argument('--te_checkpoint', default='')
    parser.add_argument('--text_encoder', default='/data1/ach/bert-base-uncased')
    parser.add_argument('--evaluate', action='store_true')    
    parser.add_argument('--device', default='cuda:6')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=False, type=bool)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)
