import argparse
from ast import arg
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
import torch
import torch.nn.functional as F
import torch.distributed as dist
from transformers import BertTokenizer
import utils
from dataset import create_dataset, create_sampler, create_loader
from scheduler import create_scheduler
from optim import create_optimizer
from models.model_te import BertTEModel 
from tqdm import tqdm
from contextlib import nullcontext
# from torch.utils.tensorboard import SummaryWriter


def train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config):
    # train
    model.train()  
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_te', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    step_size = 100
    warmup_iterations = warmup_steps*step_size  
    K = config.gradient_accumulation_steps
    my_context = model.no_sync if config.local_rank != -1 and i % K != 0 else nullcontext

    optimizer.zero_grad()
    for i,(premise, hypos, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        targets = targets.to(device,non_blocking=True)   
        text_input = tokenizer(list(zip(premise,hypos)), padding='longest', return_tensors="pt").to(device)  

        with my_context():
            loss_te = model(text_input,targets=targets)                  
            loss = loss_te / K
            loss.backward()  # 积累梯度，不应用梯度改变
        if i % K == 0:
            optimizer.step()
            optimizer.zero_grad()   
        
        metric_logger.update(loss_te=loss_te.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if epoch==0 and i%step_size==0 and i<=warmup_iterations: 
            scheduler.step(i//step_size)         
        
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
    print_freq = 50
    start_time = time.time()  
    predictions=[]
    goldens=[]
    for premise, hypos, targets in metric_logger.log_every(data_loader, print_freq, header):
        targets = targets.to(device,non_blocking=True)   
        text_input = tokenizer(list(zip(premise,hypos)), padding='longest', return_tensors="pt").to(device)  
        prediction = model(text_input,targets=targets,train=False)   
        prediction = utils.concat_all_gather(prediction,config.distributed).to('cpu')   
        targets = utils.concat_all_gather(targets,config.distributed).to('cpu')            
        predictions.append(prediction)
        goldens.append(targets)
    predictions=torch.cat(predictions)
    goldens=torch.cat(goldens)
    _, pred_class = predictions.max(1)
    accuracy = (goldens==pred_class).sum() / goldens.size(0)
    print("evaluation dataset size is ", goldens.size(0))
    print("Averaged stats:", accuracy)       
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str)) 

    eval_result =  {'accuracy': accuracy.item()}
    return eval_result


def main(config):
     
    device = torch.device(config.device)
    # fix the seed for reproducibility
    seed = config.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    #### Dataset #### 
    print("Creating dataset")
    train_dataset, val_dataset, test_dataset = create_dataset(config)  

    if config.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler([train_dataset,val_dataset,test_dataset], [True, False, False], num_tasks, global_rank)
    else:
        samplers = [None, None, None]
    
    train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset],samplers,
                                                          batch_size=[config['batch_size_train']]+[config['batch_size_test']]*2,
                                                          num_workers=[4,4,4],
                                                          is_trains=[True, False, False], 
                                                          collate_fns=[None,None,None])   
       
    tokenizer = BertTokenizer.from_pretrained(config['text_encoder'])

    #### Model #### 
    print("Creating model")
    model = BertTEModel(config=config, tokenizer=tokenizer)
    
    if config.checkpoint:    
        checkpoint = torch.load(config.checkpoint, map_location='cpu') 
        state_dict = checkpoint['model']              
        msg = model.load_state_dict(state_dict,strict=False)  
        print('load checkpoint from %s'%config.checkpoint)
        print(msg)  
        
    
    model = model.to(device)   
    model_without_ddp = model
    if config.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.gpu])
        model_without_ddp = model.module   
    
    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)  
    
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']
    best = 0
    best_epoch = 0
    best_standard = 'accuracy'
    if config.eval_before_train:
        acc_val = evaluation(model_without_ddp, val_loader, tokenizer, device, config)
        acc_test = evaluation(model_without_ddp, test_loader, tokenizer, device, config)
    print("Start training")
    print("***** Running training *****")
    print(f"  Num examples = {len(train_loader)}")
    print(f"  Num Epochs = {max_epoch}")
    print(f"  Instantaneous batch size per device = {config.batch_size_train}")
    print(f"  Total train batch size (w. parallel, distributed & accumulation) = {config.total_train_batch_size}")
    print(f"  Gradient Accumulation steps = {config.gradient_accumulation_steps}")
    start_time = time.time()    
    for epoch in range(0, max_epoch):
        if not config.evaluate:
            if config.distributed:
                train_loader.sampler.set_epoch(epoch)
            train_stats = train(model, train_loader, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler, config)  
            
        val_result = evaluation(model_without_ddp, val_loader, tokenizer, device, config)
        test_result = evaluation(model_without_ddp, test_loader, tokenizer, device, config)
    
        if utils.is_main_process():  
            
            if config.evaluate:                
                log_stats = {**{f'val_{k}': v for k, v in val_result.items()},
                             **{f'test_{k}': v for k, v in test_result.items()},                  
                             'epoch': epoch,
                            }
                with open(os.path.join(config.output_dir, "log.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")     
            else:
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             **{f'val_{k}': v for k, v in val_result.items()},
                             **{f'test_{k}': v for k, v in test_result.items()},                  
                             'epoch': epoch,
                            }
                with open(os.path.join(config.output_dir, "log.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")   
                
            save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'config': config,
                    'epoch': epoch,
                }
            torch.save(save_obj, os.path.join(config.output_dir, 'checkpoint-{}.pth'.format(epoch))) 

            if val_result[best_standard]>best:
                best = val_result[best_standard]    
                best_epoch = epoch
                torch.save(save_obj, os.path.join(config.output_dir, 'checkpoint_best.pth'))
                    
        if config.evaluate: 
            break
           
        lr_scheduler.step(epoch+warmup_steps+1)  
        if utils.is_dist_avail_and_initialized():
            dist.barrier()     
        torch.cuda.empty_cache()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 

    if utils.is_main_process():   
        with open(os.path.join(config.output_dir, "log.txt"),"a") as f:
            f.write("best epoch: %d"%best_epoch)               

def parse_args():
    parser = argparse.ArgumentParser(
        description="necessarily parameters for run this code."
    )     
    parser.add_argument('--config', default='./configs/TE.yaml')
    parser.add_argument('--output_dir', default='./output/textual_entailment')        
    parser.add_argument('--checkpoint', default='')   
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--dist_backend', default='nccl')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int, help='gradient accumulation for increase batch virtually.')
    parser.add_argument('--local_rank', default=-1, type=int, help='device number of current process.')
    parser.add_argument('--eval_before_train', action='store_true')
    args = parser.parse_args()
    return args
            
if __name__ == '__main__':

    # set configuration for training or evaluating
    args = parse_args()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    utils.init_distributed_mode(args)
    args.total_train_batch_size = config['batch_size_train'] * args.gradient_accumulation_steps * args.world_size
    
    config=utils.AttrDict(config)
    args=utils.AttrDict(args.__dict__)
    config.update(args)

    print("all global configuration is here:\n",config)
    if utils.is_main_process():
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)   
        yaml.dump(dict(config), open(os.path.join(config.output_dir, 'global_config.yaml'), 'w')) 
    main(config)
