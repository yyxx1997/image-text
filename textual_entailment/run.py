import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from model import BertTEModel
import os
import time
from transformers import BertTokenizer
from transformers import logging
from utils import *
from torch.utils.tensorboard import SummaryWriter

logging.set_verbosity_error()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class BertDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.data_size = len(dataset)

    def __len__(self):
        return self.data_size

    def __getitem__(self, index):
        return self.dataset[index]

def coffate_fn(examples):
    inputs, targets = [], []
    for label,pre,hypo in examples:
        inputs.append((pre,hypo))
        targets.append(label)
    inputs = tokenizer(inputs,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=512)
    targets = torch.tensor(targets)
    return inputs, targets


train_batch_size = 24
dev_batch_size = 24
num_epoch = 2  # 训练轮次
check_step = 1  # 用以训练中途对模型进行检验：每check_step个epoch进行一次测试和保存模型
data_path = "./data/xnli/xnli.jsonl"  # 数据所在地址
train_ratio = 0.9  # 训练集比例
dev_ratio = 0.5
learning_rate = 1e-5  # 优化器的学习率

# 获取训练、测试数据、分类类别总数
train_data, dev_data, test_data = load_sentence_mrpc(data_path=data_path, train_ratio=train_ratio,dev_ratio=dev_ratio)
print(len(train_data),len(dev_data))
# dev_data=train_data
train_dataset = BertDataset(train_data)
dev_dataset = BertDataset(dev_data)
test_dataset = BertDataset(test_data)

train_dataloader = DataLoader(train_dataset,
                            batch_size=train_batch_size,
                            collate_fn=coffate_fn,
                            shuffle=True)
dev_dataloader = DataLoader(dev_dataset,
                            batch_size=dev_batch_size,
                            collate_fn=coffate_fn)
test_dataloader = DataLoader(test_dataset,
                            batch_size=1,
                            collate_fn=coffate_fn)

device = torch.device('cuda')
pretrained_model_name = 'bert-base-uncased'
model = BertTEModel(2, pretrained_model_name)
model.to(device)
tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
optimizer = Adam(model.parameters(), learning_rate)  #使用Adam优化器
CE_loss = nn.CrossEntropyLoss()  # 使用crossentropy作为二分类任务的损失函数
timestamp = time.strftime("%m_%d_%H_%M", time.localtime())
logs_dir='./logs/xnli/'+timestamp
writer = SummaryWriter(logs_dir+'/tb')

for epoch in range(1, num_epoch + 1):
    # 记录当前epoch的总loss
    train_loss = 0
    model.train()
    for inputs, targets in tqdm(train_dataloader, desc=f"Training Epoch {epoch}"):

        input_ids, input_tyi, input_attn_mask = inputs['input_ids'], inputs[
            'token_type_ids'], inputs['attention_mask']
        input_ids = input_ids.to(device)
        input_attn_mask = input_attn_mask.to(device)
        input_tyi = input_tyi.to(device)
        targets = targets.to(device)
        # 清除现有的梯度
        optimizer.zero_grad()

        # 模型前向传播，model(inputs)等同于model.forward(inputs)
        bert_output = model(input_ids, input_tyi, input_attn_mask)

        loss = CE_loss(bert_output, targets)

        # 梯度反向传播
        loss.backward()

        # 根据反向传播的值更新模型的参数
        optimizer.step()

        # 统计总的损失，.item()方法用于取出tensor中的值
        train_loss += loss.item()
    

    acc = 0
    dev_loss = 0
    with torch.no_grad():
        model.eval()
        for inputs, targets in tqdm(dev_dataloader, desc=f"Evaluating"):
            input_ids, input_tyi, input_attn_mask = inputs['input_ids'], inputs[
                'token_type_ids'], inputs['attention_mask']
            input_ids = input_ids.to('cuda')
            input_attn_mask = input_attn_mask.to('cuda')
            input_tyi = input_tyi.to('cuda')
            targets = targets.to('cuda')
            
            bert_output = model(input_ids, input_tyi, input_attn_mask)
            loss = CE_loss(bert_output, targets)
            dev_loss += loss.item()
            acc += (bert_output.argmax(dim=1) == targets).sum().item()
    acc_avg= acc / len(dev_dataset)
    train_loss_avg = train_loss / len(train_dataset)
    dev_loss_avg = dev_loss / len(dev_dataset)

    
    print(f"train loss: {train_loss_avg:.3f}")
    print(f"dev loss: {dev_loss_avg:.3f}")
    print(f"Dev Acc: {acc_avg :.3f}")

    writer.add_scalar('train/loss', train_loss_avg, epoch)
    writer.add_scalar('dev/loss', dev_loss_avg, epoch)
    writer.add_scalar('dev/accuracy', acc_avg, epoch)
    if epoch % check_step == 0:
        # 保存模型
        checkpoints_dirname = logs_dir
        os.makedirs(checkpoints_dirname, exist_ok=True)
        save_pretrained(model,checkpoints_dirname + '/checkpoint-{}/'.format(epoch))


# acc统计模型在测试数据上分类结果中的正确个数
acc = 0
with torch.no_grad():
    model.eval()
    for inputs, targets in tqdm(test_dataloader, desc=f"Testing"):

            input_ids, input_tyi, input_attn_mask = inputs['input_ids'], inputs[
                'token_type_ids'], inputs['attention_mask']
            input_ids = input_ids.to('cuda')
            input_attn_mask = input_attn_mask.to('cuda')
            input_tyi = input_tyi.to('cuda')
            targets = targets.to('cuda')
            
            bert_output = model(input_ids, input_tyi, input_attn_mask)
            acc += (bert_output.argmax(dim=1) == targets).sum().item()
    #输出在测试集上的准确率
    acc_avg= acc / len(test_dataset)
    print(f"Test Acc: {acc_avg:.3f}")