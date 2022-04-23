import torch.nn as nn
from transformers import BertModel
import torch.nn.functional as F


class BertTEModel(nn.Module):

    # 初始化类
    def __init__(self,
                 config,
                 tokenizer=None,
                 ):
        """
        Args: 
            class_size  :指定分类模型的最终类别数目，以确定线性分类器的映射维度
            pretrained_name :用以指定bert的预训练模型
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.config = config
        self.text_encoder = BertModel.from_pretrained(config['text_encoder'],
                                                      return_dict=True,
                                                      add_pooling_layer=False
                                                      )
        self.textual_cls_head = nn.Sequential(
            nn.Linear(self.text_encoder.config.hidden_size,
                      self.text_encoder.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.text_encoder.config.hidden_size,
                      self.config['class_number'])
        )

    def forward(self, text, targets, train=True):
        textual_entailment = self.text_encoder(text.input_ids,
                                               attention_mask=text.attention_mask,
                                               return_dict=True
                                               )
        te_hiddens = textual_entailment.last_hidden_state[:, 0, :]
        prediction = self.textual_cls_head(te_hiddens)
        if train:
            loss_te = F.cross_entropy(prediction, targets)
            return loss_te
        else:
            return prediction
