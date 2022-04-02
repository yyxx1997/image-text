import torch.nn as nn
from transformers import BertModel


class BertTEModel(nn.Module):

    # 初始化类
    def __init__(self, class_size, pretrained_name='bert-base-uncased'):
        """
        Args: 
            class_size  :指定分类模型的最终类别数目，以确定线性分类器的映射维度
            pretrained_name :用以指定bert的预训练模型
        """
        super(BertTEModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_name,
                                              return_dict=True)
        self.classifier = nn.Linear(768, class_size)
        self.sm=nn.Softmax(dim=-1)

    def forward(self, input_ids, input_tyi, input_attn_mask):
        output = self.bert(input_ids, input_tyi, input_attn_mask)
        categories_numberic = self.sm(self.classifier(output.pooler_output))
        return categories_numberic