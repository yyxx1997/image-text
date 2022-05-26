from functools import partial
from models.vit import VisionTransformer
from models.xbert import BertConfig, BertModel
from transformers import BertModel as tBert
import torch
from torch import nn
import torch.nn.functional as F


class ALBEF(nn.Module):
    def __init__(self,
                 text_encoder=None,
                 tokenizer=None,
                 config=None,
                 ):
        super().__init__()

        self.tokenizer = tokenizer
        self.distill = config['distill']

        self.visual_encoder = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12,
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))

        bert_config = BertConfig.from_json_file(config['bert_config'])
        self.te_bert = tBert.from_pretrained(text_encoder, add_pooling_layer=False)
        self.text_encoder = BertModel.from_pretrained(text_encoder, config=bert_config, add_pooling_layer=False)

        self.cls_head = nn.Sequential(
            nn.Linear(self.text_encoder.config.hidden_size,
                      self.text_encoder.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.text_encoder.config.hidden_size, 2)
        )

        if self.distill:
            self.visual_encoder_m = VisionTransformer(
                img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12,
                mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
            self.text_encoder_m = BertModel.from_pretrained(
                text_encoder, config=bert_config, add_pooling_layer=False)
            self.cls_head_m = nn.Sequential(
                nn.Linear(self.text_encoder.config.hidden_size,
                          self.text_encoder.config.hidden_size),
                nn.ReLU(),
                nn.Linear(self.text_encoder.config.hidden_size, 2)
            )

            self.model_pairs = [[self.visual_encoder, self.visual_encoder_m],
                                [self.text_encoder, self.text_encoder_m],
                                [self.cls_head, self.cls_head_m],
                                ]
            self.copy_params()
            self.momentum = 0.995

        # dependent textual entailment model & joint VE+TE model


        self.textual_cls_head = nn.Sequential(
            nn.Linear(self.text_encoder.config.hidden_size,
                      self.text_encoder.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.text_encoder.config.hidden_size, 2)
        )

        self.joint_cls_head = nn.Sequential(
            nn.Linear(self.text_encoder.config.hidden_size,
                      self.text_encoder.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.text_encoder.config.hidden_size, 2)
        )

        self.gate_net = nn.Linear(2*self.text_encoder.config.hidden_size, 2)

    def forward(self, image, text, hypo, targets, alpha=0, train=True,image_bool=None,text_bool=None):
        prediction_joint = None
        if image_bool is not None:
            image_embeds = self.visual_encoder(image)
            image_atts = torch.ones(
                image_embeds.size()[:-1], dtype=torch.long).to(image.device)
            output = self.text_encoder(hypo.input_ids,
                                       attention_mask=hypo.attention_mask,
                                       encoder_hidden_states=image_embeds,
                                       encoder_attention_mask=image_atts,
                                       return_dict=True
                                       )
            ve_hiddens = output.last_hidden_state[:, 0, :]
            prediction_ve = self.cls_head(ve_hiddens[image_bool[0]==1])
        if text_bool is not None:
            textual_entailment = self.te_bert(text.input_ids,
                                                attention_mask=text.attention_mask,
                                                return_dict=True
                                                )
            te_hiddens = textual_entailment.last_hidden_state[:, 0, :]
            prediction_te = self.textual_cls_head(te_hiddens[text_bool[0]==1])
        if image_bool is not None and text_bool is not None:
            joint_bool = image_bool[0] * text_bool[0]
            concated_outputs = torch.cat((ve_hiddens[joint_bool==1], te_hiddens[joint_bool==1]), dim=-1)
            gated_values = self.gate_net(concated_outputs)
            gated_values = nn.functional.softmax(gated_values, dim=-1)
            # B * 2
            g0 = gated_values[:, 0].unsqueeze(-1)
            g1 = gated_values[:, 1].unsqueeze(-1)
            joint_hiddens = g0 * ve_hiddens[joint_bool==1] + g1 * te_hiddens[joint_bool==1]
            prediction_joint = self.joint_cls_head(joint_hiddens)

        if train:

            loss_te = F.cross_entropy(prediction_te, targets[text_bool[0]==1]) if text_bool[0].sum()!=0 else torch.tensor(float(0),requires_grad=True)

            if self.distill:
                with torch.no_grad():
                    self._momentum_update()
                    image_embeds_m = self.visual_encoder_m(image)
                    output_m = self.text_encoder_m(text.input_ids,
                                                   attention_mask=text.attention_mask,
                                                   encoder_hidden_states=image_embeds_m,
                                                   encoder_attention_mask=image_atts,
                                                   return_dict=True
                                                   )
                    prediction_m = self.cls_head_m(
                        output_m.last_hidden_state[:, 0, :])

                loss_ve = (1-alpha)*F.cross_entropy(prediction_ve, targets) - alpha*torch.sum(
                    F.log_softmax(prediction_ve, dim=1)*F.softmax(prediction_m, dim=1), dim=1).mean()
            else:
                loss_ve = F.cross_entropy(prediction_ve, targets[image_bool[0]==1]) if image_bool[0].sum()!=0 else torch.tensor(float(0),requires_grad=True)

            
            loss_joint = F.cross_entropy(prediction_joint, targets[joint_bool==1]) if joint_bool.sum()!=0 else torch.tensor(float(0),requires_grad=True)
            if loss_joint==0 or loss_te==0 or loss_ve==0:
                print("true")
            return loss_ve, loss_te, loss_joint

        else:
            return prediction_joint

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + \
                    param.data * (1. - self.momentum)
