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
        self.mask_patch_rate = config['mask_patch_rate'] if 'mask_patch_rate' in config else None
        self.visual_encoder = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12,
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))

        bert_config = BertConfig.from_json_file(config['bert_config'])

        self.text_encoder = BertModel.from_pretrained(
            text_encoder, config=bert_config, add_pooling_layer=False)

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
        self.te_bert = tBert.from_pretrained(
            text_encoder, add_pooling_layer=False)

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

    def forward(self, image, text, hypo, targets=None, mode=0, alpha=0, train=True):

        if train:
            targets_ve ,targets_te = targets.clone(),targets.clone()
            targets_te[mode>=2] = -1
            targets_ve[(mode == 1) + (mode == 3)] = -1
            image_embeds, attn = self.visual_encoder(image)
            image_atts = torch.ones(
                image_embeds.size()[:-1], dtype=torch.long).to(image.device)

            textual_entailment = self.te_bert(text.input_ids,
                                                attention_mask=text.attention_mask,
                                                return_dict=True
                                                )
            te_hiddens = textual_entailment.last_hidden_state[:, 0, :]
            prediction_te = self.textual_cls_head(te_hiddens)
            loss_te = F.cross_entropy(prediction_te, targets_te, ignore_index=-1)

            if self.mask_patch_rate:
                attn_cls = attn[:,:,0].detach().sum(dim=1)
                extra_augment = 4
                entail_pos = torch.where(targets_ve==1)[0][:extra_augment]
                extra_image_embeds = image_embeds[entail_pos]
                extra_attn = attn_cls[entail_pos]
                _, wait_mask_pos = extra_attn.topk(k=int(attn_cls.size(-1)*self.mask_patch_rate))
                extra_image_atts = torch.ones(extra_image_embeds.size()[:-1], dtype=torch.long).to(image.device)
                for bs, topks in enumerate(wait_mask_pos):
                    extra_image_atts[bs][topks]=0
                extra_targets = torch.zeros(entail_pos.size(0), dtype=torch.long).to(image.device)
                image_embeds=torch.cat((image_embeds,extra_image_embeds),dim=0)
                image_atts=torch.cat((image_atts,extra_image_atts),dim=0)
                extra_hypo_input_ids = hypo.input_ids[entail_pos]
                extra_hypo_atts = hypo.attention_mask[entail_pos]
                hypo.input_ids = torch.cat((hypo.input_ids,extra_hypo_input_ids),dim=0)
                hypo.attention_mask = torch.cat((hypo.attention_mask,extra_hypo_atts),dim=0)
                targets_ve = torch.cat((targets_ve,extra_targets))
                te_hiddens = torch.cat((te_hiddens,te_hiddens[entail_pos]),dim=0)
                

            visual_entailment = self.text_encoder(hypo.input_ids,
                                        attention_mask=hypo.attention_mask,
                                        encoder_hidden_states=image_embeds,
                                        encoder_attention_mask=image_atts,
                                        return_dict=True
                                        )
            ve_hiddens = visual_entailment.last_hidden_state[:, 0, :]
            prediction_ve = self.cls_head(ve_hiddens)

            concated_outputs = torch.cat((ve_hiddens, te_hiddens), dim=-1)
            gated_values = self.gate_net(concated_outputs)
            gated_values = nn.functional.softmax(gated_values, dim=-1)
            # B * 2
            g0 = gated_values[:, 0].unsqueeze(-1)
            g1 = gated_values[:, 1].unsqueeze(-1)
            joint_hiddens = g0 * ve_hiddens + g1 * te_hiddens
            prediction_joint = self.joint_cls_head(joint_hiddens)

            

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
                loss_ve = F.cross_entropy(prediction_ve, targets_ve, ignore_index=-1)

            
            loss_joint = F.cross_entropy(prediction_joint, targets_ve, ignore_index=-1)

            return loss_ve, loss_te, loss_joint

        else:
            image_embeds, attn = self.visual_encoder(image)
            image_atts = torch.ones(
                image_embeds.size()[:-1], dtype=torch.long).to(image.device)

            textual_entailment = self.te_bert(text.input_ids,
                                                attention_mask=text.attention_mask,
                                                return_dict=True
                                                )
            te_hiddens = textual_entailment.last_hidden_state[:, 0, :]

            output = self.text_encoder(hypo.input_ids,
                                       attention_mask=hypo.attention_mask,
                                       encoder_hidden_states=image_embeds,
                                       encoder_attention_mask=image_atts,
                                       return_dict=True
                                       )
            ve_hiddens = output.last_hidden_state[:, 0, :]

            concated_outputs = torch.cat((ve_hiddens, te_hiddens), dim=-1)
            gated_values = self.gate_net(concated_outputs)
            gated_values = nn.functional.softmax(gated_values, dim=-1)
            # B * 2
            g0 = gated_values[:, 0].unsqueeze(-1)
            g1 = gated_values[:, 1].unsqueeze(-1)
            joint_hiddens = g0 * ve_hiddens + g1 * te_hiddens
            prediction_joint = self.joint_cls_head(joint_hiddens)
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
