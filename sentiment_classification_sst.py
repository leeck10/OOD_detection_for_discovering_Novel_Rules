import re
import torch
import pytorch_lightning as pl
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    BertConfig,
    BertModel,
    get_constant_schedule_with_warmup,
    get_constant_schedule,
    BertForTokenClassification,
    BertPreTrainedModel
)
from ..utils.util import accuracy_precision_recall_f1
import csv

from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, NLLLoss

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, sents, labels, rationales=None):
        self.sents = sents
        self.labels = labels
        self.rationales = rationales

    def __getitem__(self, index):
        sent = self.sents[index]
        label = self.labels[index]
        if self.rationales is not None:
            rationale = self.rationales[index]
            return sent, label, rationale
        return sent, label

    def __len__(self):
        return len(self.sents)

def my_collate_fn(batch):
    collate_sents = []
    collate_labels = []

    for sent, label in batch:
        collate_sents.append(sent)
        collate_labels.append(label)
    return [collate_sents, torch.as_tensor(collate_labels)]
    
def my_collate_fn_token(batch):
    collate_sents = []
    collate_labels = []

    for sent, label in batch:
        collate_sents.append(sent)
        collate_labels.append(torch.as_tensor(label))
    collate_labels = torch.nn.utils.rnn.pad_sequence(collate_labels, batch_first=True)
    return [collate_sents, collate_labels]

def my_collate_fn_rationale(batch):
    collate_sents = []
    collate_labels = []
    collate_rationales = []

    for sent, label, rationale in batch:
        collate_sents.append(sent)
        collate_labels.append(label)
        collate_rationales.append(torch.as_tensor(rationale))
    collate_rationales = torch.nn.utils.rnn.pad_sequence(collate_rationales, batch_first=True)
    return [collate_sents, torch.as_tensor(collate_labels), collate_rationales]


def load_sst(path, tokenizer, dataset, num_labels, rationale_path, token_cls, lower=False):
    dataset_orig = []
    sents, labels, rationales = [], [], []
    skipped_count = 0  # 건너뛴 행 수

    if num_labels == 15:
        label_idx = {
            'entailment_HS': 0, 'entailment_PS': 1, 'entailment_COUNT': 2, 'entailment_PA': 3, 'entailment_ES': 4,
            'contradiction_CW_adj': 5, 'contradiction_CW_noun': 6, 'contradiction_CV': 7, 'contradiction_NS': 8,
            'contradiction_SOS': 9, 'contradiction_IH': 10, 'contradiction_NI': 11,
            'neutral_AM': 12, 'neutral_CON': 13, 'neutral_SSNCV': 14,
        }
    # if num_labels == 19:
    #     label_idx = {
    #         'entailment_HS': 0, 'entailment_PS': 1, 'entailment_COUNT': 2, 'entailment_PA': 3, 'entailment_ES': 4,
    #         'contradiction_CW_adj': 5, 'contradiction_CW_noun': 6, 'contradiction_CV': 7, 'contradiction_NS': 8,
    #         'contradiction_SOS': 9, 'contradiction_IH': 10, 'contradiction_NI': 11,
    #         'neutral_AM': 12, 'neutral_CON': 13, 'neutral_SSNCV': 14, 'neutral_CA': 15, 'neutral_EI' : 16, 
    #         'entailment_RG': 17, 'neutral_VS' : 18
    #     }
    else:
        label_idx = {'entailment': 0, 'neutral': 1, 'contradiction': 2}

    if dataset == 'coco':
        with open(path, mode="r", encoding="utf-8") as f:
            lines = f.readlines()
            if token_cls:
                with open(rationale_path, mode="r", encoding="utf-8") as fr:
                    lines_rationale = fr.readlines()
                for line, line_rationale in zip(lines, lines_rationale):
                    line = line.split('\t')
                    if len(line) < 3:  # Ensure correct format
                        skipped_count += 1
                        continue
                    if line[0].strip() not in label_idx:  # Skip invalid labels
                        skipped_count += 1
                        continue
                    label = list(map(int, line_rationale.strip().split()))
                    sent = (line[1].strip(), line[2].strip())
                    sents.append(sent)
                    labels.append(label)
                    dataset_orig.append((sent, label))
                print(f"Skipped lines with invalid labels or malformed data: {skipped_count}")
                return MyDataset(sents, labels), dataset_orig
            elif rationale_path is not None:
                with open(rationale_path, mode="r", encoding="utf-8") as fr:
                    lines_rationale = fr.readlines()
                for line, line_rationale in zip(lines, lines_rationale):
                    line = line.split('\t')
                    if len(line) < 3:  # Ensure correct format
                        skipped_count += 1
                        continue
                    if line[0].strip() not in label_idx:  # Skip invalid labels
                        skipped_count += 1
                        continue
                    label = label_idx[line[0].strip()]
                    sent = (line[1].strip(), line[2].strip())
                    try:
                        rationale = list(map(int, line_rationale.strip().split(' ')))
                    except Exception as e:
                        print(f"Error processing rationale: {line} with error: {e}")
                        import pdb; pdb.set_trace()
                    sents.append(sent)
                    labels.append(label)
                    rationales.append(rationale)
                    dataset_orig.append((sent, label, rationale))
                print(f"Skipped lines with invalid labels or malformed data: {skipped_count}")
                return MyDataset(sents, labels, rationales), dataset_orig
            else:
                for line in lines:
                    line = line.split('\t')
                    if len(line) < 3:  # Ensure correct format
                        print(f"Skipping malformed line: {line}")
                        skipped_count += 1
                        continue
                    if line[0].strip() not in label_idx:  # Skip invalid labels
                        skipped_count += 1
                        continue
                    try:
                        label = label_idx[line[0].strip()]
                        if num_labels == 15 and label == 15:
                        # if num_labels == 19 and label == 19:
                            continue
                        sent = (line[1].strip(), line[2].strip())
                    except Exception as e:
                        print(f"Error processing line: {line} with error: {e}")
                        import pdb; pdb.set_trace()
                    sents.append(sent)
                    labels.append(label)
                    dataset_orig.append((sent, label))
                print(f"Skipped lines with invalid labels or malformed data: {skipped_count}")
                return MyDataset(sents, labels), dataset_orig



# def load_sst(path, tokenizer, dataset, num_labels, rationale_path, token_cls, lower=False):
#     dataset_orig = []
#     sents, labels, rationales= [], [], []
#     skipped_count = 0  # 건너뛴 행 수

#     if num_labels == 14:
#         label_idx = {'entailment_HS': 0,'entailment_PS': 1,'entailment_COUNT': 2,'entailment_PA': 3,'entailment_ES': 4, 'contradiction_CW_adj': 5,'contradiction_CW_noun': 6,'contradiction_CV': 7,'contradiction_NS': 8,'contradiction_SOS': 9,'contradiction_IH': 10,'contradiction_NI': 11, 'neutral_AM': 12,'neutral_CON': 13,'neutral_SSNCV': 14}
#     else:
#         label_idx = {'entailment': 0, 'neutral': 1, 'contradiction': 2}

#     if dataset == 'coco':
#         with open(path, mode="r", encoding="utf-8") as f:
#             lines = f.readlines()
#             if token_cls:
#                 with open(rationale_path, mode="r", encoding="utf-8") as fr:
#                     lines_rationale = fr.readlines()
#                 for line, line_rationale in zip(lines, lines_rationale):
#                     line = line.split('\t')
#                     if num_labels == 14 and label_idx[line[0].strip()] == 14:
#                         continue
#                     label = list(map(int, line_rationale.strip().split()))
#                     sent = (line[1].strip(), line[2].strip())
#                     sents.append(sent)
#                     labels.append(label)
#                     dataset_orig.append((sent, label))
#                 return MyDataset(sents, labels), dataset_orig
#             elif rationale_path is not None:
#                 with open(rationale_path, mode="r", encoding="utf-8") as fr:
#                     lines_rationale = fr.readlines()
#                 for line, line_rationale in zip(lines, lines_rationale):
#                     line = line.split('\t')
#                     label = label_idx[line[0].strip()]
#                     if num_labels == 14 and label == 14:
#                         continue
#                     sent = (line[1].strip(), line[2].strip())
#                     try:
#                         rationale = list(map(int, line_rationale.strip().split(' ')))
#                     except:
#                         import pdb; pdb.set_trace()
#                     sents.append(sent)
#                     labels.append(label)
#                     rationales.append(rationale)
#                     dataset_orig.append((sent, label, rationale))
#                 return MyDataset(sents, labels, rationales), dataset_orig
#             else:
#                 # for line in lines:
                    
#                 #     line = line.split('\t')
#                 #     try:
#                 #         label = label_idx[line[0].strip()]
#                 #         if num_labels == 14 and label == 14:
#                 #             continue
#                 #         sent = (line[1].strip(), line[2].strip())
#                 #     except:
#                 #         import pdb; pdb.set_trace()
#                 #     sents.append(sent)
#                 #     labels.append(label)
#                 #     dataset_orig.append((sent, label))
                
#                 # 오류나서 skip 코드 추가
#                 for line in lines:
#                     line = line.split('\t')
                
#                     if len(line) < 3:  # Ensure that there are at least 3 elements in the line
#                         print(f"Skipping malformed line: {line}")
#                         continue  # Skip this line if it's malformed
                
#                     try:
#                         label = label_idx.get(line[0].strip(), None)
#                         if label is None or (num_labels == 14 and label == 14):
#                             continue
#                         sent = (line[1].strip(), line[2].strip())
#                     except Exception as e:
#                         print(f"Error processing line: {line} with error: {e}")
#                         import pdb; pdb.set_trace()
                
#                     sents.append(sent)
#                     labels.append(label)
#                     dataset_orig.append((sent, label))

#                 return MyDataset(sents, labels), dataset_orig

                
    else:
        with open(path, mode="r", encoding="utf-8") as f:
            lines = csv.reader(f)
            lines.__next__()

            if token_cls:
                with open(rationale_path, mode="r", encoding="utf-8") as fr:
                    lines_rationale = fr.readlines()
                for line, line_rationale in zip(lines, lines_rationale):
                    if num_labels == 14 and label_idx[line[1].strip()] == 14:
                        continue
                    label = list(map(int, line_rationale.strip().split()))
                    sent = (line[2].strip(), line[3].strip())
                    sents.append(sent)
                    labels.append(label)
                    dataset_orig.append((sent, label))
                return MyDataset(sents, labels), dataset_orig
            elif rationale_path is not None:
                with open(rationale_path, mode="r", encoding="utf-8") as fr:
                    lines_rationale = fr.readlines()
                for line, line_rationale in zip(lines, lines_rationale):
                    label = label_idx[line[1].strip()]
                    if num_labels == 14 and label == 14:
                        continue
                    sent = (line[2].strip(), line[3].strip())
                    rationale = list(map(int, line_rationale.strip().split(' ')))
                    sents.append(sent)
                    labels.append(label)
                    rationales.append(rationale)
                    dataset_orig.append((sent, label, rationale))
                return MyDataset(sents, labels, rationales), dataset_orig
            else:
                for line in lines:
                    label = label_idx[line[1].strip()]
                    if num_labels == 14 and label == 14:
                        continue
                    sent = (line[2].strip(), line[3].strip())
                    sents.append(sent)
                    labels.append(label)
                    dataset_orig.append((sent, label))
                return MyDataset(sents, labels), dataset_orig

class SentimentClassificationSST(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.tokenizer = BertTokenizer.from_pretrained(self.hparams.model)

    def prepare_data(self):
        # assign to use in dataloaders
        self.hparams.val_filename = "datasets/sci_chatgpt/test_data/snli_1.0_dev_output_edit.txt" # 고정시킴(inference 위해)
        # self.hparams.val_filename = "datasets/sci_chatgpt/test_data/19rules_ml_val_data.txt"
        # self.hparams.val_filename = "datasets/sci_chatgpt/test_data/15rules_ml_val_data.txt"
        
        if not hasattr(self, "train_dataset") or not hasattr(
            self, "train_dataset_orig"
        ):
            self.train_dataset, self.train_dataset_orig = load_sst(
                self.hparams.train_filename, self.tokenizer, self.hparams.dataset, self.hparams.num_labels, rationale_path=self.hparams.train_rationale, token_cls=self.hparams.token_cls
            )
        if not hasattr(self, "val_dataset") or not hasattr(self, "val_dataset_orig"):
            self.val_dataset, self.val_dataset_orig = load_sst(
                self.hparams.val_filename, self.tokenizer, self.hparams.dataset, self.hparams.num_labels, rationale_path=self.hparams.val_rationale, token_cls=self.hparams.token_cls
            )

    def train_dataloader(self, shuffle=True):
        if self.hparams.token_cls:
            collate_fn = my_collate_fn_token
        elif self.hparams.train_rationale is not None:
            collate_fn = my_collate_fn_rationale
        else:
            collate_fn = my_collate_fn
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.hparams.batch_size, shuffle=shuffle, collate_fn=collate_fn, num_workers=8
        )

    def val_dataloader(self):
        if self.hparams.token_cls:
            collate_fn = my_collate_fn_token
        elif self.hparams.val_rationale is not None:
            collate_fn = my_collate_fn_rationale
        else:
            collate_fn = my_collate_fn
        return torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.hparams.batch_size, collate_fn=collate_fn, num_workers=8
        )

    def training_step(self, batch, batch_idx=None):
        # input_ids, mask, labels = batch
        inputs = self.tokenizer.batch_encode_plus(batch[0], pad_to_max_length=True, return_tensors='pt').to('cuda')
        input_ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']
        labels = batch[1]
        rationale_ids = None

        if self.hparams.token_cls:
            loss, logits = self.forward(input_ids, mask, token_type_ids, labels=labels)
            logit = torch.masked_select(logits.argmax(-1).view(-1), (mask.view(-1)) == 1)
            label = torch.masked_select(labels.view(-1), (mask.view(-1)) == 1)
            acc, _, _, f1 = accuracy_precision_recall_f1(
                logit, label, average=True
            )
        else:
            if len(batch) == 3:
                rationale_ids = batch[2]
            # import pdb; pdb.set_trace()
            logits = self.forward(input_ids, mask, token_type_ids, rationale_ids=rationale_ids)[0]
            #import pdb; pdb.set_trace()
            loss = torch.nn.functional.cross_entropy(logits, labels, reduction="none").mean(
                -1
            )
            acc, _, _, f1 = accuracy_precision_recall_f1(
                logits.argmax(-1), labels, average=True
            )

        outputs_dict = {
            "acc": acc,
            "f1": f1,
        }

        outputs_dict = {
            "loss": loss,
            **outputs_dict,
            "log": outputs_dict,
            "progress_bar": outputs_dict,
        }

        outputs_dict = {
            "{}{}".format("" if self.training else "val_", k): v
            for k, v in outputs_dict.items()
        }

        return outputs_dict

    def validation_step(self, batch, batch_idx=None):
        return self.training_step(batch, batch_idx)

    def validation_epoch_end(self, outputs):

        outputs_dict = {
            k: sum(e[k] for e in outputs) / len(outputs) for k in ("val_acc", "val_f1")
        }

        outputs_dict = {
            "val_loss": -outputs_dict["val_f1"],
            **outputs_dict,
            "log": outputs_dict,
        }

        return outputs_dict

    def configure_optimizers(self):
        optimizers = [
            torch.optim.Adam(self.parameters(), self.hparams.learning_rate),
        ]
        schedulers = [
            {
                "scheduler": get_constant_schedule_with_warmup(optimizers[0], 200),
                "interval": "step",
            },
        ]

        return optimizers, schedulers


class BertSentimentClassificationSST(SentimentClassificationSST):
    def __init__(self, hparams):
        super().__init__(hparams)

        # config = BertConfig.from_pretrained(self.hparams.model, hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1) # dropout 비율 수정
        config = BertConfig.from_pretrained(self.hparams.model)
        config.num_labels = self.hparams.num_labels
        config.rationale = None
        config.output_hidden_states=True
        if self.hparams.token_cls:
            self.net = BertForTokenClassification.from_pretrained(
            self.hparams.model, config=config
        )
        else:
            if self.hparams.train_rationale is not None:
                config.rationale = True
            self.net = BertForSequenceClassification.from_pretrained(
                self.hparams.model, config=config
            )

    def forward(self, input_ids, mask, token_type_ids, rationale_ids=None, labels=None):
        if self.hparams.token_cls:
            return self.net(input_ids=input_ids, attention_mask=mask, token_type_ids=token_type_ids, labels=labels)
        else:
            return self.net(input_ids=input_ids, attention_mask=mask, token_type_ids=token_type_ids, rationale_ids=rationale_ids, labels=labels)
        
        
#########################################################################################################
#########################################################################################################
#########################################################################################################
#########################################################################################################
#########################################################################################################



def softmax_with_temperature(logits, T):
    logits = logits / T  # Temperature scaling
    max_logits = torch.max(logits, dim=1, keepdim=True)[0]
    exp_logits = torch.exp(logits - max_logits)
    sum_exp_logits = torch.sum(exp_logits, dim=1, keepdim=True)
    y = exp_logits / sum_exp_logits
    return y



class BertForSequenceClassification_hhs(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        rationale_ids=None,
        labels=None,
        temperature=None,
        epsilon=None
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import BertTokenizer, BertForSequenceClassification
        import torch

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)

        loss, logits = outputs[:2]

        """
        
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]
        pooled_output.requires_grad_(True)
        #pooled_output.retain_grad() # 중간 미분
        #pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        #print('len(outputs) :', len(outputs))
        #print('outputs[0].size() :', outputs[0].size())
        #print('outputs[1].size() :', outputs[1].size())
        #print('len(outputs[2]) :', len(outputs[2]))
        #print('pooled_output.size() :', pooled_output.size())
        #print('logits.size() :', logits.size())
        #exit(1)


        #outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        logits = softmax_with_temperature(logits, temperature)
        logits2 = torch.log(logits)

        labels = logits.argmax(dim=-1)
        #loss_fct = CrossEntropyLoss()
        loss_fct = NLLLoss()
        loss = loss_fct(logits2.view(-1, self.num_labels), labels.view(-1))

        loss.backward()
        
        #epsilon = 0.002
        
        new_pooled_output = pooled_output - epsilon*torch.sign(pooled_output.grad)
        
        #print('pooled_output.grad.size() :', pooled_output.grad.size())
        #print('pooled_output.grad.size() :', pooled_output.grad.size())
        #print('new_pooled_output.size() :', new_pooled_output.size())
        #exit(1)

        new_logits = self.classifier(new_pooled_output)
        
        new_logits = softmax_with_temperature(new_logits, temperature)

        '''
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        '''
        
        
        #return outputs  # (loss), logits, (hidden_states), (attentions)
        return logits, new_logits


class BertForSequenceClassification_hhs2(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        rationale_ids=None,
        labels=None,
        temperature=None,
        epsilon=None
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import BertTokenizer, BertForSequenceClassification
        import torch

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)

        loss, logits = outputs[:2]

        """
        
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]
        #pooled_output.requires_grad_(True)
        #pooled_output.retain_grad() # 중간 미분
        pooled_output = self.dropout(pooled_output) # cls dropout 하는 부분
        #logits = self.classifier(pooled_output)
        logits = pooled_output

        #print('len(outputs) :', len(outputs))
        #print('outputs[0].size() :', outputs[0].size())
        #print('outputs[1].size() :', outputs[1].size())
        #print('len(outputs[2]) :', len(outputs[2]))
        #print('pooled_output.size() :', pooled_output.size())
        #print('logits.size() :', logits.size())
        #exit(1)


        #outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        '''
        logits = softmax_with_temperature(logits, temperature)
        logits2 = torch.log(logits)

        labels = logits.argmax(dim=-1)
        #loss_fct = CrossEntropyLoss()
        loss_fct = NLLLoss()
        loss = loss_fct(logits2.view(-1, self.num_labels), labels.view(-1))

        loss.backward()
        
        #epsilon = 0.002
        
        new_pooled_output = pooled_output - epsilon*torch.sign(pooled_output.grad)
        
        #print('pooled_output.grad.size() :', pooled_output.grad.size())
        #print('pooled_output.grad.size() :', pooled_output.grad.size())
        #print('new_pooled_output.size() :', new_pooled_output.size())
        #exit(1)

        new_logits = self.classifier(new_pooled_output)
        
        new_logits = softmax_with_temperature(new_logits, temperature)
        '''
        '''
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        '''
        
        
        #return outputs  # (loss), logits, (hidden_states), (attentions)
        return logits


class BertSentimentClassificationSST_hhs(SentimentClassificationSST):
    def __init__(self, hparams):
        super().__init__(hparams)

        config = BertConfig.from_pretrained(self.hparams.model)
        config.attention_probs_dropout_prob = 0.0
        config.hidden_dropout_prob = 0.0
        #print('config :', config)
        #exit(1)
        config.num_labels = self.hparams.num_labels
        config.rationale = None
        config.output_hidden_states=True
        if self.hparams.token_cls:
            self.net = BertForTokenClassification.from_pretrained(
            self.hparams.model, config=config
        )
        else: # 대충 이걸로 돌아갈 예정
            if self.hparams.train_rationale is not None:
                config.rationale = True
            self.net = BertForSequenceClassification_hhs.from_pretrained(
                self.hparams.model, config=config
            )
    # temperature 인자 추가
    def forward(self, input_ids, mask, token_type_ids, rationale_ids=None, labels=None, temperature=None, epsilon=None):
        if self.hparams.token_cls:
            return self.net(input_ids=input_ids, attention_mask=mask, token_type_ids=token_type_ids, labels=labels)
        else:
            return self.net(input_ids=input_ids, attention_mask=mask, token_type_ids=token_type_ids, rationale_ids=rationale_ids, labels=labels, temperature=temperature, epsilon=epsilon)


class BertSentimentClassificationSST_hhs2(SentimentClassificationSST):
    def __init__(self, hparams):
        super().__init__(hparams)

        config = BertConfig.from_pretrained(self.hparams.model)
        #config.attention_probs_dropout_prob = 0.0
        #config.hidden_dropout_prob = 0.0
        #print('config :', config)
        #exit(1)
        config.num_labels = self.hparams.num_labels
        config.rationale = None
        config.output_hidden_states=True
        if self.hparams.token_cls:
            self.net = BertForTokenClassification.from_pretrained(
            self.hparams.model, config=config
        )
        else: # 대충 이걸로 돌아갈 예정
            if self.hparams.train_rationale is not None:
                config.rationale = True
            self.net = BertForSequenceClassification_hhs2.from_pretrained(
                self.hparams.model, config=config
            )
    # temperature 인자 추가
    def forward(self, input_ids, mask, token_type_ids, rationale_ids=None, labels=None, temperature=None, epsilon=None):
        if self.hparams.token_cls:
            return self.net(input_ids=input_ids, attention_mask=mask, token_type_ids=token_type_ids, labels=labels)
        else:
            return self.net(input_ids=input_ids, attention_mask=mask, token_type_ids=token_type_ids, rationale_ids=rationale_ids, labels=labels)


#########################################################################################################
#########################################################################################################
#########################################################################################################
#########################################################################################################
#########################################################################################################


#added by hanjy
class BertSentimentClassificationSST_hjy(SentimentClassificationSST):
    def __init__(self, hparams):
        super().__init__(hparams)
        config = BertConfig.from_pretrained(self.hparams.model)
        config.num_labels = self.hparams.num_labels
        config.rationale = None
        if self.hparams.token_cls:
            self.net = BertForTokenClassification.from_pretrained(
            self.hparams.model, config=config
        )
        else:
            if self.hparams.train_rationale is not None:
                config.rationale = True
            self.net = BertForSequenceClassification.from_pretrained(
                self.hparams.model, config=config
            )

    def forward(self, input_ids, attention_mask, token_type_ids, rationale_ids=None, labels=None):
        print("In forward, token_cls is:", self.hparams.token_cls)
        outputs = self.net(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
        sequence_output = outputs[0]
        cls_token_embedding = sequence_output[:, 0, :]
        return cls_token_embedding, outputs, sequence_output
        

class RecurrentSentimentClassificationSST(SentimentClassificationSST):
    def __init__(self, hparams):
        super().__init__(hparams)

        self.emb = BertForSequenceClassification.from_pretrained(
            hparams.model
        ).bert.embeddings.word_embeddings.requires_grad_(False)

        self.gru = torch.nn.GRU(
            input_size=self.emb.embedding_dim,
            hidden_size=self.emb.embedding_dim,
            batch_first=True,
        )

        self.classifier = torch.nn.Linear(self.emb.embedding_dim, 5)

    def forward(self, input_ids, mask, labels=None):
        x = self.emb(input_ids)
        x = torch.nn.utils.rnn.pack_padded_sequence(
            x, mask.sum(-1), batch_first=True, enforce_sorted=False
        )

        _, h = self.gru(x)

        return (self.classifier(h[0]),)
