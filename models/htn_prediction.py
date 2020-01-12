# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from pytorch_transformers import BertModel
from utils import constants
from models.encoder_decoder import TransformerEncoder, TRSAEncoder
from models.sub_layers import CNNClassifier
from icecream import ic


class HTNPrediction(nn.Module):
    def __init__(self, n_token, num_labels, model_type, device, args):
        super(HTNPrediction, self).__init__()
        dropout, drop_att = args["dropout"], args["drop_att"]
        d_model = args["d_model"]
        n_head = args["n_head"]
        d_inner = d_model * 4
        d_head = d_model // n_head
        self.word_emb = nn.Embedding(n_token, d_model, padding_idx=constants.PAD)
        self.dropout = nn.Dropout(dropout)
        torch.nn.init.normal_(self.word_emb.weight, 0.0, 0.1)
        # Encoder
        if model_type == "transformer":
            self.encoder = TransformerEncoder(
                word_emb=self.word_emb, n_layer=args["n_layer"], n_head=n_head, d_model=d_model,
                d_head=d_head, d_inner=d_inner, dropout=dropout, drop_att=drop_att,
                pre_layer_norm=args["pre_layer_norm"], device=device
            )
        elif model_type == "bert":
            self.encoder = BertModel.from_pretrained("bert-base-uncased")
            d_model = 768
        elif model_type == "trsa":
            self.encoder = TRSAEncoder(
                word_emb=self.word_emb, n_layer=args["n_layer"], n_head=n_head, d_model=d_model,
                d_head=d_head, d_inner=d_inner, dropout=dropout, drop_att=drop_att,
                pre_layer_norm=args["pre_layer_norm"], device=device
            )
        else:
            assert False, "{} is not a legal model_type name".format(model_type)
        self.model_type = model_type
        self.use_cnn = not args["no_cnn"]
        if self.use_cnn:
            self.cnn = CNNClassifier(d_model, args["num_filters"], output_dim=d_model)
        else:
            self.cnn = None
        self.classifier = nn.Linear(d_model, num_labels)
        torch.nn.init.kaiming_normal_(self.classifier.weight, nonlinearity='relu')
        torch.nn.init.constant_(self.classifier.bias, 0.0)
        self.device = device

    def forward(self, batch):
        if self.model_type == "trsa":
            _, _, src_seqs, seg_select, *_, da, _ = batch
            enc_out, _ = self.encoder(src_seqs, seg_select)
            if self.use_cnn:
                cls_signal = self.cnn(enc_out)
            else:
                cls_signal = enc_out[:, -1, :]
        elif self.model_type == "bert":
            _, _, src_seq, segment_seq, src_mask, *_ = batch
            src_seq, segment_seq = src_seq.to(self.device), segment_seq.to(self.device)
            src_mask = src_mask.to(self.device)
            outputs = self.encoder(src_seq, segment_seq, attention_mask=src_mask)
            if self.use_cnn:
                cls_signal = self.cnn(outputs[0])
            else:
                cls_signal = outputs[1]
        else:
            _, _, src_seq, *_, da, _ = batch
            enc_out, _ = self.encoder(src_seq)
            if self.use_cnn:
                cls_signal = self.cnn(enc_out)
            else:
                cls_signal = enc_out[:, 0, :]
        cls_signal = self.dropout(cls_signal)
        labels = self.classifier(cls_signal)
        return labels


def main():
    import os
    import json
    from utils.tools import Tokenizer
    from utils.parsers import da_parse_opt
    args = da_parse_opt()
    ic(args)
    with open(os.path.join(args.data_dir, "vocab.json"), 'r') as f:
        vocabulary = json.loads(f.read())
    vocab, ivocab = vocabulary['vocab'], vocabulary['rev']
    tokenizer = Tokenizer(vocab, ivocab)
    device = torch.device("cuda")
    model = HTNPrediction(len(vocab), constants.act_len, args.model, device, vars(args))
    model = model.to(device)
    with open(os.path.join(args.data_dir, "test.json"), mode="r") as f:
        data = json.loads(f.read())
    if args.model == "mem_htn":
        from data_provider.mem_dataset import DataLoader, HTNDataset, collate_fn
        dataset = HTNDataset(data, tokenizer, 50)
        dataloader = DataLoader(
            dataset, batch_size=64, shuffle=False, num_workers=0, collate_fn=collate_fn
        )
    else:
        from data_provider.htn_dataset import DataLoader, HTNDataset, collate_fn
        dataset = HTNDataset(data, tokenizer, 128)
        dataloader = DataLoader(
            dataset, batch_size=64, shuffle=False, num_workers=0, collate_fn=collate_fn
        )
    dataiter = iter(dataloader)
    optimizer = torch.optim.Adam(
        filter(lambda x: x.requires_grad, model.parameters()),
        lr=1e-3, betas=(0.9, 0.98), eps=1e-09)
    loss_fc = torch.nn.BCEWithLogitsLoss()
    for i in range(10):
        batch = next(dataiter)
        *_, da, y = batch
        optimizer.zero_grad()
        model.zero_grad()
        labels = model(batch)
        da = da.to(device)
        loss = loss_fc(labels.view(-1, 1), da.view(-1, 1))
        loss.backward()
        ic(loss.item())
        optimizer.step()


if __name__ == '__main__':
    main()
