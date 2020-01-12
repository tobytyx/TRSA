# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.Beam import Beam, get_inst_idx_to_tensor_position_map, collate_active_info, collect_hypothesis_and_scores
from utils import constants
from models.encoder_decoder import TransformerEncoder, TransformerDecoder, TRSAEncoder, RelDecoder
from models.encoder_decoder import RelHDSADecoder, HDSADecoder
from models.sub_layers import CNNClassifier


class HTNetwork(nn.Module):
    def __init__(self, n_token, num_labels, device, args):
        super(HTNetwork, self).__init__()
        dropout, drop_att = args["dropout"], args["drop_att"]
        d_model = args["d_model"]
        n_head = args["n_head"]
        self.encoder_type, self.decoder_type = args["encoder"].lower(), args["decoder"].lower()
        d_inner = d_model * 4
        d_head = d_model // n_head
        self.word_emb = nn.Embedding(n_token, d_model, padding_idx=constants.PAD)
        torch.nn.init.normal_(self.word_emb.weight, 0.0, 0.1)
        # Encoder
        rel_position = False
        if self.encoder_type == "trsa":
            self.encoder = TRSAEncoder(
                word_emb=self.word_emb, n_layer=args["en_layer"], n_head=n_head, d_model=d_model,
                d_head=d_head, d_inner=d_inner, dropout=dropout, drop_att=drop_att,
                pre_layer_norm=args["pre_layer_norm"], device=device
            )
            rel_position = True
        elif self.encoder_type == "transformer":
            self.encoder = TransformerEncoder(
                word_emb=self.word_emb, n_layer=args["en_layer"], n_head=n_head, d_model=d_model,
                d_head=d_head, d_inner=d_inner, dropout=dropout, drop_att=drop_att,
                pre_layer_norm=args["pre_layer_norm"], device=device
            )
        else:
            assert False, "{} is not a legal encoder name".format(self.encoder_type)

        # Decoder
        if self.decoder_type == "hdsa":
            if rel_position:
                self.decoder = RelHDSADecoder(
                    word_emb=self.word_emb, n_head=n_head, d_model=d_model,
                    d_head=d_head, d_inner=d_inner, dropout=dropout, drop_att=drop_att,
                    pre_layer_norm=args["pre_layer_norm"], device=device
                )
            else:
                self.decoder = HDSADecoder(
                    word_emb=self.word_emb, n_head=n_head, d_model=d_model,
                    d_head=d_head, d_inner=d_inner, dropout=dropout, drop_att=drop_att,
                    pre_layer_norm=args["pre_layer_norm"], device=device
                )
        elif self.decoder_type == "transformer":
            if rel_position:
                self.decoder = RelDecoder(
                    word_emb=self.word_emb, n_layer=args["de_layer"], n_head=n_head, d_model=d_model,
                    d_head=d_head, d_inner=d_inner, dropout=dropout, drop_att=drop_att,
                    pre_layer_norm=args["pre_layer_norm"], device=device
                )
            else:
                self.decoder = TransformerDecoder(
                    word_emb=self.word_emb, n_layer=args["de_layer"], n_head=n_head, d_model=d_model,
                    d_head=d_head, d_inner=d_inner, dropout=dropout, drop_att=drop_att,
                    pre_layer_norm=args["pre_layer_norm"], device=device
                )
        else:
            assert False, "{} is not a legal decoder name".format(self.decoder_type)
        self.use_cnn = not args["no_cnn"]
        if self.use_cnn:
            self.cnn = CNNClassifier(d_model, args["num_filters"], output_dim=d_model)
        else:
            self.cnn = None
        self.classifier = nn.Linear(d_model, num_labels)
        torch.nn.init.kaiming_normal_(self.classifier.weight, nonlinearity='relu')
        torch.nn.init.constant_(self.classifier.bias, 0.0)
        self.tgt_word_prj = nn.Linear(d_model, n_token, bias=False)
        torch.nn.init.kaiming_normal_(self.tgt_word_prj.weight, nonlinearity='relu')
        if args["no_share_weight"]:
            self.x_logit_scale = 1
        else:
            self.tgt_word_prj.weight = self.word_emb.weight
            self.x_logit_scale = d_model ** -0.5
        self.device = device
        self.n_bm = args["n_bm"]
        self.th = args["th"]

    def forward(self, batch, joint=True):
        if self.encoder_type == "trsa":
            _, _, src_seqs, seg_select, *_, da, tgt_seq = batch
            src_seq = src_seqs[-1]
            enc_out, _ = self.encoder(src_seqs, seg_select)
        else:
            _, _, src_seq, *_, da, tgt_seq = batch
            enc_out, _ = self.encoder(src_seq)
        dec_out = self.decoder(tgt_seq[:, :-1], src_seq, da, enc_out)
        logits = self.tgt_word_prj(dec_out) * self.x_logit_scale
        if joint:
            if self.use_cnn:
                cls_signal = self.cnn(enc_out)
            else:
                cls_signal = enc_out[:, -1, :] if self.encoder_type == "trsa" else enc_out[:, 0, :]
            labels = self.classifier(cls_signal)
            return labels, logits
        return None, logits

    def generate(self, batch, joint):
        """
        use beam search
        :param batch:
        :param joint: use gt action or predicted action
        :return: list, b * sentence
        :return: b * len * len, attn
        """
        n_bm = self.n_bm
        with torch.no_grad():
            # -- Encode
            if self.encoder_type == "trsa":
                _, _, src_seqs, seg_select, *_, da, tgt_seq = batch
                src_seq = src_seqs[-1]
                enc_out, attn = self.encoder(src_seqs, seg_select)
            else:
                _, _, src_seq, *_, da, tgt_seq = batch
                enc_out, attn = self.encoder(src_seq)
            src_seq, da = src_seq.to(self.device), da.to(self.device)
            if joint:
                if self.use_cnn:
                    cls_signal = self.cnn(enc_out)
                else:
                    cls_signal = enc_out[:, -1, :] if self.encoder_type == "trsa" else enc_out[:, 0, :]
                labels = self.classifier(cls_signal)
                da = torch.sigmoid(labels)
                da = (da > self.th).long()
            dialogue_actions = torch.zeros_like(da)
            dialogue_actions.copy_(da)
            dialogue_actions = dialogue_actions.long()
            da = da.float()
            # -- Repeat data for beam search
            n_inst, len_s, d_h = enc_out.size()
            src_seq = src_seq.repeat(1, n_bm).view(n_inst * n_bm, len_s)
            enc_out = enc_out.repeat(1, n_bm, 1).view(n_inst * n_bm, len_s, d_h)
            da = da.repeat(1, n_bm).view(n_inst * n_bm, -1)

            # -- Prepare beams
            inst_dec_beams = [Beam(n_bm, device=self.device) for _ in range(n_inst)]

            # -- Bookkeeping for active or not
            active_inst_idx_list = list(range(n_inst))
            inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

            # -- Decode
            for len_dec_seq in range(1, constants.RESP_MAX_LEN + 1):
                active_inst_idx_list = self.beam_decode_step(
                    inst_dec_beams, len_dec_seq, da, src_seq, enc_out, inst_idx_to_position_map, n_bm)
                if not active_inst_idx_list:
                    break  # all instances have finished their path to <EOS>
                da, src_seq, enc_out, inst_idx_to_position_map = collate_active_info(
                    da, src_seq, enc_out,
                    inst_idx_to_position_map=inst_idx_to_position_map, active_inst_idx_list=active_inst_idx_list,
                    n_bm=n_bm, device=self.device)

        batch_hyp, _ = collect_hypothesis_and_scores(inst_dec_beams, n_bm)
        result = []
        for hyps in batch_hyp:
            finished = False
            for r in hyps:
                if 8 <= len(r) < 40:
                    result.append(r)
                    finished = True
                    break
            if not finished:
                result.append(hyps[0])
        return result, dialogue_actions, attn

    def beam_decode_step(self, inst_dec_beams, len_dec_seq, da, src_seq, enc_output, inst_idx_to_position_map, n_bm):
        n_active_inst = len(inst_idx_to_position_map)
        # prepare_beam_dec_seq
        dec_seq = [b.get_current_state() for b in inst_dec_beams if not b.done]
        dec_seq = torch.stack(dec_seq).to(self.device)
        dec_seq = dec_seq.view(-1, len_dec_seq)
        # predict_word
        dec_output = self.decoder(dec_seq, src_seq, da, enc_output)
        dec_output = dec_output[:, -1, :]  # Pick the last step: (bh * bm) * d_h
        word_prob = F.log_softmax(self.tgt_word_prj(dec_output) * self.x_logit_scale, dim=1)
        word_prob = word_prob.view(n_active_inst, n_bm, -1)

        # Update the beam with predicted word prob information and collect incomplete instances
        active_inst_idx_list = []
        for inst_idx, inst_position in inst_idx_to_position_map.items():
            is_inst_complete = inst_dec_beams[inst_idx].advance(word_prob[inst_position])
            if not is_inst_complete:
                active_inst_idx_list += [inst_idx]
        return active_inst_idx_list


def main():
    import os
    import json
    from utils.tools import Tokenizer
    from utils.parsers import gen_parse_opt
    args = gen_parse_opt()
    args = vars(args)
    with open(os.path.join(args["data_dir"], "vocab.json"), 'r') as f:
        vocabulary = json.loads(f.read())
    vocab, ivocab = vocabulary['vocab'], vocabulary['rev']
    tokenizer = Tokenizer(vocab, ivocab)
    device = torch.device("cuda")
    model = HTNetwork(len(vocab), constants.act_len, device, args)
    model = model.to(device)
    with open(os.path.join(args["data_dir"], "train.json"), mode="r") as f:
        data = json.loads(f.read())
    encoder_type = args["encoder"].lower()

    if encoder_type == "trsa":
        from data_provider.mem_dataset import DataLoader, HTNDataset, collate_fn
        dataset = HTNDataset(data, tokenizer, 256)
        dataloader = DataLoader(
            dataset, batch_size=8, shuffle=False, num_workers=0, collate_fn=collate_fn
        )
    else:
        from data_provider.htn_dataset import DataLoader, HTNDataset, collate_fn
        dataset = HTNDataset(data, tokenizer, 256)
        dataloader = DataLoader(
            dataset, batch_size=8, shuffle=False, num_workers=0, collate_fn=collate_fn
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
        labels, logits = model(batch)
        y, da = y.to(device), da.to(device)
        pred = logits.contiguous().view(-1, logits.size(2))
        gold = y[:, 1:].contiguous().view(-1)
        generator_loss = F.cross_entropy(pred, gold, ignore_index=constants.PAD, reduction="mean")
        classifier_loss = loss_fc(labels.view(-1, 1), da.view(-1, 1))
        loss = generator_loss + classifier_loss
        loss.backward()
        optimizer.step()
    batch = next(dataiter)
    model.eval()
    sents, predict_das, attn = model.generate(batch, True)


if __name__ == '__main__':
    main()
