# -*- coding: utf-8 -*-
import torch.nn as nn
import torch
from models.sub_layers import get_non_pad_mask, get_attn_key_pad_mask, get_subsequent_mask, mem_attn_key_pad_mask
from models.sub_layers import collect_mems, PositionWiseFF, PositionalEmbedding
from models.sub_layers import MixMultiHeadAttention
from utils.constants import domains, functions, arguments

"""---------Encoder & Decoder Layers----------"""


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_head, dropout, drop_att, pre_layer_norm, avg, rel):
        super(EncoderLayer, self).__init__()

        self.slf_attn = MixMultiHeadAttention(n_head, d_model, d_head, dropout, drop_att, pre_layer_norm, avg, rel)
        self.pos_ffn = PositionWiseFF(d_model, d_inner, dropout, pre_layer_norm=pre_layer_norm)
        if rel:
            self.pos_emb = PositionalEmbedding(demb=d_model)
            self.r_w_bias = nn.Parameter(torch.Tensor(n_head, d_head))
            self.r_r_bias = nn.Parameter(torch.Tensor(n_head, d_head))
            nn.init.normal_(self.r_w_bias, 0.0, 0.1)
            nn.init.normal_(self.r_r_bias, 0.0, 0.1)
        self.rel = rel

    def forward(self, q, k, v, da=None, slf_attn_mask=None, non_pad_mask=None):
        if self.rel:
            k_len, q_len = k.size(1), q.size(1)
            pos_seq = torch.arange(k_len-1, -q_len, -1.0).to(q.device)
            pos_emb = self.pos_emb(pos_seq)
            r = pos_emb.unsqueeze(0).expand(q_len, -1, -1)
            r_w_bias, r_r_bias = self.r_w_bias, self.r_r_bias
        else:
            r, r_w_bias, r_r_bias = None, None, None
        output, attn = self.slf_attn(q, k, v, da, r, r_w_bias, r_r_bias, slf_attn_mask)
        output *= non_pad_mask
        output = self.pos_ffn(output)
        output *= non_pad_mask
        return output, attn


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_head, dropout, drop_att, pre_layer_norm, avg, rel):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MixMultiHeadAttention(n_head, d_model, d_head, dropout, drop_att, pre_layer_norm, avg, rel)
        self.enc_attn = MixMultiHeadAttention(n_head, d_model, d_head, dropout, drop_att, pre_layer_norm, avg, rel)
        self.pos_ffn = PositionWiseFF(d_model, d_inner, dropout)
        if rel:
            self.pos_emb = PositionalEmbedding(demb=d_model)
            self.r_w_bias = nn.Parameter(torch.Tensor(n_head, d_head))
            self.r_r_bias = nn.Parameter(torch.Tensor(n_head, d_head))
            nn.init.normal_(self.r_w_bias, 0.0, 0.1)
            nn.init.normal_(self.r_r_bias, 0.0, 0.1)
        self.rel = rel

    def forward(self, dec_inp, enc_out, da=None, non_pad_mask=None, slf_attn_mask=None, dec_enc_attn_mask=None):
        if self.rel:
            q_len, k_len = dec_inp.size(1), enc_out.size(1)
            pos_seq = torch.arange(q_len-1, -q_len, -1.0).to(dec_inp.device)
            pos_emb = self.pos_emb(pos_seq)
            r_slf_attn = pos_emb.unsqueeze(0).expand(q_len, -1, -1)
            pos_seq = torch.arange(k_len-1, -q_len, -1.0).to(dec_inp.device)
            pos_emb = self.pos_emb(pos_seq)
            r_enc_attn = pos_emb.unsqueeze(0).expand(q_len, -1, -1)
            r_w_bias, r_r_bias = self.r_w_bias, self.r_r_bias
        else:
            r_slf_attn, r_enc_attn = None, None
            r_w_bias, r_r_bias = None, None
        dec_out, _ = self.slf_attn(dec_inp, dec_inp, dec_inp, da, r_slf_attn, r_w_bias, r_r_bias, slf_attn_mask)
        dec_out *= non_pad_mask
        dec_out, _ = self.enc_attn(dec_out, enc_out, enc_out, da, r_enc_attn, r_w_bias, r_r_bias, dec_enc_attn_mask)
        dec_out *= non_pad_mask
        dec_out = self.pos_ffn(dec_out)
        dec_out *= non_pad_mask
        return dec_out


"""---------Encoder & Decoder----------"""


class TransformerEncoder(nn.Module):
    def __init__(self, word_emb, n_layer, n_head, d_model, d_head, d_inner, dropout, drop_att, pre_layer_norm, device):
        super(TransformerEncoder, self).__init__()
        self.position_emb = PositionalEmbedding(demb=d_model)
        self.word_emb = word_emb
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_head, dropout, drop_att, pre_layer_norm, avg=False, rel=False)
            for _ in range(n_layer)])
        self.device = device

    def forward(self, src_seq):
        src_seq = src_seq.to(self.device)
        non_pad_mask = get_non_pad_mask(src_seq)
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
        pos_seq = torch.arange(0.0, src_seq.size(1), 1.0).to(self.device)
        enc_inp = self.word_emb(src_seq) + self.position_emb(pos_seq, src_seq.size(0))
        attn = None
        for layer in self.layers:
            enc_inp, attn = layer(enc_inp, enc_inp, enc_inp, slf_attn_mask=slf_attn_mask, non_pad_mask=non_pad_mask)
        enc_output = enc_inp
        attn = attn.detach()
        return enc_output, attn


class RelEncoder(nn.Module):
    def __init__(self, word_emb, n_layer, n_head, d_model,
                 d_head, d_inner, dropout, drop_att, pre_layer_norm, device):
        super(RelEncoder, self).__init__()
        self.word_emb = word_emb
        self.device = device
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_head, dropout, drop_att, pre_layer_norm, avg=False, rel=True)
            for _ in range(n_layer)
        ])

    def forward(self, src_seq):
        """

        :param src_seq: B * len
        :return:
        """
        src_seq = src_seq.to(self.device)
        non_pad_mask = get_non_pad_mask(src_seq)
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
        enc_inp = self.word_emb(src_seq)
        attn = None
        for i, layer in enumerate(self.layers):
            enc_inp, attn = layer(enc_inp, enc_inp, enc_inp, slf_attn_mask=slf_attn_mask, non_pad_mask=non_pad_mask)
        enc_out = enc_inp
        attn = attn.detach()
        return enc_out, attn


class TRSAEncoder(nn.Module):
    def __init__(self, word_emb, n_layer, n_head, d_model,
                 d_head, d_inner, dropout, drop_att, pre_layer_norm, device):
        super(TRSAEncoder, self).__init__()
        self.word_emb = word_emb
        self.dropout = nn.Dropout(dropout)
        self.device = device
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_head, dropout, drop_att, pre_layer_norm, avg=False, rel=True)
            for _ in range(n_layer)
        ])
        self.pos_emb = PositionalEmbedding(d_model)

    def update_mem(self, hids, mem, hids_pad_mask, mem_pad_mask):
        """
        update memories
        :param hids: layer list, B * q_len
        :param mem: layer list, B * m_len
        :param hids_pad_mask: B * q_len
        :param mem_pad_mask: B * m_len
        :return:
        """
        new_mem = []
        if mem is None:
            new_mem_pad_mask = hids_pad_mask
            for i in range(len(hids)):
                new_mem.append(hids[i])
            return new_mem, new_mem_pad_mask
        assert len(hids) == len(mem), 'len(hids) != len(mem)'
        new_mem_pad_mask = torch.cat([mem_pad_mask, hids_pad_mask], dim=1)
        for i in range(len(hids)):
            cat = torch.cat([mem[i].detach(), hids[i]], dim=1)
            new_mem.append(cat)
        return new_mem, new_mem_pad_mask

    def _forward(self, src_seq, memories=None, mem_pad_mask=None):
        """
        forward for each segment
        :param src_seq: B * q_len
        :param memories: layer * B * m_len * hidden
        :param mem_pad_mask: B * m_len
        :return:
        """
        word_emb = self.word_emb(src_seq)
        enc_inp = self.dropout(word_emb)
        hids = []
        non_pad_mask = get_non_pad_mask(src_seq)
        slf_attn_mask, src_pad_mask = mem_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq, mem_pad_mask=mem_pad_mask)
        attn = None
        for i, layer in enumerate(self.layers):
            hids.append(enc_inp)
            if memories is not None:
                mem_i = memories[i]
                mem_inp = torch.cat([mem_i, enc_inp], dim=1)
            else:
                mem_inp = enc_inp
            enc_inp, attn = layer(enc_inp, mem_inp, mem_inp, slf_attn_mask=slf_attn_mask, non_pad_mask=non_pad_mask)
        new_mem, new_mem_mask = self.update_mem(hids, memories, src_pad_mask, mem_pad_mask)
        enc_inp = self.dropout(enc_inp)
        return enc_inp, attn, new_mem, new_mem_mask

    def forward(self, src_seqs, seg_select):
        """
        mem transformer encoder based on transformer  large
        :param src_seqs: tuple, (B *len0, B * len1, ...) len=seg_len
        :param seg_select: tuple, (B0, B1, ...) len=seg_len
        :return: B * last_len * emb
        """
        memories, mem_pad_mask = None, None
        enc_out = None
        attns = []
        for src_seq, mask in zip(src_seqs, seg_select):
            src_seq, mask = src_seq.to(self.device), mask.to(self.device)
            active, non_active = torch.nonzero(mask).squeeze(-1), torch.nonzero(1-mask).squeeze(-1)
            indices = torch.cat([active, non_active])
            _, recover_indices = torch.sort(indices, dim=0)
            src_seq = torch.index_select(src_seq, 0, active)
            if memories is not None:
                memories = [torch.index_select(mem, 0, active) for mem in memories]
                mem_pad_mask = torch.index_select(mem_pad_mask, 0, active)
            enc_out, attn, memories, mem_pad_mask = self._forward(src_seq, memories, mem_pad_mask)
            attn = attn.detach()
            memories, mem_pad_mask, attn = collect_mems(memories, mem_pad_mask, attn, recover_indices)
            attns.append(attn)
        return enc_out, attns


class TransformerDecoder(nn.Module):
    def __init__(self, word_emb, n_layer, n_head, d_model, d_head, d_inner, dropout, drop_att, pre_layer_norm, device):
        super(TransformerDecoder, self).__init__()
        self.word_emb = word_emb
        self.pos_emb = PositionalEmbedding(demb=d_model)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_head, dropout, drop_att, pre_layer_norm, avg=False, rel=False)
            for _ in range(n_layer)])
        self.device = device

    def forward(self, tgt_seq, src_seq, da, enc_out):
        tgt_seq, src_seq = tgt_seq.to(self.device), src_seq.to(self.device)
        enc_out = enc_out.to(self.device)
        non_pad_mask = get_non_pad_mask(tgt_seq)
        slf_attn_mask_subseq = get_subsequent_mask(tgt_seq)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
        dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=tgt_seq)

        pos_seq = torch.arange(0.0, tgt_seq.size(1), 1.0).to(self.device)
        dec_inp = self.word_emb(tgt_seq) + self.pos_emb(pos_seq, tgt_seq.size(0))
        for layer in self.layers:
            dec_inp = layer(dec_inp, enc_out, non_pad_mask=non_pad_mask, slf_attn_mask=slf_attn_mask,
                            dec_enc_attn_mask=dec_enc_attn_mask)
        return dec_inp


class RelDecoder(nn.Module):
    def __init__(self, word_emb, n_layer, n_head, d_model, d_head, d_inner, dropout, drop_att, pre_layer_norm, device):
        super(RelDecoder, self).__init__()
        self.word_emb = word_emb
        self.pos_emb = PositionalEmbedding(demb=d_model)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_head, dropout, drop_att, pre_layer_norm, avg=False, rel=True)
            for _ in range(n_layer)])
        self.device = device

    def forward(self, tgt_seq, src_seq, da, enc_out):
        tgt_seq, src_seq = tgt_seq.to(self.device), src_seq.to(self.device)
        enc_out = enc_out.to(self.device)
        non_pad_mask = get_non_pad_mask(tgt_seq)
        slf_attn_mask_subseq = get_subsequent_mask(tgt_seq)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
        dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=tgt_seq)

        pos_seq = torch.arange(0.0, tgt_seq.size(1), 1.0).to(self.device)
        dec_inp = self.word_emb(tgt_seq) + self.pos_emb(pos_seq, tgt_seq.size(0))
        for layer in self.layers:
            dec_inp = layer(dec_inp, enc_out, non_pad_mask=non_pad_mask, slf_attn_mask=slf_attn_mask,
                            dec_enc_attn_mask=dec_enc_attn_mask)
        return dec_inp


class HDSADecoder(nn.Module):
    def __init__(self, word_emb, n_head, d_model, d_head, d_inner, dropout, drop_att, pre_layer_norm, device):
        super(HDSADecoder, self).__init__()
        self.word_emb = word_emb
        self.pos_emb = PositionalEmbedding(demb=d_model)
        self.prior_layer = DecoderLayer(d_model, d_inner, len(domains), d_head, dropout, drop_att,
                                        pre_layer_norm, avg=True, rel=False)
        self.middle_layer = DecoderLayer(d_model, d_inner, len(functions), d_head, dropout, drop_att,
                                         pre_layer_norm, avg=True, rel=False)
        self.post_layer = DecoderLayer(d_model, d_inner, len(arguments), d_head, dropout, drop_att,
                                       pre_layer_norm, avg=True, rel=False)
        self.final_layer = DecoderLayer(d_model, d_inner, n_head, d_head, dropout, drop_att,
                                        pre_layer_norm, avg=False, rel=False)
        self.device = device

    def forward(self, tgt_seq, src_seq, da, enc_out):
        tgt_seq, src_seq = tgt_seq.to(self.device), src_seq.to(self.device)
        da, enc_out = da.to(self.device), enc_out.to(self.device)
        non_pad_mask = get_non_pad_mask(tgt_seq)
        slf_attn_mask_subseq = get_subsequent_mask(tgt_seq)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
        dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=tgt_seq)

        pos_seq = torch.arange(0.0, tgt_seq.size(1), 1.0).to(self.device)
        dec_inp = self.word_emb(tgt_seq) + self.pos_emb(pos_seq, tgt_seq.size(0))
        da_domain = da[:, :len(domains)]
        da_func = da[:, len(domains):len(domains) + len(functions)]
        da_argument = da[:, len(domains) + len(functions):]

        dec_inp = self.prior_layer(dec_inp, enc_out, da_domain,
                                   non_pad_mask=non_pad_mask, slf_attn_mask=slf_attn_mask,
                                   dec_enc_attn_mask=dec_enc_attn_mask)
        dec_inp = self.middle_layer(dec_inp, enc_out, da_func,
                                    non_pad_mask=non_pad_mask, slf_attn_mask=slf_attn_mask,
                                    dec_enc_attn_mask=dec_enc_attn_mask)
        dec_inp = self.post_layer(dec_inp, enc_out, da_argument,
                                  non_pad_mask=non_pad_mask, slf_attn_mask=slf_attn_mask,
                                  dec_enc_attn_mask=dec_enc_attn_mask)
        dec_inp = self.final_layer(dec_inp, enc_out, non_pad_mask=non_pad_mask, slf_attn_mask=slf_attn_mask,
                                   dec_enc_attn_mask=dec_enc_attn_mask)
        return dec_inp


class RelHDSADecoder(nn.Module):
    def __init__(self, word_emb, n_head, d_model, d_head, d_inner, dropout, drop_att, pre_layer_norm, device):
        super(RelHDSADecoder, self).__init__()
        self.word_emb = word_emb
        self.pos_emb = PositionalEmbedding(demb=d_model)
        self.prior_layer = DecoderLayer(d_model, d_inner, len(domains), d_head, dropout, drop_att,
                                        pre_layer_norm, avg=True, rel=True)
        self.middle_layer = DecoderLayer(d_model, d_inner, len(functions), d_head, dropout, drop_att,
                                         pre_layer_norm, avg=True, rel=True)
        self.post_layer = DecoderLayer(d_model, d_inner, len(arguments), d_head, dropout, drop_att,
                                       pre_layer_norm, avg=True, rel=True)
        self.final_layer = DecoderLayer(d_model, d_inner, n_head, d_head, dropout, drop_att,
                                        pre_layer_norm, avg=False, rel=True)
        self.device = device

    def forward(self, tgt_seq, src_seq, da, enc_out):
        tgt_seq, src_seq = tgt_seq.to(self.device), src_seq.to(self.device)
        da, enc_out = da.to(self.device), enc_out.to(self.device)
        non_pad_mask = get_non_pad_mask(tgt_seq)
        slf_attn_mask_subseq = get_subsequent_mask(tgt_seq)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
        dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=tgt_seq)

        dec_inp = self.word_emb(tgt_seq)
        da_domain = da[:, :len(domains)]
        da_func = da[:, len(domains):len(domains) + len(functions)]
        da_argument = da[:, len(domains) + len(functions):]
        dec_inp = self.prior_layer(dec_inp, enc_out, da_domain,
                                   non_pad_mask=non_pad_mask, slf_attn_mask=slf_attn_mask,
                                   dec_enc_attn_mask=dec_enc_attn_mask)
        dec_inp = self.middle_layer(dec_inp, enc_out, da_func,
                                    non_pad_mask=non_pad_mask, slf_attn_mask=slf_attn_mask,
                                    dec_enc_attn_mask=dec_enc_attn_mask)
        dec_inp = self.post_layer(dec_inp, enc_out, da_argument,
                                  non_pad_mask=non_pad_mask, slf_attn_mask=slf_attn_mask,
                                  dec_enc_attn_mask=dec_enc_attn_mask)
        dec_inp = self.final_layer(dec_inp, enc_out,
                                   non_pad_mask=non_pad_mask, slf_attn_mask=slf_attn_mask,
                                   dec_enc_attn_mask=dec_enc_attn_mask)
        return dec_inp
