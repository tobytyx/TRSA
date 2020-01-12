# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.Beam import Beam, collate_active_info, collect_hypothesis_and_scores, get_inst_idx_to_tensor_position_map
from utils import constants as Constants
import math
import numpy as np


def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask


def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEmbedding, self).__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class AverageHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super(AverageHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, a, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()
        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        a = a.permute(1, 0).contiguous()[:, :, None, None]

        # output = output * a
        output = torch.sum(output * a, 0)
        output = output.view(sz_b, len_q, -1)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)  # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask

        return enc_output, enc_slf_attn


class AvgDecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, n_head_enc, dropout=0.1):
        super(AvgDecoderLayer, self).__init__()
        self.slf_attn = AverageHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head_enc, d_model, d_model // n_head_enc, d_model // n_head_enc, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, act_vecs, dec_input, enc_output, non_pad_mask=None, slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(act_vecs, dec_input, dec_input, dec_input, mask=slf_attn_mask)
        dec_output *= non_pad_mask
        dec_output, dec_enc_attn = self.enc_attn(dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output *= non_pad_mask

        dec_output = self.pos_ffn(dec_output)
        dec_output *= non_pad_mask

        return dec_output, dec_slf_attn, None


class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, dec_input, enc_output, non_pad_mask=None, slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask)
        dec_output *= non_pad_mask

        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output *= non_pad_mask

        dec_output = self.pos_ffn(dec_output)
        dec_output *= non_pad_mask

        return dec_output, dec_slf_attn, dec_enc_attn


class HDSATransformer(nn.Module):
    def __init__(self, vocab_size, d_word_vec, n_layers, d_model, n_head, dropout=0.1):
        super(HDSATransformer, self).__init__()
        self.take_domain = True

        self.tgt_word_emb = nn.Embedding(vocab_size, d_word_vec, padding_idx=Constants.PAD)
        self.post_word_emb = PositionalEmbedding(d_model=d_word_vec)

        d_inner = d_model * 4
        d_k, d_v = d_model // n_head, d_model // n_head

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

        d_inner = d_model * 4
        d_k, d_v = d_model // n_head, d_model // n_head
        if self.take_domain:
            self.prior_layer_stack = AvgDecoderLayer(d_model, d_inner, len(Constants.domains),
                                                     d_k, d_v, n_head_enc=n_head, dropout=dropout)
            self.middle_layer_stack = AvgDecoderLayer(d_model, d_inner, len(Constants.functions),
                                                      d_k, d_v, n_head_enc=n_head, dropout=dropout)
            self.post_layer_stack = AvgDecoderLayer(d_model, d_inner, len(Constants.arguments),
                                                    d_k, d_v, n_head_enc=n_head, dropout=dropout)
            self.final_layer_stack = DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
        else:
            self.prior_layer_stack = AvgDecoderLayer(d_model, d_inner, len(Constants.functions),
                                                     d_k, d_v, n_head_enc=n_head, dropout=dropout)
            self.middle_layer_stack = AvgDecoderLayer(d_model, d_inner, len(Constants.arguments),
                                                      d_k, d_v, n_head_enc=n_head, dropout=dropout)
            self.post_layer_stack = DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)

        self.tgt_word_prj = nn.Linear(d_model, vocab_size, bias=False)
        self.softmax = nn.Softmax(-1)

    def encoder(self, src_seq):
        non_pad_mask = get_non_pad_mask(src_seq)
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
        enc_inp = self.tgt_word_emb(src_seq) + self.post_word_emb(src_seq)

        for layer in self.layer_stack:
            enc_inp, _ = layer(enc_inp, non_pad_mask, slf_attn_mask)
        enc_output = enc_inp
        return enc_output

    def decoder(self, tgt_seq, src_seq, act_vecs, enc_output):
        non_pad_mask = get_non_pad_mask(tgt_seq)
        slf_attn_mask_subseq = get_subsequent_mask(tgt_seq)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
        dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=tgt_seq)

        # -- Forward
        dec_inp = self.tgt_word_emb(tgt_seq) + self.post_word_emb(tgt_seq)
        domain_vecs = act_vecs[:, :len(Constants.domains)]
        function_vecs = act_vecs[:, len(Constants.domains):len(Constants.domains) + len(Constants.functions)]
        argument_vecs = act_vecs[:, len(Constants.domains) + len(Constants.functions):]
        if self.take_domain:
            dec_inp, _, _ = self.prior_layer_stack(
                domain_vecs, dec_inp, enc_output, non_pad_mask, slf_attn_mask, dec_enc_attn_mask)
            dec_inp, _, _ = self.middle_layer_stack(
                function_vecs, dec_inp, enc_output, non_pad_mask, slf_attn_mask, dec_enc_attn_mask)
            dec_inp, _, _ = self.post_layer_stack(
                argument_vecs, dec_inp, enc_output, non_pad_mask, slf_attn_mask, dec_enc_attn_mask)
            dec_inp, _, _ = self.final_layer_stack(dec_inp, enc_output, non_pad_mask, slf_attn_mask, dec_enc_attn_mask)
        else:
            dec_inp, _, _ = self.prior_layer_stack(
                function_vecs, dec_inp, enc_output, non_pad_mask, slf_attn_mask, dec_enc_attn_mask)
            dec_inp, _, _ = self.middle_layer_stack(
                argument_vecs, dec_inp, enc_output, non_pad_mask, slf_attn_mask, dec_enc_attn_mask)
            dec_inp, _, _ = self.post_layer_stack(dec_inp, enc_output, non_pad_mask, slf_attn_mask, dec_enc_attn_mask)
        return dec_inp

    def forward(self, tgt_seq, src_seq, act_vecs):
        tgt_seq = tgt_seq[:, :-1]
        enc_output = self.encoder(src_seq)
        dec_output = self.decoder(tgt_seq, src_seq, act_vecs, enc_output)
        logits = self.tgt_word_prj(dec_output)
        return logits

    def generate(self, act_vecs, src_seq, n_bm, max_token_seq_len, device):
        with torch.no_grad():
            # -- Encode
            src_enc = self.encoder(src_seq)

            # -- Repeat data for beam search
            n_inst, len_s, d_h = src_enc.size()
            src_seq = src_seq.repeat(1, n_bm).view(n_inst * n_bm, len_s)
            src_enc = src_enc.repeat(1, n_bm, 1).view(n_inst * n_bm, len_s, d_h)
            act_vecs = act_vecs.repeat(1, n_bm).view(n_inst * n_bm, -1)

            # -- Prepare beams
            inst_dec_beams = [Beam(n_bm, device=device) for _ in range(n_inst)]

            # -- Bookkeeping for active or not
            active_inst_idx_list = list(range(n_inst))
            inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)
            # inst_idx_to_position_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9}

            # -- Decode
            for len_dec_seq in range(1, max_token_seq_len + 1):
                active_inst_idx_list = self.beam_decode_step(
                    inst_dec_beams, len_dec_seq, act_vecs, src_seq, src_enc, inst_idx_to_position_map, n_bm, device)
                if not active_inst_idx_list:
                    break  # all instances have finished their path to <EOS>
                act_vecs, src_seq, src_enc, inst_idx_to_position_map = collate_active_info(
                    act_vecs, src_seq, src_enc, inst_idx_to_position_map=inst_idx_to_position_map,
                    active_inst_idx_list=active_inst_idx_list,
                    n_bm=n_bm, device=device
                )

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
        return result

    def beam_decode_step(self, inst_dec_beams, len_dec_seq, act_vecs, src_seq, src_enc,
                         inst_idx_to_position_map, n_bm, device):
        n_active_inst = len(inst_idx_to_position_map)
        # prepare_beam_dec_seq
        dec_seq = [b.get_current_state() for b in inst_dec_beams if not b.done]
        dec_seq = torch.stack(dec_seq).to(device)
        dec_seq = dec_seq.view(-1, len_dec_seq)
        # predict_word
        dec_output = self.decoder(dec_seq, src_seq, act_vecs, src_enc)
        dec_output = dec_output[:, -1, :]  # Pick the last step: (bh * bm) * d_h
        word_prob = F.log_softmax(self.tgt_word_prj(dec_output), dim=1)
        word_prob = word_prob.view(n_active_inst, n_bm, -1)

        # Update the beam with predicted word prob information and collect incomplete instances
        active_inst_idx_list = []
        for inst_idx, inst_position in inst_idx_to_position_map.items():
            is_inst_complete = inst_dec_beams[inst_idx].advance(word_prob[inst_position])
            if not is_inst_complete:
                active_inst_idx_list += [inst_idx]
        return active_inst_idx_list
