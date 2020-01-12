# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
from utils import constants
import torch.nn.functional


def collect_mems(mems, mem_pad_mask, attn, recover_indices):
    """
    collect the batch, including the active & non active part
    :param mems: [ac_B * mem_len * hidden], len=n_layer
    :param mem_pad_mask: ac_B * mem_len
    :param attn: ac_B * q_len * mem_len * n_layer
    :param recover_indices: B
    :return: [B * mem_len * hidden], len=n_layer
    """
    device, dtype = mems[0].device, mems[0].dtype
    active_bs, mem_len, hidden_size = mems[0].size()
    bs = recover_indices.size(0)
    # recover memories
    new_mems = []
    for i in range(len(mems)):
        pad_mem = torch.zeros(bs-active_bs, mem_len, hidden_size, dtype=dtype, device=device)
        new_mem = torch.cat([mems[i], pad_mem], dim=0)
        new_mem = torch.index_select(new_mem, 0, recover_indices)
        new_mems.append(new_mem)
    # recover mem mask
    pad_mem_pad_mask = torch.ones(bs-active_bs, mem_len, dtype=mem_pad_mask.dtype, device=device)
    new_mem_pad_mask = torch.cat([mem_pad_mask, pad_mem_pad_mask], dim=0)
    new_mem_pad_mask = torch.index_select(new_mem_pad_mask, 0, recover_indices)
    # recover attn
    _, q_len, _, n_head = attn.size()
    pad_attn = torch.zeros(bs - active_bs, q_len, mem_len, n_head, dtype=attn.dtype, device=device)
    new_attn = torch.cat([attn, pad_attn], dim=0)
    new_attn = torch.index_select(new_attn, 0, recover_indices)
    return new_mems, new_mem_pad_mask, new_attn


def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(constants.PAD).type(torch.float).unsqueeze(-1)


def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask


def mem_attn_key_pad_mask(seq_k, seq_q, mem_pad_mask=None):
    """
    For masking out the padding part of key sequence.
    Expand to fit the shape of key query attention matrix.
    :param seq_k: B * len_k
    :param seq_q: B * len_q
    :param mem_pad_mask: B * len_k'
    :return key_pad_mask: B * lq x (lk'+lk or lk)
    :return query_pad_mask: B * len_k
    """
    len_q = seq_q.size(1)
    k_pad_mask = seq_k.eq(constants.PAD)
    q_pad_mask = seq_q.eq(constants.PAD)
    if mem_pad_mask is not None:
        k_pad_mask = torch.cat((mem_pad_mask, k_pad_mask), dim=1)  # B * (lk'+lk)
    k_pad_mask = k_pad_mask.unsqueeze(1).expand(-1, len_q, -1)  # B * lq * (lk'+lk or lk)
    return k_pad_mask, q_pad_mask


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask


def _rel_shift(x):
    zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]), device=x.device, dtype=x.dtype)
    x_padded = torch.cat([zero_pad, x], dim=1)
    x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])
    x = x_padded[1:].view_as(x)[:, :x.size(1) - x.size(0) + 1, :]
    return x


class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[None, :, :].expand(bsz, -1, -1)
        else:
            return pos_emb


class PositionWiseFF(nn.Module):
    def __init__(self, d_model, d_inner, dropout, pre_layer_norm=False):
        super(PositionWiseFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout
        core_line1 = nn.Linear(d_model, d_inner)
        core_line2 = nn.Linear(d_inner, d_model)
        torch.nn.init.kaiming_normal_(core_line1.weight, nonlinearity='relu')
        nn.init.constant_(core_line1.bias, 0.0)
        torch.nn.init.kaiming_normal_(core_line2.weight, nonlinearity='relu')
        nn.init.constant_(core_line2.bias, 0.0)
        self.CoreNet = nn.Sequential(
            core_line1,
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            core_line2,
            nn.Dropout(dropout)
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.pre_layer_norm = pre_layer_norm

    def forward(self, inp):
        if self.pre_layer_norm:
            # layer normalization + position wise feed-forward
            core_out = self.CoreNet(self.layer_norm(inp))
            # residual connection
            output = core_out + inp
        else:
            # position wise feed-forward
            core_out = self.CoreNet(inp)
            # residual connection + layer normalization
            output = self.layer_norm(inp + core_out)

        return output


class MixMultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, drop_att, pre_layer_norm=False, avg=False, rel=False):
        super(MixMultiHeadAttention, self).__init__()

        self.is_avg = avg
        self.is_rel = rel
        self.n_head = n_head
        self.d_head = d_head

        self.w_qs = nn.Linear(d_model, n_head * d_head)
        self.w_ks = nn.Linear(d_model, n_head * d_head)
        self.w_vs = nn.Linear(d_model, n_head * d_head)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_head)))
        nn.init.constant_(self.w_qs.bias, 0.0)
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_head)))
        nn.init.constant_(self.w_ks.bias, 0.0)
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_head)))
        nn.init.constant_(self.w_vs.bias, 0.0)
        self.scale = d_head ** -0.5
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.drop_att = nn.Dropout(drop_att)
        self.pre_layer_norm = pre_layer_norm

        if self.is_avg:
            self.fc = nn.Linear(d_head, d_model)
        else:
            self.fc = nn.Linear(n_head * d_head, d_model)
        torch.nn.init.kaiming_normal_(self.fc.weight, nonlinearity='relu')
        torch.nn.init.constant_(self.fc.bias, 0.0)
        if self.is_rel:
            self.r_net = nn.Linear(d_model, self.n_head * self.d_head, bias=False)
            torch.nn.init.kaiming_normal_(self.r_net.weight, nonlinearity='relu')

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, da=None, r=None, r_w_bias=None, r_r_bias=None, mask=None):
        """
        mix original & relative & average multi-head attention
        :param q: B * q_len * d_model
        :param k: B * k_len * d_model
        :param v: B * k_len * d_model
        :param da: B * n_head
        :param r: q_len * k_len * d_model
        :param r_w_bias: n_head * d_head
        :param r_r_bias: n_head * d_head
        :param mask: q_len * k_len or B * q_len * k_len
        :return: B * q_len * d_model
        """
        d_head, n_head = self.d_head, self.n_head

        sz_b, q_len, k_len = q.size(0), q.size(1), k.size(1)
        residual = q
        if self.pre_layer_norm:
            q, k, v = self.layer_norm(q), self.layer_norm(k), self.layer_norm(v)
        q = self.w_qs(q).view(sz_b, q_len, n_head, d_head)
        k = self.w_ks(k).view(sz_b, k_len, n_head, d_head)
        v = self.w_vs(v).view(sz_b, k_len, n_head, d_head)

        if self.is_rel:
            assert (r is not None and r_w_bias is not None and r_r_bias is not None)
            # ic(q_len, k_len, r.size())
            r = _rel_shift(r)
            # ic(r.size())
            r = self.r_net(r).view(q_len, k_len, self.n_head, self.d_head)
            rw_q = q + r_w_bias  # bsz x q_len x n_head x d_head
            AC = torch.einsum('bqnd,bknd->bqkn', [rw_q, k])
            rr_q = q + r_r_bias
            BD = torch.einsum('bqnd,qknd->bqkn', [rr_q, r])
            attn_score = AC + BD
        else:
            attn_score = torch.einsum('bqnd,bknd->bqkn', [q, k])
        attn_score.mul_(self.scale)

        if mask is not None and mask.any().item():
            if mask.dim() == 2:
                mask = mask[None, :, :, None]
            elif mask.dim() == 3:
                # mask: B * q_len * k_len
                mask = mask[:, :, :, None]
            attn_score = attn_score.float().masked_fill(mask, -float('inf')).type_as(attn_score)

        # [bsz x qlen x klen x n_head]
        attn_prob = torch.nn.functional.softmax(attn_score, dim=2)
        attn_prob = self.drop_att(attn_prob)
        # compute attention vector
        attn_vec = torch.einsum('bqkn,bknd->bqnd', [attn_prob, v])
        if self.is_avg:
            assert (da is not None)
            # 实现同一层不同输出的叠加，论文中有提到，这样可以保证每一层的输出都是同一维度的。
            attn_vec = torch.einsum('bqnd,bn->bqd', [attn_vec, da])
        else:
            attn_vec = attn_vec.contiguous().view(sz_b, q_len, self.n_head * self.d_head)
        # linear projection
        output = self.fc(attn_vec)
        output = self.dropout(output)

        if self.pre_layer_norm:
            # residual connection
            output = residual + output
        else:
            # residual connection + layer normalization
            output = self.layer_norm(residual + output)
        return output, attn_prob


class CNNClassifier(nn.Module):
    """
    The CNN has one convolution layer for each ngram filter size. Each convolution operation gives
    out a vector of size num_filters. The number of times a convolution layer will be used
    is ``num_tokens - ngram_size + 1``.

    This operation is repeated for every ngram size passed, and consequently the dimensionality of
    the output after maxpooling is ``len(ngram_filter_sizes) * num_filters``.  This then gets
    (optionally) projected down to a lower dimensional output, specified by ``output_dim``.

    We then use a fully connected layer to project in back to the desired output_dim.  For more
    details, refer to "A Sensitivity Analysis of (and Practitioners’ Guide to) Convolutional Neural
    Networks for Sentence Classification", Zhang and Wallace 2016, particularly Figure 1.

    Args:
        input_dim (int): This is the input dimension to the encoder
        num_filters (int): This is the output dim for each convolution layer, which is the number
          of "filters" learned by that layer.
        ngram_filter_sizes (:class:`tuple` of :class:`int`, optional): This specifies both the
          number of convolution layers we will create and their sizes. The default of
          ``(2, 3, 4, 5)`` will have four convolution layers, corresponding to encoding ngrams of
          size 2 to 5 with some number of filters.
        output_dim (int or None, optional) : After doing convolutions and pooling, we'll project the
          collected features into a vector of this size.  If this value is ``None``, we will just
          return the result of the max pooling, giving an output of shape
          ``len(ngram_filter_sizes) * num_filters``.
    """

    def __init__(self, input_dim, num_filters, ngram_filter_sizes=(2, 3, 4, 5), output_dim=None):
        super(CNNClassifier, self).__init__()
        self._ngram_filter_sizes = ngram_filter_sizes
        self._activation = nn.ReLU()
        self._output_dim = output_dim

        self._convolution_layers = [
            nn.Conv1d(
                in_channels=input_dim,
                out_channels=num_filters,
                kernel_size=ngram_size) for ngram_size in self._ngram_filter_sizes
        ]
        for i, conv_layer in enumerate(self._convolution_layers):
            self.add_module('conv_layer_%d' % i, conv_layer)

        maxpool_output_dim = num_filters * len(self._ngram_filter_sizes)
        if self._output_dim:
            self.projection_layer = nn.Linear(maxpool_output_dim, self._output_dim)
        else:
            self.projection_layer = None
            self._output_dim = maxpool_output_dim

    def forward(self, tokens, mask=None):
        """
        Args:
            tokens: [batch_size, num_tokens, input_dim]
            mask: torch.FloatTensor, [b, n]
        Returns:
            (:class:`torch.FloatTensor` [batch_size, output_dim]): Encoding of sequence.
        """
        if mask is not None:
            tokens = tokens * mask.unsqueeze(-1).float()
        tokens = torch.transpose(tokens, 1, 2)
        filter_outputs = []
        for i in range(len(self._convolution_layers)):
            convolution_layer = getattr(self, 'conv_layer_{}'.format(i))
            filter_outputs.append(self._activation(convolution_layer(tokens)).max(dim=2)[0])

        # Now we have a list of `num_conv_layers` tensors of shape `(batch_size, num_filters)`.
        # Concatenating them gives us a tensor of shape
        # `(batch_size, num_filters * num_conv_layers)`.
        maxpool_output = torch.cat(filter_outputs, dim=1) if len(filter_outputs) > 1 else filter_outputs[0]
        if self.projection_layer:
            result = self.projection_layer(maxpool_output)
        else:
            result = maxpool_output
        return result
