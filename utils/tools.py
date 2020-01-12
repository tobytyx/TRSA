from utils import constants
import json
import regex as re
import torch


class Tokenizer(object):
    def __init__(self, vocab, ivocab, lower_case=True):
        super(Tokenizer, self).__init__()
        self.lower_case = lower_case
        self.ivocab = ivocab
        self.vocab = vocab
        self.vocab_len = len(self.vocab)

    def tokenize(self, sent):
        if self.lower_case:
            return re.split(r"\s+", sent.lower())
        else:
            return re.split(r"\s+", sent)

    def get_word_id(self, w):
        if w in self.vocab:
            return self.vocab[w]
        else:
            return self.vocab[constants.UNK_WORD]

    def get_word(self, k):
        k = str(k)
        return self.ivocab[k]
            
    def convert_tokens_to_ids(self, sent):
        return [self.get_word_id(w) for w in sent]

    def convert_id_to_tokens(self, word_ids, remain_eos=False):
        if isinstance(word_ids, list):
            if remain_eos:
                return " ".join([self.get_word(wid) for wid in word_ids if wid != constants.PAD])
            else:
                return " ".join([self.get_word(wid) for wid in word_ids if wid not in [constants.PAD, constants.EOS]])
        else:
            if remain_eos:
                return " ".join([self.get_word(wid.item()) for wid in word_ids if wid != constants.PAD])
            else:
                return " ".join(
                    [self.get_word(wid.item()) for wid in word_ids if wid not in [constants.PAD, constants.EOS]])

    def convert_template(self, template_ids):
        return [self.get_word(wid) for wid in template_ids if wid != constants.PAD]


# d_p: pred, d_r: val.json


def dedelex(delex_preds, mode="test"):
    """
    dedelex delex preds.
    :param delex_preds: {"name": [sys[0], sys[1], ...]}
    :param mode: which dataset should be used
    :return:
    """
    need_replace = 0
    success = 0

    with open("data/{}.json".format(mode)) as f:
        gt_dialogues = json.load(f)

    for dial in gt_dialogues:
        name = dial['name']
        turns = dial['turns']
        for turn_id in range(len(delex_preds[name])):
            kb = turns[turn_id]['source']
            act = turns[turn_id]['act']
            words = delex_preds[name][turn_id].split(' ')
            for i in range(len(words)):    
                if "[" in words[i] and "]" in words[i]:
                    need_replace += 1.
                    if words[i] in kb:
                        words[i] = kb[words[i]]
                        success += 1.
                    elif "taxi" in words[i]:
                        if words[i] == "[taxi_type]" and "domain-taxi-inform-car" in act:
                            words[i] = act["domain-taxi-inform-car"]
                            success += 1.
                        elif words[i] == "[taxi_phone]" and "domain-taxi-inform-phone" in act:
                            words[i] = act["domain-taxi-inform-phone"]
                            success += 1.
            delex_preds[name][turn_id] = " ".join(words)
    success_rate = success / need_replace
    return success_rate


def get_attn_graph(seqs, attns, encoder_type):
    if encoder_type == "trsa":
        assert len(seqs) == len(attns)
        attn_graphs = []
        max_len = max([attn.size(2) for attn in attns])
        for attn in attns:
            bsz, q, k, n_head = attn.size()
            attn_pad = torch.zeros(bsz, q, max_len-k, n_head, dtype=attn.dtype, device=attn.device)
            full_attn = torch.cat([attn, attn_pad], dim=2)
            attn_graphs.append(full_attn)
        attn_graphs = torch.cat(attn_graphs, dim=1)
        seqs = torch.cat(seqs, dim=1)
    else:
        attn_graphs = attns
    attn_graphs = attn_graphs.cpu().detach().numpy()
    seqs = seqs.cpu().detach().tolist()
    return seqs, attn_graphs

