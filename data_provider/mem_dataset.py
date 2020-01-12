# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset, DataLoader
from utils import constants
from utils.tools import Tokenizer
import json
import os


def collate_fn(data):
    """
    :param data:
    :return:
    """
    names, turns, xs, x_lenses, data_sources, bs, das, ys = zip(*data)
    bsz = len(xs)
    max_his_len = max(len(x) for x in xs)
    src_seqs = []
    seg_selects = []
    for seg in range(max_his_len):
        max_len = max([x_lens[-(seg+1)] for x_lens in x_lenses if len(x_lens) > seg])
        src_seq = []
        seg_select = []
        for b in range(bsz):
            if len(xs[b]) > seg:
                seq_len = x_lenses[b][-(seg+1)]
                src_seq.append(
                    [constants.PAD] * (max_len - seq_len) + xs[b][-(seg + 1)]
                )
                seg_select.append(1)
            else:
                src_seq.append([constants.PAD] * max_len)
                seg_select.append(0)
        src_seq = torch.tensor(src_seq, dtype=torch.long)
        seg_select = torch.tensor(seg_select, dtype=torch.long)
        src_seqs.append(src_seq)
        seg_selects.append(seg_select)
    src_seqs.reverse()
    seg_selects.reverse()
    das = torch.tensor(das, dtype=torch.float)
    max_len = max([len(y) for y in ys])
    ys = torch.tensor([y + [constants.PAD] * (max_len - len(y)) for y in ys], dtype=torch.long)
    data_sources = torch.tensor(data_sources, dtype=torch.float)
    bs = torch.tensor(bs, dtype=torch.float)
    return names, turns, src_seqs, seg_selects, data_sources, bs, das, ys


class HTNDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=50, da_source="", use_da=False):
        self.names, self.turns, self.belief_states = [], [], []
        self.data_sources, self.sources = [], []
        self.histories, self.users, self.syses, self.dialogue_actions = [], [], [], []
        self.max_length = max_length
        self.use_ds = use_da
        if da_source:
            with open(da_source) as f:
                pred_da = json.load(f)

        mix = []
        for dialogue in data:
            name = dialogue["name"]
            for dialogue_turn in dialogue["turns"]:
                turn = dialogue_turn["turn"]
                history = []
                for i in range(turn):
                    history.append(dialogue["turns"][i]["user"])
                    history.append(dialogue["turns"][i]["sys"])
                da = pred_da[name][str(turn)] if da_source else json.loads(dialogue_turn["dialogue_action"])
                data_source = [0] * constants.entity_len
                source = []
                for k in dialogue_turn["source"].keys():
                    if k in constants.ontology["entities"]:
                        index = constants.ontology["entities"].index(k)
                        data_source[index] = 1
                        source.append(k)
                source = "nothing" if len(source) == 0 else " , ".join(source)
                bs = json.loads(dialogue_turn["belief_state"])
                mix.append(
                    (name, turn, history, da, dialogue_turn["user"], dialogue_turn["sys"], data_source, bs, source)
                )
        mix.sort(key=lambda x: x[1])
        for name, turn, history, action, user, sys, data_source, bs, source in mix:
            self.names.append(name)
            self.turns.append(turn)
            self.histories.append(history)
            self.dialogue_actions.append(action)
            self.users.append(user)
            self.syses.append(sys)
            self.data_sources.append(data_source)
            self.belief_states.append(bs)
            self.sources.append(source)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name, turn, bs = self.names[idx], self.turns[idx], self.belief_states[idx]
        source, data_source = self.sources[idx], self.data_sources[idx]
        history, user, sys, da = self.histories[idx], self.users[idx], self.syses[idx], self.dialogue_actions[idx]
        x = []
        x_lens = []
        for i in range(turn):
            if i == 0:
                tokens = self.tokenizer.tokenize(history[i*2]) + [constants.EOS_WORD]
                tokens = self.tokenizer.convert_tokens_to_ids(tokens[-self.max_length:])
                x.append(tokens)
                x_lens.append(len(tokens))
            else:
                u_tokens = self.tokenizer.tokenize(history[i*2-1]) + [constants.EOS_WORD]
                s_tokens = self.tokenizer.tokenize(history[i*2]) + [constants.EOS_WORD]
                tokens = u_tokens + s_tokens
                tokens = self.tokenizer.convert_tokens_to_ids(tokens[-self.max_length:])
                x.append(tokens)
                x_lens.append(len(tokens))
        user_tokens = self.tokenizer.tokenize(user) + [constants.EOS_WORD]
        if self.use_ds:
            source_tokens = self.tokenizer.tokenize(source)
            user_tokens += source_tokens + [constants.EOS_WORD]
        user_tokens = self.tokenizer.convert_tokens_to_ids(user_tokens[-self.max_length:])
        x.append(user_tokens)
        x_lens.append(len(user_tokens))
        # 构建y
        sys_tokens = self.tokenizer.tokenize(sys)
        sys_tokens = [constants.SOS_WORD] + sys_tokens[:constants.RESP_MAX_LEN - 2] + [constants.EOS_WORD]
        y = self.tokenizer.convert_tokens_to_ids(sys_tokens)
        return name, turn, x, x_lens, data_source, bs, da, y


def prepare_dataloader(args, tokenizer, mode="train"):
    data = []
    if mode == "full_train":
        with open(os.path.join(args["data_dir"], "train.json"), mode="r") as f:
            data.extend(json.load(f))
        with open(os.path.join(args["data_dir"], "val.json"), mode="r") as f:
            data.extend(json.load(f))
    else:
        with open(os.path.join(args["data_dir"], mode + ".json"), mode="r") as f:
            data.extend(json.load(f))
    da_source = ""
    if args["da"] == "bert" and "train" not in mode:
        da_source = os.path.join(args["data_dir"], "bert_{}_prediction.json".format(mode))
    elif args["da"] == "trsa" and "train" not in mode:
        da_source = os.path.join(args["data_dir"], "bert_{}_prediction.json".format(mode))
    dataset = HTNDataset(data, tokenizer, args["max_seq_length"], da_source, args["data_source"])
    dataloader = DataLoader(
        dataset, batch_size=args["batch_size"], shuffle=False, num_workers=4, collate_fn=collate_fn
    )
    return dataloader


def test():
    with open("data/vocab.json", mode='r') as f:
        vocabulary = json.loads(f.read())
    with open("data/train.json", mode="r") as f:
        data = json.loads(f.read())
    vocab, ivocab = vocabulary['vocab'], vocabulary['rev']
    tokenizer = Tokenizer(vocab, ivocab)
    dataset = HTNDataset(
        data=data, tokenizer=tokenizer, max_length=32
    )
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0, collate_fn=collate_fn)
    seg_lens = {}
    for i, batch in enumerate(dataloader):
        seg_len = len(batch[2])
        if seg_len not in seg_lens:
            seg_lens[seg_len] = 0
        seg_lens[seg_len] += 1
    print(seg_lens)

if __name__ == "__main__":
    test()
