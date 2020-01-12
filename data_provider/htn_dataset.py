# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset, DataLoader
from utils import constants
from utils.tools import Tokenizer
from icecream import ic
import json
import os


def collate_fn(data):
    """
    :param data:
    :return:
    """
    names, turns, xs, x_segments, data_sources, bs, das, ys = zip(*data)
    # generator
    max_len = max(len(x) for x in xs)
    x_batch = torch.tensor([x + [constants.PAD] * (max_len - len(x)) for x in xs], dtype=torch.long)
    segment_batch = torch.tensor([x + [0] * (max_len - len(x)) for x in x_segments], dtype=torch.long)
    max_len = max(len(y) for y in ys)
    da_batch = torch.tensor(das, dtype=torch.float)
    y_batch = torch.tensor([y + [constants.PAD] * (max_len - len(y)) for y in ys], dtype=torch.long)
    data_sources = torch.tensor(data_sources, dtype=torch.float)
    bs = torch.tensor(bs, dtype=torch.float)
    return names, turns, x_batch, segment_batch, data_sources, bs, da_batch, y_batch


class HTNDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128, da_source="", use_ds=False):
        self.sources = []
        self.names, self.turns, self.data_sources, self.belief_states = [], [], [], []
        self.histories, self.users, self.syses, self.dialogue_actions = [], [], [], []
        self.max_length = max_length
        self.use_ds = use_ds
        if da_source:
            with open(da_source) as f:
                pred_da = json.load(f)

        for dialogue in data:
            name = dialogue["name"]
            for dialogue_turn in dialogue["turns"]:
                turn = dialogue_turn["turn"]
                history = []
                for i in range(turn):
                    history.append(dialogue["turns"][i]["user"])
                    history.append(dialogue["turns"][i]["sys"])
                data_source = [0] * constants.entity_len
                source = []
                for k in dialogue_turn["source"].keys():
                    if k in constants.ontology["entities"]:
                        index = constants.ontology["entities"].index(k)
                        data_source[index] = 1
                        source.append(k)
                if len(source) == 0:
                    self.sources.append("nothing")
                else:
                    self.sources.append("have " + " , ".join(source))
                self.histories.append(history)
                self.names.append(name)
                self.turns.append(turn)
                da = pred_da[name][str(turn)] if da_source else json.loads(dialogue_turn["dialogue_action"])
                self.dialogue_actions.append(da)
                self.data_sources.append(data_source)
                self.users.append(dialogue_turn["user"])
                self.syses.append(dialogue_turn["sys"])
                self.belief_states.append(json.loads(dialogue_turn["belief_state"]))
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name, turn, data_source, bs = self.names[idx], self.turns[idx], self.data_sources[idx], self.belief_states[idx]
        source = self.sources[idx]
        history, user, sys, da = self.histories[idx], self.users[idx], self.syses[idx], self.dialogue_actions[idx]
        segment_sys, segment_user = 2, 1
        sys_tokens, user_tokens = self.tokenizer.tokenize(sys), self.tokenizer.tokenize(user)
        x, x_segment = [constants.SOS_WORD], [segment_user]
        for i in range(len(history)):
            segment = segment_sys if i % 2 == 1 else segment_user
            history_tokens = self.tokenizer.tokenize(history[i])
            x = x + history_tokens + [constants.EOS_WORD]
            x_segment = x_segment + [segment] * (len(history_tokens) + 1)
        x = x + user_tokens + [constants.EOS_WORD]
        x_segment = x_segment + [segment_user] * (len(user_tokens) + 1)
        if self.use_ds:
            source_tokens = self.tokenizer.tokenize(source)
            x = x + source_tokens + [constants.EOS_WORD]
            x_segment = x_segment + [segment_sys] * (len(source_tokens) + 1)
        x_len = len(x)
        if x_len > self.max_length:
            x, x_segment = x[x_len - self.max_length:], x_segment[x_len - self.max_length:]
            x[0] = constants.SOS_WORD
        x = self.tokenizer.convert_tokens_to_ids(x)
        sys_tokens = [constants.SOS_WORD] + sys_tokens[:constants.RESP_MAX_LEN - 2] + [constants.EOS_WORD]
        y = self.tokenizer.convert_tokens_to_ids(sys_tokens)
        assert len(x) == len(x_segment), "{}, {}".format(len(x), len(x_segment))
        return name, turn, x, x_segment, data_source, bs, da, y


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
        data=data, tokenizer=tokenizer, max_length=256
    )
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=collate_fn)
    num = 0
    for i, batch in enumerate(dataloader):
        if i > num:
            break
        names, turns, x, *_,  data_source, _, da, y = batch
        ic(names[0], turns[0])
        print("x: ", tokenizer.convert_id_to_tokens(x[0], remain_eos=True))
        print("data_source: ", data_source.tolist()[0])
        print("y: ", tokenizer.convert_id_to_tokens(y[0], remain_eos=True))
        print("da: ", da[0])


if __name__ == "__main__":
    test()
