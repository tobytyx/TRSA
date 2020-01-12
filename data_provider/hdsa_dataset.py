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
    names, turns, xs, x_segments, das, ys = zip(*data)
    # generator
    max_len = max(len(x) for x in xs)
    x_batch = torch.tensor([x + [constants.PAD] * (max_len - len(x)) for x in xs], dtype=torch.long)
    max_len = max(len(y) for y in ys)
    da_batch = torch.tensor(das, dtype=torch.float)
    y_batch = torch.tensor([y + [constants.PAD] * (max_len - len(y)) for y in ys], dtype=torch.long)
    return names, turns, x_batch, da_batch, y_batch


class HDSADataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128, da_source=""):
        self.names, self.turns, self.sources = [], [], []
        self.histories, self.users, self.syses, self.dialogue_actions = [], [], [], []
        self.max_length = max_length
        if da_source:
            with open(da_source) as f:
                pred_da = json.load(f)

        for dialogue in data:
            name = dialogue["name"]
            for dialogue_turn in dialogue["turns"]:
                turn = dialogue_turn["turn"]
                source = []
                history = []
                for i in range(turn):
                    history.append(dialogue["turns"][i]["user"])
                    history.append(dialogue["turns"][i]["sys"])
                for k in dialogue_turn["source"].keys():
                    source.append(k)
                if len(source) == 0:
                    source = ["no information"]
                self.histories.append(history)
                self.names.append(name)
                self.turns.append(turn)
                da = pred_da[name][str(turn)] if da_source else json.loads(dialogue_turn["dialogue_action"])
                self.dialogue_actions.append(da)
                self.sources.append(source)
                self.users.append(dialogue_turn["user"])
                self.syses.append(dialogue_turn["sys"])
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name, turn, source = self.names[idx], self.turns[idx], self.sources[idx]
        history, user, sys, da = self.histories[idx], self.users[idx], self.syses[idx], self.dialogue_actions[idx]
        segment_sys, segment_user = 2, 1
        sys_tokens, user_tokens = self.tokenizer.tokenize(sys), self.tokenizer.tokenize(user)

        # source_tokens = self.tokenizer.tokenize(" ".join(source))
        x, x_segment = [constants.SOS_WORD], [segment_user]
        for i in range(len(history)):
            segment = segment_sys if i % 2 == 1 else segment_user
            history_tokens = self.tokenizer.tokenize(history[i])
            x = x + history_tokens + [constants.EOS_WORD]
            x_segment = x_segment + [segment] * (len(history_tokens) + 1)
        x = x + user_tokens
        x_segment = x_segment + [segment_user]
        x_len = len(x)
        if x_len > self.max_length:
            x, x_segment = x[x_len - self.max_length:], x_segment[x_len - self.max_length:]
            x[0] = constants.SOS_WORD
        x = self.tokenizer.convert_tokens_to_ids(x)
        sys_tokens = [constants.SOS_WORD] + sys_tokens[:constants.RESP_MAX_LEN - 2] + [constants.EOS_WORD]
        y = self.tokenizer.convert_tokens_to_ids(sys_tokens)
        return name, turn, x, x_segment, da, y


def prepare_dataloader(args, tokenizer, mode="train"):
    with open(os.path.join(args["data_dir"], mode + ".json"), mode="r") as f:
        data = json.loads(f.read())
    da_source = ""
    if mode != "train":
        da_source = os.path.join(args["data_dir"], "bert_{}_prediction.json".format(mode))
    dataset = HDSADataset(data, tokenizer, args["max_seq_length"], da_source)
    dataloader = DataLoader(
        dataset, batch_size=args["batch_size"], shuffle=False, num_workers=4, collate_fn=collate_fn
    )
    return dataloader
