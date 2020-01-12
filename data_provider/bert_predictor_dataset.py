# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset, DataLoader
from utils import constants
import os
from pytorch_transformers import BertTokenizer
from icecream import ic
import json
import regex as re


def collate_fn(data):
    """
    :param data:
    :return:
    """
    names, turns, xs, x_segments, data_sources, bses, das = zip(*data)
    max_len = max(len(x) for x in xs)
    x_batch = torch.tensor([x + [0] * (max_len - len(x)) for x in xs], dtype=torch.long)
    segment_batch = torch.tensor([x + [0] * (max_len - len(x)) for x in x_segments], dtype=torch.long)
    x_mask = torch.tensor([[1] * len(x) + [0] * (max_len - len(x)) for x in xs], dtype=torch.long)
    data_sources = torch.tensor(data_sources, dtype=torch.float)
    bses = torch.tensor(bses, dtype=torch.float)
    da_batch = torch.tensor(das, dtype=torch.float)
    return names, turns, x_batch, segment_batch, x_mask, data_sources, bses, da_batch, 0  # 最后站位


class BertPredictionDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        self.names, self.turns, self.sources = [], [], []
        self.data_sources = []
        self.histories, self.users, self.syses, self.dialogue_actions = [], [], [], []
        self.belief_states = []
        self.max_length = max_length
        rule = r"\[([a-z]+)_([a-z]+)\]"
        for dialogue in data:
            name = dialogue["name"]
            for dialogue_turn in dialogue["turns"]:
                turn = dialogue_turn["turn"]
                source = ""
                history = dialogue["turns"][turn-1]["sys_orig"] if turn > 0 else "conversation start"
                data_sources = [0] * constants.entity_len
                for k, v in dialogue_turn["source"].items():
                    match = re.search(rule, k)
                    source += "{} {} is {} ".format(match.group(1), match.group(2), v)
                    if k in constants.ontology["entities"] and "value" not in k:
                        index = constants.ontology["entities"].index(k)
                        data_sources[index] = 1
                if len(source) == 0:
                    source = "nothing"
                self.data_sources.append(data_sources)
                self.histories.append(history)
                self.names.append(name)
                self.turns.append(turn)
                self.dialogue_actions.append(json.loads(dialogue_turn["dialogue_action"]))
                self.sources.append(source)
                self.users.append(dialogue_turn["user_orig"])
                self.syses.append(dialogue_turn["sys_orig"])
                self.belief_states.append(json.loads(dialogue_turn["belief_state"]))
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name, turn, source, data_source = self.names[idx], self.turns[idx], self.sources[idx], self.data_sources[idx]
        history, user, da = self.histories[idx], self.users[idx], self.dialogue_actions[idx]
        bs = self.belief_states[idx]
        segment_user, segment_sys, segment_source = 1, 0, 0
        user_tokens = self.tokenizer.tokenize(user)
        sys_tokens = self.tokenizer.tokenize(history)
        source_tokens = self.tokenizer.tokenize(source)
        # 构建x & segment: u0, source0, s0, u1, source1, s1 ... un, source-n
        x, x_segment = [constants.CLS_WORD], [segment_user]
        _truncate_seq_pair(user_tokens, sys_tokens, max_length=self.max_length-3)
        x += sys_tokens + [constants.SEP_WORD] + user_tokens + [constants.SEP_WORD]
        x_segment += [segment_sys] * (len(sys_tokens) + 1) + [segment_user] * (len(user_tokens) + 1)

        source_len = self.max_length - len(x) - 1
        if source_len > 0:
            x += source_tokens[:source_len] + [constants.SEP_WORD]
            x_segment += [segment_source] * (len(source_tokens[:source_len])+1)
        x = self.tokenizer.convert_tokens_to_ids(x)
        return name, turn, x, x_segment, data_source, bs, da


def convert_special_token_to_normal(string):
    rule = r"\[([a-z]+)_([a-z]+)\]"
    normal_tokens = []
    tokens = string.split(" ")
    for token in tokens:
        match = re.search(rule, token)
        if match:
            normal_tokens.append(match.group(1))
            normal_tokens.append(match.group(2))
        else:
            normal_tokens.append(token)
    return " ".join(normal_tokens)


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
    dataset = BertPredictionDataset(data, tokenizer, args["max_seq_length"])
    dataloader = DataLoader(
        dataset, batch_size=args["batch_size"], shuffle=False, num_workers=4, collate_fn=collate_fn
    )
    return dataloader


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def test():
    with open("data/train.json", mode="r") as f:
        data = json.loads(f.read())
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    dataset = BertPredictionDataset(data, tokenizer, 256)
    dataloader = DataLoader(
        dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=collate_fn
    )
    num = 50
    for i, batch in enumerate(dataloader):
        if i > num:
            break
        names, turns, x, x_segment, x_mask, bs, db, da = batch
        x = x.tolist()
        ic(names[0], turns[0])
        print("x: ", tokenizer.convert_ids_to_tokens(x[0]))


if __name__ == "__main__":
    test()
