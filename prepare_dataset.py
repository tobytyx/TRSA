# -*- coding:utf-8 -*-
import os
import json
import regex as re
from copy import copy
from utils import constants


data_vocab = {}
system_vocab = {
    "[PAD]": 0,
    "[EOS]": 1,
    "[SOS]": 2,
    "[UNK]": 3,
    "[CLS]": 4,
    "[SEP]": 5
}


def update_vocab(sent):
    global data_vocab
    tokens = sent.split(" ")
    for token in tokens:
        if token not in data_vocab:
            data_vocab[token] = 0
        data_vocab[token] += 1


def correct_dataset(dialogue_turn):
    """

    :param dialogue_turn:
    :return:
    """
    correct_match = {
        # "restaurant-inform-choice": "[value_count]",
        "hotel-inform-stars": "[hotel_type]",
        # "hotel-inform-choice": "[value_count]",
        "attraction-inform-price": "[attraction_pricerange]",
        # "attraction-inform-choice": "[value_count]",
        "attraction-inform-type": "[attraction_type]",
        # "train-inform-choice": "[value_count]",
        # "train-inform-time": "[value_count]",
        "taxi-inform-departure": "[taxi_departure]"
    }
    for act, match_value in correct_match.items():
        if act in dialogue_turn["act"]:
            value = dialogue_turn["act"][act]
            if value != "?":
                re.sub(value, match_value, dialogue_turn["sys"])
    if "train-inform-time" in dialogue_turn["act"]:
        re.sub("", " minutes", dialogue_turn["sys"])
    source = dialogue_turn["source"]
    reference_count = 0
    for k in source:
        if "reference" in k:
            reference_count += 1
    if reference_count > 1:
        for k in source:
            if k in dialogue_turn["sys"]:
                dialogue_turn["source"] = {k: source[k]}
                break


def main(options="all", data_dir="./data/raw_data", output="./data"):
    with open(os.path.join(data_dir, "db_pointer.json")) as f:
        db_pointers = json.loads(f.read())
    if options == "all":
        data_files = ["train", "val", "test"]
    else:
        data_files = [options]
    for data_file in data_files:
        with open(os.path.join(data_dir, "raw_"+data_file+".json")) as f:
            data = json.loads(f.read())
        new_data = []
        for dialogue in data:
            turn = 0
            name = dialogue["file"]
            dialogue_data = {"name": name, "turns": []}
            for each_round in dialogue["info"]:
                # 已包含sys, sys_orig, user, user_orig, act, source, BS, KB
                dialogue_turn = copy(each_round)
                dialogue_turn["turn"] = turn
                bs = [0] * len(constants.belief_state)
                if each_round['BS'] != "None":
                    for domain in each_round['BS']:
                        for key, value in each_round['BS'][domain]:
                            bs[constants.belief_state.index(domain + '-' + key)] = 1
                dialogue_turn["belief_state"] = bs
                if name not in db_pointers:
                    dialogue_turn["db_pointer"] = [0] * 30
                else:
                    dialogue_turn["db_pointer"] = db_pointers[name][turn]
                act_vecs = [0] * constants.act_len
                if each_round['act'] != "None":
                    for w in each_round['act']:
                        d, f, s = w.split('-')
                        act_vecs[constants.domains.index(d)] = 1
                        act_vecs[len(constants.domains) + constants.functions.index(f)] = 1
                        act_vecs[len(constants.domains) + len(constants.functions) + constants.arguments.index(s)] = 1
                dialogue_turn["dialogue_action"] = act_vecs
                correct_dataset(dialogue_turn)

                update_vocab(dialogue_turn["sys"])
                update_vocab(dialogue_turn["user"])
                update_vocab(" ".join(dialogue_turn["source"]))

                turn += 1
                dialogue_data["turns"].append(dialogue_turn)
            new_data.append(dialogue_data)
        with open(os.path.join(output, data_file+".json"), mode="w") as f:
            f.write(json.dumps(new_data, ensure_ascii=False))
    with open(os.path.join(output, "vocab.json"), mode="w") as f:
        index = 6
        global data_vocab, system_vocab
        vocab = {k: v for k, v in system_vocab.items()}
        for k, v in data_vocab.items():
            if v > 2:
                vocab[k] = index
                index += 1
        ivocab = {str(v): k for k, v in vocab.items()}
        f.write(json.dumps({"vocab": vocab, "rev": ivocab}, indent=2))


if __name__ == "__main__":
    main()
