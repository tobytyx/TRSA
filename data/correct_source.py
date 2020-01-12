# -*- coding: utf-8 -*-
import json
import regex as re
with open("train.json") as f:
    train = json.load(f)
with open("val.json") as f:
    val = json.load(f)
with open("test.json") as f:
    test = json.load(f)
with open("ontology.json") as f:
    ontology = json.load(f)

rule = r"\[[a-z]*_[a-z]*\]"
values = ["[value_count]", "[value_day]", "[value_place]", "[value_time]"]
ref_rule = r"\[.*_reference\]"
source_template = {
    "restaurant": {
        "[restaurant_postcode]": "xxx",
        "[restaurant_phone]": "xxx",
        "[restaurant_food]": "xxx",
        "[restaurant_pricerange]": "xxx",
        "[restaurant_address]": "xxx",
        "[restaurant_area]": "xxx",
        "[restaurant_name]": "xxx"
    },
    "hotel": {
      "[hotel_phone]": "xxx",
      "[hotel_name]": "xxx",
      "[hotel_postcode]": "xxx",
      "[hotel_area]": "xxx",
      "[hotel_pricerange]": "xxx",
      "[hotel_address]": "xxx"
    },
    "attraction": {
      "[attraction_phone]": "xxx",
      "[attraction_name]": "xxx",
      "[attraction_pricerange]": "xxx",
      "[attraction_postcode]": "xxx",
      "[attraction_area]": "xxx",
      "[attraction_address]": "xxx"
    },
    "train": {
        "[train_destination]": "xxx",
        "[train_duration]": "xxx",
        "[train_leaveat]": "xxx",
        "[train_trainid]": "xxx",
        "[train_departure]": "xxx",
        "[train_price]": "xxx",
        "[train_day]": "xxx",
        "[train_arriveby]": "xxx"
    },
    "taxi": {
        "[taxi_phone]": "xxx",
        "[taxi_type]": "xxx"
    },
    "hospital": {
        "[hospital_address]": "xxx",
        "[hospital_name]": "xxx",
        "[hospital_phone]": "xxx",
        "[hospital_postcode]": "xxx",
        "[hospital_department]": "xxx"
    },
    "police": {
        "[police_name]": "xxx",
        "[police_address]": "xxx",
        "[police_phone]": "xxx",
        "[police_postcode]": "xxx"
    }
}


for dataset in [train, val, test]:
    for dial in dataset:
        for turn in dial["turns"]:
            domains = set()
            res = re.findall(rule, turn["sys"])
            refs = []
            value_keys = set()
            for r in res:
                domain, argument = r[1:-1].split("_")
                # 加上对应的value
                if domain == "value":
                    value_keys.add(r)
                else:
                    if argument == "reference":
                        refs.append(r)
                    else:
                        domains.add(domain)
            # 去掉source有但是sys中没有的ref
            for key in list(turn["source"].keys()):
                if len(re.findall(ref_rule, key)) > 0:
                    if key not in refs:
                        turn["source"].pop(key)
                else:
                    domain, argument = key[1:-1].split("_")
                    # 去掉乱写的source
                    if domain != "value" and key not in source_template[domain]:
                        turn["source"].pop(key)
            # 加上在sys中出现了但是在source中没有出现的ref
            for ref in refs:
                if ref not in turn["source"]:
                    turn["source"][ref] = "xxx"
            source_domains = set()
            for key in turn["source"]:
                domain, argument = key[1:-1].split("_")
                if argument != "reference":
                    source_domains.add(domain)
            if source_domains != domains:
                unness = source_domains - domains
                lack = domains - source_domains
                # 多余 = sys中没有但source中有的domain
                for d in unness:
                    for key in list(turn["source"].keys()):
                        if d in key:
                            turn["source"].pop(key)
                # 缺少 = sys中有但source中没有的domain
                for d in lack:
                    for k, v in source_template[d].items():
                        turn["source"][k] = v
            # train_pricerange -> train_price
            if "[train_pricerange]" in turn["source"]:
                turn["source"]["[train_price]"] = turn["source"].pop("[train_pricerange]")
            # train_id -> train_trainid
            if "[train_id]" in turn["source"]:
                if "[train_trainid]" in turn["source"]:
                    turn["source"].pop("[train_id]")
                else:
                    turn["source"]["[train_trainid]"] = turn["source"].pop("[train_id]")
    for dial in dataset:
        for turn in dial["turns"]:
            entities = re.findall(rule, turn["sys"])
            for entity in entities:
                if entity not in turn["source"] and entity not in values:
                    turn["source"][entity] = "xxx"
            for k in list(turn["source"].keys()):
                domain = k[1:-1].split("_")[0]
                if "reference" in k:
                    if domain == "train":
                        if "[train_trainid]" not in turn["source"]:
                            turn["source"]["[train_trainid]"] = "xxx"
                    elif domain == "taxi":
                        if "taxi_phone" not in turn["source"]:
                            turn["source"]["[taxi_phone]"] = "xxx"
                    else:
                        fig = "[" + domain + "_name]"
                        if fig not in turn["source"]:
                            turn["source"][fig] = "xxx"
                if k not in source_template[domain] and "reference" not in k:
                    turn["source"].pop(k)

with open("train.json", mode="w") as f:
    json.dump(train, f, indent=2)
with open("val.json", mode="w") as f:
    json.dump(val, f, indent=2)
with open("test.json", mode="w") as f:
    json.dump(test, f, indent=2)
