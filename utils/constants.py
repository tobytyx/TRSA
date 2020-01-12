# -*- coding: utf-8 -*-
import json

PAD = 0
EOS = 1
SOS = 2
UNK = 3
CLS = 4
SEP = 5

PAD_WORD = '[PAD]'
EOS_WORD = '[EOS]'
SOS_WORD = '[SOS]'
UNK_WORD = '[UNK]'
CLS_WORD = '[CLS]'
SEP_WORD = '[SEP]'

TEMPLATE_MAX_LEN = 50
RESP_MAX_LEN = 40

domains = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'hospital', 'police', 'bus', 'booking', 'general']
functions = ['inform', 'request', 'recommend', 'book', 'select', 'sorry', 'none']
arguments = ['pricerange', 'id', 'address', 'postcode', 'type', 'food', 'phone', 'name', 'area', 'choice', 
             'price', 'time', 'reference', 'none', 'parking', 'stars', 'internet', 'day', 'arriveby', 'departure', 
             'destination', 'leaveat', 'duration', 'trainid', 'people', 'department', 'stay']


used_levels = domains + functions + arguments
act_len = len(used_levels)
db_len = 30

with open('data/ontology.json', 'r') as f:
    ontology = json.load(f)

bs_len = len(ontology["belief_state"])
entity_len = len(ontology["entities"])
