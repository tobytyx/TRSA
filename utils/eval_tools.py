# -*- coding: utf-8 -*-
import torch
import json
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from utils.constants import used_levels, SOS_WORD, EOS_WORD, ontology
import numpy
import copy
import sqlite3

domains = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'hospital', 'police']
requestables = ['phone', 'address', 'postcode', 'reference', 'id']
bleu_smooth = SmoothingFunction()
conn = sqlite3.connect("data/whole.db")
cursor = conn.cursor()


def vec2da(vec):
    return [used_levels[i] for i in range(len(vec)) if vec[i] > 0]


def clean(string):
    return string.lower().replace("'", "''").strip()


def query_result(domain, turn, real_belief=False):
    # query the db
    sql_query = "select * from {}".format(domain)
    items = turn.items() if real_belief else turn['metadata'][domain]['semi'].items()
    conditions = []
    for key, val in items:
        if val in ["", "dontcare", "not mentioned", "don't care", "dont care", "do n't care"]:
            pass
        else:
            val2 = clean(val)
            if key == 'leaveAt':
                conditions.append(r" " + key + " > " + r"'" + val2 + r"'")
            elif key == 'arriveBy':
                conditions.append(r" " + key + " < " + r"'" + val2 + r"'")
            else:
                conditions.append(r" " + key + "=" + r"'" + val2 + r"'")

    if len(conditions) == 0:
        return []
    sql_query += " where " + " and ".join(conditions)
    try:
        result = cursor.execute(sql_query).fetchall()
        return result
    except Exception:
        print(sql_query)
        return []


def obtain_tp_tn_fn_fp(pred, act, tp, tn, fn, fp, elem_wise=False):
    if isinstance(pred, torch.Tensor):
        if elem_wise:
            tp += ((pred.data == 1) & (act.data == 1)).sum(0)
            tn += ((pred.data == 0) & (act.data == 0)).sum(0)
            fn += ((pred.data == 0) & (act.data == 1)).sum(0)
            fp += ((pred.data == 1) & (act.data == 0)).sum(0)
        else:
            tp += ((pred.data == 1) & (act.data == 1)).cpu().sum().item()
            tn += ((pred.data == 0) & (act.data == 0)).cpu().sum().item()
            fn += ((pred.data == 0) & (act.data == 1)).cpu().sum().item()
            fp += ((pred.data == 1) & (act.data == 0)).cpu().sum().item()
        return tp, tn, fn, fp
    else:
        tp += ((pred > 0).astype('long') & (act > 0).astype('long')).sum()
        tn += ((pred == 0).astype('long') & (act == 0).astype('long')).sum()
        fn += ((pred == 0).astype('long') & (act > 0).astype('long')).sum()
        fp += ((pred > 0).astype('long') & (act == 0).astype('long')).sum()
        return tp, tn, fn, fp


def f1_score(hypothesis, corpus):
    """
    :param hypothesis: dict
    :param corpus:
    :return F1: F1 score
    """
    tp, tn, fn, fp = 0, 0, 0, 0
    # accumulate ngram statistics
    files = hypothesis.keys()
    for f in files:
        hyps, refs = hypothesis[f], corpus[f]
        hyps, refs = [hyp.split() for hyp in hyps], [ref.split() for ref in refs]
        # Shawn's evaluation
        for hyp, ref in zip(hyps, refs):
            pred = numpy.zeros((len(ontology["entities"]),), 'float32')
            gt = numpy.zeros((len(ontology["entities"]),), 'float32')
            for h in hyp:
                if h in ontology["entities"]:
                    pred[ontology["entities"].index(h)] += 1
            for r in ref:
                if r in ontology["entities"]:
                    gt[ontology["entities"].index(r)] += 1
            tp, tn, fn, fp = obtain_tp_tn_fn_fp(pred, gt, tp, tn, fn, fp)

    precision = tp / (tp + fp + 0.001)
    recall = tp / (tp + fn + 0.001)
    f1 = (2 * precision * recall + 0.001) / (precision + recall + 0.001)
    return f1


def bleu_score(hypothesis, references):
    refs = []
    hyps = []
    for name in hypothesis.keys():
        assert len(hypothesis[name]) == len(references[name])
        for i in range(len(hypothesis[name])):
            ref = references[name][i].split(' ')
            hyp = hypothesis[name][i].split(' ')
            # to compile with the baseline
            ref = [SOS_WORD] + ref + [EOS_WORD]
            hyp = [SOS_WORD] + hyp + [EOS_WORD]
            refs.append([ref])
            hyps.append(hyp)
    score = corpus_bleu(refs, hyps, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=bleu_smooth.method2)
    return score


def parse_goal(goal, d, domain):
    """
    Parses user goal into dictionary format.
    :param goal: dict
    :param d: ground truth dialogue, from delex.json
    :param domain: str
    :return goal
    """
    goal[domain] = {}
    goal[domain] = {'informable': [], 'requestable': [], 'booking': []}
    if 'info' in d['goal'][domain]:
        if domain == 'train':
            # we consider dialogues only where train had to be booked!
            if 'book' in d['goal'][domain]:
                goal[domain]['requestable'].append('reference')
            if 'reqt' in d['goal'][domain]:
                if 'trainID' in d['goal'][domain]['reqt']:
                    goal[domain]['requestable'].append('id')
        else:
            if 'reqt' in d['goal'][domain]:
                for s in d['goal'][domain]['reqt']:  # additional requests:
                    if s in requestables:
                        # ones that can be easily delexicalized
                        goal[domain]['requestable'].append(s)
            if 'book' in d['goal'][domain]:
                goal[domain]['requestable'].append("reference")
        goal[domain]["informable"] = d['goal'][domain]['info']
        if 'book' in d['goal'][domain]:
            goal[domain]["booking"] = d['goal'][domain]['book']
    return goal


def evaluate_dialogue(dialog, real_dialogue):
    """
    get the list of domains in the goal
    :param dialog:
    :param real_dialogue:
    :return:
    """
    goal = {}
    for domain in domains:
        if real_dialogue['goal'][domain]:
            goal = parse_goal(goal, real_dialogue, domain)
    real_requestables = {domain: copy.copy(goal[domain]['requestable']) for domain in goal.keys()}
    # CHECK IF MATCH HAPPENED
    venue_offered = {domain: [] for domain in goal.keys()}
    provided_requestables = {domain: [] for domain in goal.keys()}
    # analysis the infomation from the generated dialogue
    for turn, sent_t in enumerate(dialog):
        for domain in goal.keys():
            # Search for the only restaurant, hotel, attraction or train with an ID
            if '[{}_name]'.format(domain) in sent_t or 'trainid]' in sent_t:
                if domain in ['restaurant', 'hotel', 'attraction', 'train']:
                    # get the possible scope by delex id/name
                    venues = query_result(domain, real_dialogue['log'][turn * 2 + 1])
                    # if venue has changed
                    if len(venue_offered[domain]) == 0 and venues:
                        venue_offered[domain] = venues  # random.sample(venues, 1)
                    else:
                        flag = True
                        for ven in venue_offered[domain]:
                            if ven not in venues:
                                flag = False
                                break
                        if not flag and venues:  # sometimes there are no results so sample won't work
                            venue_offered[domain] = venues
                else:
                    venue_offered[domain] = '[' + domain + '_name]'
            for requestable in requestables:
                if requestable == 'reference':
                    if domain + '_reference' in sent_t:
                        if 'restaurant_reference' in sent_t:
                            if real_dialogue['log'][turn * 2]['db_pointer'][-5] == 1:
                                provided_requestables[domain].append('reference')

                        elif 'hotel_reference' in sent_t:
                            if real_dialogue['log'][turn * 2]['db_pointer'][-3] == 1:
                                provided_requestables[domain].append('reference')

                        elif 'train_reference' in sent_t:
                            if real_dialogue['log'][turn * 2]['db_pointer'][-1] == 1:
                                provided_requestables[domain].append('reference')
                        else:
                            provided_requestables[domain].append('reference')
                else:
                    if domain + '_' + requestable + ']' in sent_t:
                        provided_requestables[domain].append(requestable)

    # if name was given in the task
    for domain in goal.keys():
        if 'name' in goal[domain]['informable']:
            venue_offered[domain] = '[' + domain + '_name]'

        # special domains - entity does not need to be provided
        if domain in ['taxi', 'police', 'hospital']:
            venue_offered[domain] = '[' + domain + '_name]'
        if domain == 'train':
            if not venue_offered[domain]:
                if goal[domain]['requestable'] and 'id' not in goal[domain]['requestable']:
                    venue_offered[domain] = '[' + domain + '_name]'
    """
    Given all inform and requestable slots, we go through each domain from the user goal
    and check whether right entity was provided and all requestable slots were given to the user.
    The dialogue is successful if that's the case for all domains.
    """
    # HARD EVAL
    # [match, success, checked]
    stats = {'restaurant': [0, 0, 0], 'hotel': [0, 0, 0], 'attraction': [0, 0, 0], 'train': [0, 0, 0],
             'taxi': [0, 0, 0], 'hospital': [0, 0, 0], 'police': [0, 0, 0]}

    match = 0  # match domain
    success = 0  # successfully providing all requests
    # MATCH
    for domain in goal.keys():
        match_stat = 0
        if domain in ['restaurant', 'hotel', 'attraction', 'train']:
            if type(venue_offered[domain]) is str and '_name' in venue_offered[domain]:
                match += 1
                match_stat = 1
            elif venue_offered[domain]:
                groundtruth = query_result(domain, goal[domain]['informable'], real_belief=True)
                if set(venue_offered[domain]).issubset(set(groundtruth)):
                    match += 1
                    match_stat = 1
        else:
            if '[' + domain + '_name]' in venue_offered[domain]:
                match += 1
                match_stat = 1
        stats[domain][0] = match_stat
        stats[domain][2] = 1
    match = 1 if match == len(goal) else 0

    # SUCCESS
    if match:
        for domain in goal.keys():
            success_stat = 0
            domain_success = 0
            if len(real_requestables[domain]) == 0:
                success += 1
                success_stat = 1
                stats[domain][1] = success_stat
                continue
            # if values in sentences are super set of requestables
            for request in set(provided_requestables[domain]):
                if request in real_requestables[domain]:
                    domain_success += 1

            if domain_success >= len(real_requestables[domain]):
                success += 1
                success_stat = 1

            stats[domain][1] = success_stat

        success = 1 if success >= len(real_requestables) else 0
    return success, match, stats


def evaluate_model(dialogues):
    """Gathers statistics for the whole sets."""
    with open('data/raw_data/delex.json') as f:
        delex_dialogues = json.load(f)
    fails = {"match_fail": [], "success_fail": []}
    successes, matches = 0, 0
    total = 0
    for filename, dial in dialogues.items():
        if filename not in delex_dialogues:
            filename += ".json"
        data = delex_dialogues[filename]
        success, match, _ = evaluate_dialogue(dial, data)
        if not success:
            fails["success_fail"].append(filename)
        if not match:
            fails["match_fail"].append(filename)
        successes += success
        matches += match
        total += 1

        # Print results
    matches = matches / float(total)
    successes = successes / float(total)
    return matches, successes, fails


def main():
    with open("data/test.json") as f:
        test = json.load(f)
    dialogues = {}
    for dial in test:
        dialogues[dial["name"]] = []
        for turn in dial["turns"]:
            dialogues[dial["name"]].append(turn["sys"])
    with open("data/references/test_reference.json") as f:
        gt_turns = json.loads(f.read())
    bleu = bleu_score(dialogues, gt_turns)
    entity_f1 = f1_score(dialogues, gt_turns)
    m, s, fails = evaluate_model(dialogues)
    print("bleu: {:.3f}%, F1: {:.3f}%, matches: {:.2f}%, successes: {:.2f}%".format(
        bleu*100, entity_f1*100, m*100, s*100)
    )


if __name__ == '__main__':
    main()
