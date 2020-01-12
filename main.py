# -*- coding: utf-8 -*-
import logging
from logging.handlers import RotatingFileHandler
import os
import torch
import json
from utils.parsers import gen_parse_opt
from models.htn import HTNetwork
from utils.tools import Tokenizer, dedelex, get_attn_graph
from utils import constants
from utils.eval_tools import f1_score, bleu_score, evaluate_model, obtain_tp_tn_fn_fp, vec2da
from icecream import ic
from utils.scheduler import MySchedule, MyMSSchedule
import pickle


def cal_performance(loss_fc, labels, logits, y, da, alpha=0.5):
    """
    :param loss_fc: BCEWithLogitsLoss
    :param labels:
    :param logits:
    :param y:
    :param da:
    :param alpha:
    :return:
    """
    if alpha > 1 or alpha < 0:
        alpha = 0.5
    pred = logits.contiguous().view(-1, logits.size(2))
    gold = y[:, 1:].contiguous().view(-1)
    generator_loss = torch.nn.functional.cross_entropy(pred, gold, ignore_index=constants.PAD, reduction="mean")
    if labels is not None:
        classifier_loss = loss_fc(labels.view(-1, 1), da.view(-1, 1))
        loss = alpha * generator_loss + (1-alpha) * classifier_loss
    else:
        loss = generator_loss
    return loss


def train(model, train_loader, val_loader, tokenizer, device, args):
    ic(args)
    checkpoint_dir = os.path.join("checkpoints", args["name"])
    model_path = os.path.join(checkpoint_dir, "model_{}.bin")
    args_path = os.path.join(checkpoint_dir, "args.json")
    optimizer = torch.optim.Adam(model.parameters(), args["learning_rate"])
    if args["scheduler"] == "linear":
        scheduler = MySchedule(optimizer, 5, args["epoch"])
    else:
        scheduler = MyMSSchedule(optimizer, 5, [50, 100, 150])
    loss_fct = torch.nn.BCEWithLogitsLoss()
    best_bleu, bleu_not_best_count, loss = 0, 0, 0
    best_match, best_success, best_entity_f1 = 0, 0, 0
    best_epoch = 0
    with open(args_path, mode="w") as f:
        f.write(json.dumps(args, indent=2))
    for epoch in range(1, args["epoch"] + 1):
        model.train()
        for _, batch in enumerate(train_loader):
            *_, da, y = batch
            da, y = da.to(device), y.to(device)
            optimizer.zero_grad()
            model.zero_grad()
            labels, logits = model(batch, joint=args["joint"])
            loss = cal_performance(loss_fct, labels, logits, y, da, args["alpha"])
            loss.backward()
            optimizer.step()
        scheduler.step()
        if epoch % 5 == 0 or epoch == args["epoch"]:
            logger.info(
                "Epoch {}, Loss: {:.4f}, lr: {:.7f}".format(epoch, loss.item(), optimizer.param_groups[0]['lr'])
            )
            torch.save(model.state_dict(), model_path.format(epoch))
            bleu, entity_f1, match_rate, success_rate = evaluate(model, val_loader, tokenizer, args)
            if success_rate > best_success:
                best_bleu, best_entity_f1 = bleu, entity_f1
                best_match, best_success = match_rate, success_rate
                best_epoch = epoch
                bleu_not_best_count = 0
            else:
                bleu_not_best_count += 1
    logger.info("Best Epoch: {}, BLEU: {:.3f}%, Entity F1 {:.3f}%, Match: {:.3f}%, Success: {:3f}%\n".format(
        best_epoch, best_bleu*100, best_entity_f1*100, best_match*100, best_success*100))


def evaluate(model, data_loader, tokenizer, args):
    if args["option"] == "full_train":
        with open(os.path.join(args["data_dir"], "references/test_reference.json"), mode="r") as f:
            gt_turns = json.loads(f.read())
    else:
        with open(os.path.join(args["data_dir"], "references/val_reference.json"), mode="r") as f:
            gt_turns = json.loads(f.read())
    model.eval()
    model_turns = {}
    da_output = {}
    joint = args["joint"] and (args["da"] == "predict")
    tp, tn, fn, fp = 0, 0, 0, 0
    for _, batch in enumerate(data_loader):
        names, turns, *_, da, y = batch
        hyps, pred_das, _ = model.generate(batch, joint)
        for hyp_step, hyp in enumerate(hyps):
            pred = tokenizer.convert_id_to_tokens(hyp)
            name, turn = names[hyp_step], turns[hyp_step]
            if name not in model_turns:
                model_turns[name] = [(turn, pred)]
            else:
                model_turns[name].append((turn, pred))
        if pred_das is not None:
            pred_das = (pred_das > args["th"]).cpu().long().data.numpy()
            da = da.long().data.numpy()
            for j in range(len(names)):
                if names[j] not in da_output:
                    da_output[names[j]] = {}
                da_output[names[j]][turns[j]] = pred_das[j].tolist()
            tp, tn, fn, fp = obtain_tp_tn_fn_fp(pred_das, da, tp, tn, fn, fp)
    precision = tp / (tp + fp + 0.001)
    recall = tp / (tp + fn + 0.001)
    F1 = 2 * precision * recall / (precision + recall + 0.001)
    logger.info("Dialogue action prediction F1: {:.3f}%".format(F1*100))
    pred_dialogues = {}
    for name, preds in model_turns.items():
        sorted_preds = sorted(preds, key=lambda x: x[0])
        _, pred = list(zip(*sorted_preds))
        if name not in pred_dialogues:
            pred_dialogues[name] = []
        pred_dialogues[name].extend(pred)
    bleu = bleu_score(pred_dialogues, gt_turns)
    entity_f1 = f1_score(pred_dialogues, gt_turns)
    match_rate, success_rate, _ = evaluate_model(pred_dialogues)
    logger.info("BLEU = {:.3f}% EntityF1 = {:.3f}% MATCH = {:.2f}% SUCCESS = {:.2f}%".format(
        bleu*100, entity_f1*100, match_rate*100, success_rate*100)
    )
    return bleu, entity_f1, match_rate, success_rate


def test(model, data_loader, tokenizer, args):
    ic(args)
    logger.info("----------Test Part----------")
    with open('{}/references/test_reference.json'.format(args["data_dir"])) as f:
        gt_turns = json.load(f)
    model.eval()
    model_turns = {}
    # seq_attn = []
    tp, tn, fn, fp = 0, 0, 0, 0
    joint = args["joint"] and (args["da"] == "predict")
    for _, batch in enumerate(data_loader):
        names, turns, *_, da, _ = batch
        hyps, pred_da, attn = model.generate(batch, joint)
        # attn_seq_batch, attn_graph_batch = get_attn_graph(batch[2], attn, encoder_type=args["encoder"])
        # for attn_seq, attn_graph in zip(attn_seq_batch, attn_graph_batch):
        #     seq_attn.append([attn_seq, attn_graph])
        pred_da = pred_da.cpu().detach()
        da = da.long()
        tp, tn, fn, fp = obtain_tp_tn_fn_fp(pred_da, da, tp, tn, fn, fp)
        for n, hyp in enumerate(hyps):
            pred = tokenizer.convert_id_to_tokens(hyp)
            name, turn = names[n], turns[n]
            if name not in model_turns:
                model_turns[name] = []
            model_turns[name].append((turn, pred, da[n], pred_da[n]))
    precision = tp / (tp + fp + 0.001)
    recall = tp / (tp + fn + 0.001)
    f1 = 2 * precision * recall / (precision + recall + 0.001)
    logger.info("DA prediction F1: {:.3f}%, P:{:.3f}, R:{:.3f}".format(f1 * 100, precision, recall))
    pred_dialogues = {}
    pred_das = {}
    fail_das = []
    for name, preds in model_turns.items():
        sorted_preds = sorted(preds, key=lambda x: x[0])
        _, pred, da, pred_da = list(zip(*sorted_preds))
        if name not in pred_dialogues:
            pred_dialogues[name] = []
            pred_das[name] = []
        key = 0
        for d, p in zip(da, pred_da):
            pred_das[name].append({"gt": vec2da(d.tolist()), "pred": vec2da(p.tolist())})
            if torch.sum((d != p)).item() > 0:
                key = 1
        if key == 1:
            fail_das.append(name)
        pred_dialogues[name].extend(pred)
    bleu = bleu_score(pred_dialogues, gt_turns)
    # bleu = bleu_hdsa(pred_dialogues, gt_turns)
    entity_f1 = f1_score(pred_dialogues, gt_turns)
    match_rate, success_rate, fail_dialogues = evaluate_model(pred_dialogues)
    fail_dialogues["da_fail"] = fail_das

    result_sent = "DA F1 = {:.3f}% BLEU = {:.3f}% EntityF1 = {:.3f}% MATCH = {:.2f}% SUCCESS = {:.2f}%".format(
        f1 * 100, bleu * 100, entity_f1 * 100, match_rate * 100, success_rate * 100)
    logger.info(result_sent)

    delex_test_output = os.path.join("checkpoints", args["name"], "delex_test_output.json")
    with open(delex_test_output, mode="w") as f:
        f.write(json.dumps(pred_dialogues, ensure_ascii=False, indent=2))
    with open(os.path.join("checkpoints", args["name"], "pure_output.txt"), mode="w") as f:
        for dial_name in gt_turns.keys():
            for each in pred_dialogues[dial_name]:
                f.write(each+"\n")
    # with open(os.path.join("checkpoints", args["name"], "attn_graph.pkl"), mode="wb") as f:
    #     pickle.dump(seq_attn, f)
    with open(os.path.join("checkpoints", args["name"], "fail_dialogues.json"), mode="w") as f:
        f.write(json.dumps(fail_dialogues))
    with open(os.path.join("checkpoints", args["name"], "pred_das.json"), mode="w") as f:
        f.write(json.dumps(pred_das))
    logger.info("delex test output is saved in {}".format(delex_test_output))


def postprocess(args):
    with open(os.path.join("checkpoints", args["name"], "delex_test_output.json"), 'r') as f:
        pred_dialogues = json.load(f)
    with open('{}/references/test_reference.json'.format(args["data_dir"])) as f:
        gt_turns = json.load(f)
    success_rate = dedelex(pred_dialogues)
    bleu = bleu_score(pred_dialogues, gt_turns)
    with open(os.path.join("checkpoints", args["name"], "non_delex_test_output.json"), 'w') as f:
        f.write(json.dumps(pred_dialogues, indent=2))
    logger.info("Restore BLEU: {:.2f}, Dedelex Success Rate {:.2f}".format(bleu*100, success_rate*100))


def main():
    args = gen_parse_opt()
    args = vars(args)
    formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(message)s')
    checkpoint_dir = os.path.join("checkpoints", args["name"])
    if args["option"] == "test":
        args_path = os.path.join(checkpoint_dir, "args.json")
        with open(args_path, mode="r") as f:
            saved_args = json.loads(f.read())
            saved_args["option"] = "test"
            if args["load_model"]:
                saved_args["load_model"] = args["load_model"]
        args = saved_args
        file_handler = RotatingFileHandler(os.path.join("checkpoints", args["name"], "run.log"), "a")
    else:
        os.mkdir(checkpoint_dir) if not os.path.exists(checkpoint_dir) else ic("checkpoint dir already exists")
        file_handler = RotatingFileHandler(os.path.join("checkpoints", args["name"], "run.log"), "w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    device = torch.device('cuda' if not args["no_cuda"] else 'cpu')
    # create tokenizer
    with open(os.path.join(args["data_dir"], "vocab.json"), 'r') as f:
        vocabulary = json.loads(f.read())
    vocab, ivocab = vocabulary['vocab'], vocabulary['rev']
    tokenizer = Tokenizer(vocab, ivocab)

    # create model
    encoder_type = args["encoder"].lower()
    if encoder_type == "trsa":
        from data_provider.mem_dataset import prepare_dataloader
    else:
        from data_provider.htn_dataset import prepare_dataloader
    model = HTNetwork(len(vocab), constants.act_len, device, args)
    model = model.to(device)
    if 'train' in args["option"]:
        if args["option"] == "full_train":
            train_loader = prepare_dataloader(args, tokenizer, mode="full_train")
            val_loader = prepare_dataloader(args, tokenizer, mode="test")
        else:
            train_loader = prepare_dataloader(args, tokenizer, mode="train")
            val_loader = prepare_dataloader(args, tokenizer, mode="val")
        train(model, train_loader, val_loader, tokenizer, device, args)
    elif 'test' in args["option"]:
        test_loader = prepare_dataloader(args, tokenizer, mode="test")
        model_path = os.path.join(checkpoint_dir, "model.bin")
        if args["load_model"]:
            model_path = os.path.join(checkpoint_dir, args["load_model"])
        model.load_state_dict(torch.load(model_path))
        test(model, test_loader, tokenizer, args)
    elif 'postprocess' in args["option"]:
        postprocess(args)
    else:
        logger.info("Wrong option!")


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%Y/%m/%d %H:%M', level=logging.INFO)
    logger = logging.getLogger(__name__)
    main()
