# -*- coding: utf-8 -*-
import logging
from logging.handlers import RotatingFileHandler
import os
import argparse
import torch
import json
import torch.nn.functional as F
from models.hdsa import HDSATransformer
from utils.tools import Tokenizer, dedelex
from torch.optim.lr_scheduler import MultiStepLR
from utils import constants
from utils.eval_tools import f1_score, bleu_score, evaluate_model
from data_provider.hdsa_dataset import prepare_dataloader
from icecream import ic


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_opt():
    parser = argparse.ArgumentParser()
    # normal
    parser.add_argument('--option', type=str, default="train", choices=['train', 'test', 'postprocess'])
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument("--name", default="hdsa_default", type=str)
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument("--load_model", type=str, default="")
    parser.add_argument("--no_cuda", action='store_true')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_seq_length', type=int, default=50)
    parser.add_argument('--layer_num', type=int, default=3)
    parser.add_argument('--emb_dim', type=int, default=128, help="the embedding dimension")
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--th', type=float, default=0.4,)
    parser.add_argument('--head', type=int, default=4)
    parser.add_argument('--n_bm', type=int, default=2, help='number of beams for beam search')
    parser.add_argument('--evaluate_every', type=int, default=5)

    args = parser.parse_args()
    return args


def train(model, train_loader, val_loader, tokenizer, device, args):
    ce_loss_func = torch.nn.CrossEntropyLoss(ignore_index=constants.PAD)
    ce_loss_func.to(device)

    ic(args)
    checkpoint_dir = os.path.join("checkpoints", args["name"])
    model_path = os.path.join(checkpoint_dir, "model_{}.bin")
    args_path = os.path.join(checkpoint_dir, "args.json")

    optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), betas=(0.9, 0.98), eps=1e-09)
    scheduler = MultiStepLR(optimizer, milestones=[50, 100, 150, 200], gamma=0.5)

    best_bleu = 0
    with open(args_path, mode="w") as f:
        f.write(json.dumps(args, indent=2))
    for epoch in range(1, args["epoch"] + 1):
        logger.info("----------Epoch {}-------------".format(epoch))
        model.train()
        mean_loss = 0
        for _, batch in enumerate(train_loader):
            _, _, rep_in, act_vecs, resp_out = batch
            rep_in = rep_in.to(device)
            act_vecs, resp_out = act_vecs.to(device), resp_out.to(device)
            optimizer.zero_grad()
            model.zero_grad()

            logits = model(tgt_seq=resp_out, src_seq=rep_in, act_vecs=act_vecs)
            # ic(logits.size(), resp_out.size())
            loss = ce_loss_func(logits.contiguous().view(-1, logits.size(2)),
                                resp_out[:,1:].contiguous().view(-1))
            loss.backward()
            mean_loss += loss.item()
            optimizer.step()
        mean_loss = mean_loss / len(train_loader)
        logger.info("Loss: {:.5f}, lr: {:.7f}".format(mean_loss, optimizer.param_groups[0]['lr']))
        scheduler.step()
        if mean_loss < 3.0 and epoch % args["evaluate_every"] == 0:
            bleu, entity_f1, match_rate, success_rate = evaluate(model, val_loader, tokenizer, device, args)
            if bleu > best_bleu:
                best_bleu = bleu
                torch.save(model.state_dict(), model_path.format(epoch))


def test(model, data_loader, tokenizer, device, args):
    ic(args)
    logger.info("----------Test Part----------")
    gt_turns = json.load(open('{}/references/test_reference.json'.format(args["data_dir"])))
    model.eval()
    model_turns = {}
    for _, batch in enumerate(data_loader):
        names, turns, rep_in, act_vecs, _ = batch
        rep_in = rep_in.to(device)
        hyps = model.generate(act_vecs, rep_in, args["n_bm"], 40, device)
        for hyp_step, hyp in enumerate(hyps):
            pred = tokenizer.convert_id_to_tokens(hyp)
            name, turn = names[hyp_step], turns[hyp_step]
            if name not in model_turns:
                model_turns[name] = [(turn, pred)]
            else:
                model_turns[name].append((turn, pred))
    pred_dialogues = {}
    for name, preds in model_turns.items():
        sorted_preds = sorted(preds, key=lambda x: x[0])
        _, pred = list(zip(*sorted_preds))
        if name not in pred_dialogues:
            pred_dialogues[name] = []
        pred_dialogues[name].extend(pred)
    bleu = bleu_score(pred_dialogues, gt_turns)
    entity_f1 = f1_score(pred_dialogues, gt_turns)
    match_rate, success_rate, fail_dialogues = evaluate_model(pred_dialogues)

    result_sent = "BLEU = {:.3f}% EntityF1 = {:.3f}% MATCH = {:.2f}% SUCCESS = {:.2f}%".format(
        bleu * 100, entity_f1 * 100, match_rate * 100, success_rate * 100)
    logger.info(result_sent)
    delex_test_output = os.path.join("checkpoints", args["name"], "delex_test_output.json")
    with open(delex_test_output, mode="w") as f:
        f.write(json.dumps(pred_dialogues, ensure_ascii=False, indent=2))
    with open(os.path.join("checkpoints", args["name"], "fail_dialogues.json"), mode="w") as f:
        f.write(json.dumps(fail_dialogues))
    logger.info("delex test output is saved in {}".format(delex_test_output))


def evaluate(model, data_loader, tokenizer, device, args):
    with open(os.path.join(args["data_dir"], "references/val_reference.json"), mode="r") as f:
        gt_turns = json.loads(f.read())
    model.eval()
    model_turns = {}

    for _, batch in enumerate(data_loader):
        names, turns, rep_in, act_vecs, _ = batch
        rep_in, act_vecs = rep_in.to(device), act_vecs.to(device)
        hyps = model.generate(act_vecs, rep_in, args["n_bm"], 40, device)
        for hyp_step, hyp in enumerate(hyps):
            pred = tokenizer.convert_id_to_tokens(hyp)
            name, turn = names[hyp_step], turns[hyp_step]
            if name not in model_turns:
                model_turns[name] = [(turn, pred)]
            else:
                model_turns[name].append((turn, pred))
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


def postprocess(args):
    with open(os.path.join(args.output_dir, args.name, "delex_test_output.json"), 'r') as f:
        model_turns = json.load(f)
    success_rate = dedelex(model_turns)

    with open(os.path.join(args.output_dir, args.name, "non_delex_test_output.json"), 'w') as f:
        f.write(json.dumps(model_turns, indent=2))
    logger.info("Dedelex Success Rate {:.2f}".format(success_rate*100))


def main():
    args = parse_opt()
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
    model = HDSATransformer(
        tokenizer.vocab_len, args["emb_dim"], args["layer_num"], args["emb_dim"], args["head"], args["dropout"])
    model = model.to(device)
    if 'train' in args["option"]:
        train_loader = prepare_dataloader(args, tokenizer, mode="train")
        val_loader = prepare_dataloader(args, tokenizer, mode="val")
        train(model, train_loader, val_loader, tokenizer, device, args)
    elif 'test' in args["option"]:
        test_loader = prepare_dataloader(args, tokenizer, mode="test")
        model_path = os.path.join(checkpoint_dir, "model.bin")
        if args["load_model"]:
            model_path = os.path.join(checkpoint_dir, args["load_model"])
        model.load_state_dict(torch.load(model_path))
        test(model, test_loader, tokenizer, device, args)
    elif 'postprocess' in args["option"]:
        postprocess(args)
    else:
        logger.info("Wrong option!")


if __name__ == '__main__':
    main()
