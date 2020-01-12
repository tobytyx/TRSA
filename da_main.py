# -*- coding: utf-8 -*-
import logging
from logging.handlers import RotatingFileHandler
import os
import torch
import json
from utils.eval_tools import obtain_tp_tn_fn_fp
from utils import constants
from utils.parsers import da_parse_opt
from icecream import ic
from utils.scheduler import MySchedule, MyMSSchedule
from models.htn_prediction import HTNPrediction


def train(model, train_loader, val_loader, device, args):
    ic(args)
    checkpoint_dir = os.path.join("checkpoints", args["name"])
    model_path = os.path.join(checkpoint_dir, "model_{}.bin")
    args_path = os.path.join(checkpoint_dir, "args.json")

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    if args["model"] == "bert":
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = torch.optim.Adam(optimizer_grouped_parameters, args["learning_rate"])
    else:
        optimizer = torch.optim.Adam(model.parameters(), args["learning_rate"])
    if args["scheduler"] == "linear":
        scheduler = MySchedule(optimizer, 5, args["epoch"])
    else:
        scheduler = MyMSSchedule(optimizer, 3, [30, 50, 70])
    loss_fct = torch.nn.BCEWithLogitsLoss()
    best_f1, best_p, best_r, loss = 0, 0, 0, 0
    best_epoch = 0
    with open(args_path, mode="w") as f:
        f.write(json.dumps(args, indent=2))
    for epoch in range(1, args["epoch"] + 1):
        model.train()
        for _, batch in enumerate(train_loader):
            *_, da, _ = batch
            da = da.to(device)
            optimizer.zero_grad()
            model.zero_grad()
            labels = model(batch)
            loss = loss_fct(labels.view(-1, 1), da.view(-1, 1))
            loss.backward()
            optimizer.step()
        scheduler.step()
        if epoch % 5 == 0 or epoch == args["epoch"]:
            logger.info(
                "Epoch {}, Loss: {:.4f}, lr: {:.7f}".format(
                    epoch, loss.item(), optimizer.param_groups[0]['lr']
                )
            )
            torch.save(model.state_dict(), model_path.format(epoch))
            f1, p, r, _ = evaluate(model, val_loader, args)
            logger.info(" F1: {:.4f}%, P: {:.4f}%, R:{:.4f}%".format(f1 * 100, p*100, r*100))
            if f1 > best_f1:
                best_f1, best_p, best_r = f1, p, r
                best_epoch = epoch
                logger.info("update")
    with open(os.path.join(checkpoint_dir, "result.txt"), mode="w") as f:
        f.write("Best Epoch: {}, F1: {:.3f}%, Precision {:.3f}%, Recall: {:.3f}%\n".format(
            best_epoch, best_f1*100, best_p*100, best_r*100)
        )


def evaluate(model, data_loader, args):
    model.eval()
    tp, tn, fn, fp = 0, 0, 0, 0
    model_turns = {}
    for _, batch in enumerate(data_loader):
        names, turns, *_, da, _ = batch
        pred_das = model(batch)
        pred_das = (pred_das > args["th"]).cpu().long().data.numpy()
        da = da.long().data.numpy()
        tp, tn, fn, fp = obtain_tp_tn_fn_fp(pred_das, da, tp, tn, fn, fp)
        for n in range(len(names)):
            name, turn = names[n], turns[n]
            if name not in model_turns:
                model_turns[name] = {}
            model_turns[name][str(turn)] = pred_das[n]
    precision = tp / (tp + fp + 0.001)
    recall = tp / (tp + fn + 0.001)
    f1 = 2 * precision * recall / (precision + recall + 0.001)
    return f1, precision, recall, model_turns


def test(model, data_loader, args):
    ic(args)
    logger.info("----------Test Part----------")
    model.eval()
    f1, p, r, model_turns = evaluate(model, data_loader, args)
    logger.info("Dialogue action prediction F1: {:.3f}%".format(f1 * 100))
    with open(os.path.join("checkpoints", args["name"], "pred_das.json"), mode="w") as f:
        json.dump(model_turns, f)


def main():
    args = da_parse_opt()
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
    model_type = args["model"].lower()
    if model_type == "bert":
        from pytorch_transformers import BertTokenizer
        from data_provider.bert_predictor_dataset import prepare_dataloader
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        n_token = tokenizer.vocab_size
    else:
        from utils.tools import Tokenizer
        if model_type == "trsa":
            from data_provider.mem_dataset import prepare_dataloader
        else:
            from data_provider.htn_dataset import prepare_dataloader
        with open(os.path.join(args["data_dir"], "vocab.json"), 'r') as f:
            vocabulary = json.loads(f.read())
        vocab, ivocab = vocabulary['vocab'], vocabulary['rev']
        tokenizer = Tokenizer(vocab, ivocab)
        n_token = len(vocab)
    model = HTNPrediction(n_token, constants.act_len, model_type, device, args).to(device)
    if 'train' in args["option"]:
        if args["option"] == "full_train":
            train_loader = prepare_dataloader(args, tokenizer, mode="full_train")
            val_loader = prepare_dataloader(args, tokenizer, mode="test")
        else:
            train_loader = prepare_dataloader(args, tokenizer, mode="train")
            val_loader = prepare_dataloader(args, tokenizer, mode="val")
        train(model, train_loader, val_loader, device, args)
    elif 'test' in args["option"]:
        test_loader = prepare_dataloader(args, tokenizer, mode="test")
        model_path = os.path.join(checkpoint_dir, "model.bin")
        if args["load_model"]:
            model_path = os.path.join(checkpoint_dir, args["load_model"])
        model.load_state_dict(torch.load(model_path))
        test(model, test_loader, args)
    else:
        logger.info("Wrong option!")


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y/%m/%d %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)
    main()
