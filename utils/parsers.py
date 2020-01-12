# -*- coding: utf-8 -*-
import argparse


def gen_parse_opt():
    parser = argparse.ArgumentParser()
    # normal
    parser.add_argument('--option', type=str, default="full_train",
                        choices=['train', 'test', 'full_train', 'postprocess'])
    parser.add_argument('--epoch', type=int, default=150)
    parser.add_argument("--name", default="default", type=str)
    parser.add_argument("--encoder", default="trsa", type=str, choices=["trsa", "transformer"])
    parser.add_argument("--decoder", default="hdsa", type=str, choices=["hdsa", "transformer"])
    parser.add_argument('--scheduler', type=str, default="milestone", choices=["linear", "milestone"])
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument("--load_model", type=str, default="")
    parser.add_argument("--no_cuda", action='store_true')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument("--learning_rate", default=1e-3, type=float, help="The initial learning rate for Adam.")
    parser.add_argument('--joint', action="store_true")
    parser.add_argument('--da', type=str, default="predict", choices=["predict", "gt", "bert", "trsa"],
                        help="The place to load da.")
    parser.add_argument("--th", type=float, default=0.4, help="Threshold")
    parser.add_argument("--alpha", type=float, default=0.5, help="The percentage of gen loss.")
    parser.add_argument("--no_cnn", action='store_true')
    parser.add_argument('--num_filters', type=int, default=32)
    parser.add_argument('--pre_layer_norm', action="store_true", help="Do layer norm before attention")
    parser.add_argument('--max_seq_length', type=int, default=50)
    parser.add_argument('--data_source', action="store_true")
    parser.add_argument('--en_layer', type=int, default=3, help='number of encoder layers')
    parser.add_argument('--de_layer', type=int, default=3, help='number of decoder layers')
    parser.add_argument('--n_head', type=int, default=4, help='number of heads')
    parser.add_argument('--d_model', type=int, default=128, help='model_type dimension')
    parser.add_argument('--dropout', type=float, default=0.2, help='global dropout rate')
    parser.add_argument('--drop_att', type=float, default=0.2, help='attention probability dropout rate')
    parser.add_argument('--n_bm', type=int, default=2, help='number of beams for beam search')
    parser.add_argument("--no_share_weight", action="store_true",
                        help="don't share the weight between decoder embedding & output project")
    args = parser.parse_args()
    return args


def da_parse_opt():
    parser = argparse.ArgumentParser()
    # normal
    parser.add_argument('--option', type=str, default="full_train",
                        choices=['train', 'test', 'full_train'])
    parser.add_argument('--epoch', type=int, default=150)
    parser.add_argument("--name", default="da_default", type=str)
    parser.add_argument('--model', type=str, default="trsa", choices=["transformer", "bert", "trsa"])
    parser.add_argument('--scheduler', type=str, default="milestone", choices=["linear", "milestone"])
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument("--load_model", type=str, default="")
    parser.add_argument("--no_cuda", action='store_true')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument("--learning_rate", default=1e-3, type=float, help="The initial learning rate for Adam.")
    parser.add_argument('--n_layer', type=int, default=3, help='number of total layers')
    parser.add_argument("--th", type=float, default=0.4, help="Threshold")
    parser.add_argument('--da', type=str, default="gt")
    parser.add_argument("--no_cnn", action='store_true')
    parser.add_argument('--num_filters', type=int, default=32)
    parser.add_argument('--pre_layer_norm', action="store_true", help="Do layer norm before attention")
    parser.add_argument('--max_seq_length', type=int, default=50)
    parser.add_argument('--data_source', action="store_true")
    parser.add_argument('--n_head', type=int, default=4, help='number of heads')
    parser.add_argument('--d_model', type=int, default=128, help='model_type dimension')
    parser.add_argument('--dropout', type=float, default=0.2, help='global dropout rate')
    parser.add_argument('--drop_att', type=float, default=0.2, help='attention probability dropout rate')
    args = parser.parse_args()
    return args
