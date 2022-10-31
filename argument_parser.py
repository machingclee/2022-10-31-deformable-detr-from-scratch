import argparse
import numpy as np
from prodict import Prodict


class ParsedArgument(Prodict):
    lr: float
    lr_backbone_names: str
    lr_backbone: float
    lr_linear_proj_names: str
    lr_linear_proj_mult: float
    batch_size: int
    weight_decay: float
    epochs: int
    lr_drop: int
    lr_drop_epochs: int
    clip_max_norm: float
    sgd: bool
    backbone: str
    dilation: bool
    position_embedding: str
    position_embedding_scale: float
    num_feature_levels: int
    enc_layers: int
    dec_layers: int
    dim_feedforward: int
    hidden_dim: int
    dropout: float
    nheads: int
    num_queries: int
    dec_n_points: int
    enc_n_points: int
    last_height: int
    last_width: int
    no_aux_loss: bool
    set_cost_class: float
    set_cost_bbox: float
    set_cost_giou: float
    mask_loss_coef: float
    dice_loss_coef: float
    cls_loss_coef: float
    bbox_loss_coef: float
    giou_loss_coef: float
    focal_alpha: float
    dataset_file: str
    coco_path: str
    remove_difficult: bool
    output_dir: str
    device: str
    seed: int
    resume: str
    start_epoch: int
    eval: bool
    num_workers: int
    cache_mode: bool
    num_classes: int


def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    # parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_backbone', default=0, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--num_classes', default=2, type=int)
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--pre_norm', default=False, type=bool)

    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--sgd', action='store_true')
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers', default=3, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=3, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=20, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)
    parser.add_argument('--last_height', default=16, type=int,
                        help='The height of C6 feature map')
    parser.add_argument('--last_width', default=16, type=int,
                        help='The height of C6 feature map')

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', default='./data/coco', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='checkpoints',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')

    return parser
