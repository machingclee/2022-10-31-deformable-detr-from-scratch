
import argparse
import datetime
import random
import time
import numpy as np
import os
import torch
from datasets.face.wider_face_dataset import FacialLandmarkTrainingDataset
import util.misc as utils

from datasets.rust.rust_dataset import RustAnnotationDataset
from datasets.common_utils import collate_fn
from argument_parser import get_args_parser, ParsedArgument
from pathlib import Path
from torch.utils.data import DataLoader
from train_one_epoch import train_one_epoch
from deformable_detr import build
from typing import TypedDict
from prodict import Prodict


def train(args: ParsedArgument):
    print("git:\n  {}\n".format(utils.get_sha()))
    print(args)

    device = torch.device(args.device)
    args.pre_norm = False
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build(args)
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    # dataset_train = build_dataset(image_set='train', args=args)
    dataset_train = FacialLandmarkTrainingDataset()
    # dataset_val = build_dataset(image_set='val', args=args)

    data_loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    # data_loader_val = DataLoader(dataset_val, args.batch_size)

    # lr_backbone_names = ["backbone.0", "backbone.neck", "input_proj", "transformer.encoder"]
    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    for n, p in model_without_ddp.named_parameters():
        print(n)

    param_dicts = [
        {
            "params":
                [param for name, param in model_without_ddp.named_parameters()
                 if not match_name_keywords(name, args.lr_backbone_names) and not match_name_keywords(name, args.lr_linear_proj_names) and param.requires_grad],
            "lr": args.lr,
        },
        {
            "params":
                [param for name, param in model_without_ddp.named_parameters()
                 if match_name_keywords(name, args.lr_backbone_names) and param.requires_grad],
            "lr": args.lr_backbone,
        },
        {
            "params":
                [param for name, param in model_without_ddp.named_parameters()
                 if match_name_keywords(name, args.lr_linear_proj_names) and param.requires_grad],
            "lr": args.lr * args.lr_linear_proj_mult,
        }
    ]
    if args.sgd:
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                      weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    output_dir = Path(args.output_dir)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')

        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]

        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))

        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))

        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            import copy
            p_groups = copy.deepcopy(optimizer.param_groups)
            optimizer.load_state_dict(checkpoint['optimizer'])

            for pg, pg_old in zip(optimizer.param_groups, p_groups):
                pg['lr'] = pg_old['lr']
                pg['initial_lr'] = pg_old['initial_lr']

            print(optimizer.param_groups)
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            # todo: this is a hack for doing experiment that resume from checkpoint
            # and also modify lr scheduler (e.g., decrease lr in advance).
            args.override_resumed_lr_drop = True

            if args.override_resumed_lr_drop:
                print('Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler.')
                lr_scheduler.step_size = args.lr_drop
                lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))

            lr_scheduler.step(lr_scheduler.last_epoch)
            args.start_epoch = checkpoint['epoch'] + 1

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm
        )
        lr_scheduler.step()
        if args.output_dir:
            model_path = os.path.join(args.output_dir, "{}.pth".format(str(epoch).zfill(3)))
            state_dict = model.state_dict()
            torch.save(state_dict, model_path)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args: ParsedArgument = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train(args)
