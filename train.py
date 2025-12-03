import os

import datetime
import json
import time
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
from timm.optim import optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from engine import train_one_epoch

from util.datasets import All
from models.build import create_model
import warnings
warnings.filterwarnings("ignore")
import logging

from util.args import get_args_parser, img_path_map
import ipdb
from tqdm import tqdm
import util.misc as misc
import util.lr_sched as lr_sched


def main(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    cudnn.benchmark = False
    args.is_train = True

    # -------------------------------
    # Dataset
    # -------------------------------
    if args.model_name in ['0_ours', '1_zipped', '2_full', '3_hierachical', '3_rebuttal_compress', '3_rebuttal_abstract']:
        dataset_train = All(args.dataset, 'train', 'dual', args.assets_path, args.model_path, img_path_map, args.max_seq_len)
        args.mem_type = 'dual'
    elif args.model_name in ['4_origin']:
        dataset_train = All(args.dataset, 'train', 'single', args.assets_path, args.model_path, img_path_map, args.max_seq_len)
        args.mem_type = 'single'
    else:
        raise Exception('Invalid model name')

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # -------------------------------
    # Model
    # -------------------------------
    model = create_model(args)
    model.to(device)

    # -------------------------------
    # Optimizer
    # -------------------------------
    eff_batch_size = args.batch_size * args.accum_iter

    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations:", args.accum_iter)
    print("effective batch size:", eff_batch_size)

    param_groups = optim_factory.param_groups_weight_decay(model, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))

    # -------------------------------
    # AMP scaler & resume
    # -------------------------------
    loss_scaler = NativeScaler()

    misc.load_model(
        args=args,
        model_without_ddp=model,
        optimizer=optimizer,
        loss_scaler=loss_scaler
    )

    # -------------------------------
    # Logging
    # -------------------------------
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    logging.info(f"Start training for {args.epochs} epochs")
    logging.info(f"Start time: {start_time}")

    # -------------------------------
    # Training loop
    # -------------------------------
    for epoch in range(args.start_epoch, args.epochs):
        model.train()
        accum_iter = args.accum_iter
        optimizer.zero_grad()

        # prefix tokens（每个 epoch 只初始化一次）
        prefix_img = torch.tensor(
            data_loader_train.dataset.tokenizer.encode("Image: ", bos=False, eos=False),
            dtype=torch.int64, device=device
        )
        prefix_nonimg = torch.tensor(
            data_loader_train.dataset.tokenizer.encode("Image: N/A", bos=False, eos=False),
            dtype=torch.int64, device=device
        )

        epoch_loss = 0.0

        pbar = tqdm(
            enumerate(data_loader_train),
            total=len(data_loader_train),
            desc=f"Epoch {epoch}",
            ncols=120
        )

        for data_iter_step, items in pbar:

            # -------- unpack --------
            if args.mem_type == 'single':
                examples, labels, example_mask, images, indicators = items
                half_images = None
            elif args.mem_type == 'dual':
                examples, labels, example_mask, images, half_images, indicators = items
                half_images = half_images.to(device, non_blocking=True)
            else:
                raise ValueError("Invalid mem_type")

            # -------- lr scheduler (per iter) --------
            if data_iter_step % accum_iter == 0:
                lr_sched.adjust_learning_rate(
                    optimizer,
                    data_iter_step / len(data_loader_train) + epoch,
                    args
                )

            # -------- move to device --------
            examples = examples.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            images = images.to(device, non_blocking=True)
            indicators = indicators.to(device, non_blocking=True)

            # -------- forward --------
            if args.mem_type == 'single':
                c_loss = model(
                    examples, labels,
                    images=images,
                    prefix_img=prefix_img,
                    prefix_nonimg=prefix_nonimg,
                    img_indicators=indicators
                )
            else:  # dual
                c_loss = model(
                    examples, labels,
                    images=images,
                    half_images=half_images,
                    prefix_img=prefix_img,
                    prefix_nonimg=prefix_nonimg,
                    img_indicators=indicators
                )

            # -------- NaN protection --------
            if torch.isnan(c_loss):
                print(f"[Warning] NaN loss at epoch {epoch}, iter {data_iter_step}")
                c_loss = torch.nan_to_num(c_loss) * 0

            loss_value = c_loss.item()
            epoch_loss += loss_value

            # -------- backward (with grad accumulation) --------
            loss = c_loss / accum_iter
            loss_scaler(
                loss,
                optimizer,
                parameters=model.parameters(),
                update_grad=(data_iter_step + 1) % accum_iter == 0,
                clip_grad=args.clip_grad
            )

            if (data_iter_step + 1) % accum_iter == 0:
                optimizer.zero_grad()

            # -------- tqdm display --------
            lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix({
                "loss": f"{loss_value:.4f}",
                "lr": f"{lr:.2e}"
            })

        # -------- epoch statistics --------
        avg_loss = epoch_loss / len(data_loader_train)
        train_stats = {
            "closs": avg_loss,
            "lr": optimizer.param_groups[0]["lr"]
        }

        print(f"Epoch {epoch} finished | Avg Loss: {avg_loss:.6f}")


        if args.output_dir:
            misc.save_model(args=args, model=model, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch)

            log_stats = { **{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
            
            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")
                




    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    logging.info('Training time {}'.format(total_time_str))






if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.log_dir:
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - [%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filemode='a',
        filename=f'./{args.log_dir}/{dt}-Train-{args.dataset}-{args.model_name}.log'
    )
    
    main(args)
