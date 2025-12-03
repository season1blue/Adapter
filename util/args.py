
import argparse

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int, help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--llama_model_path', default='./llama', type=str, help='path of llama model')
    parser.add_argument('--llm_model', default='7B', type=str, metavar='MODEL', help='Name of llm model to train')
    parser.add_argument('--cpu_load',  action='store_true',   help='load the model on cpu and avoid OOM on gpu')
    parser.add_argument('--emb', type=int, default=320)
    parser.add_argument('--adapter_dim', type=int, default=8, metavar='LENGTH', help='the dims of adapter layer')
    parser.add_argument('--hidden_proj', type=int, default=128, metavar='LENGTH', help='the visual adapter dim')
    parser.add_argument('--adapter_scale', type=float, default=1., metavar='LENGTH', help='the scales of adapter layer')
    parser.add_argument('--max_seq_len', type=int, default=512, metavar='LENGTH', help='the maximum sequence length')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.02, help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='clip gradient', help='clips gradient norm of an iterable of parameters')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR', help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR', help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--gradient_checkpointing', action='store_true', help='saving memory costs via gradient_checkpointing')
    parser.add_argument('--warmup_epochs', type=float, default=40, metavar='N', help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--assets_path', type=str, default='/ai/teacher/dkc/HZIP/Assets')
    parser.add_argument('--model_path', type=str, default="weights")
    
    parser.add_argument('--output_dir', default='./ckpts', help='path where to save, empty for no saving')
    parser.add_argument('--tsbd', default='logs/tsbd', help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true', help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--log_dir', type=str, default='./logs/')
    

    return parser



img_path_map = {
    'VCR/train'                      : '../all_data/mqa/Rebuttal/VCR/image',
    'VCR/val'                        : '../all_data/mqa/Rebuttal/VCR/image',
    'MMBench/train'                  : '../all_data/mqa/Rebuttal/MMBench/image',
    'MMBench/val'                    : '../all_data/mqa/Rebuttal/MMBench/image',
    'SEED/train'                     : '../all_data/mqa/Rebuttal/SEED/dataset/data/image',
    'SEED/val'                       : '../all_data/mqa/Rebuttal/SEED/dataset/data/image',
    # 
    'VisWiz-Full/train'              : '../all_data/mqa/vizwiz/train',
    'VisWiz-Full/val'                : '../all_data/mqa/vizwiz/val',
    'POPE/val'                       : '../all_data/mqa/Rebuttal/POPE/dataset/Full/imgs',
    'MME/val'                        : '../all_data/mqa/Rebuttal/MME/data/images',
    'GQA-FULL/train'                 : '../all_data/mqa/GQA/images',
    'GQA-FULL/val'                   : '../all_data/mqa/GQA/images',
    'GQA/train'                      : '../all_data/mqa/GQA/images',
    'GQA/val'                        : '../all_data/mqa/GQA/images',
    'OKVQA/train'                    : '../all_data/mqa/OKVQA/train2014',
    'OKVQA/val'                      : '../all_data/mqa/OKVQA/val2014',
    'TextVQA/train'                  : '../all_data/mqa/TextVQA/train_val',
    'TextVQA/val'                    : '../all_data/mqa/TextVQA/train_val',
    'VisWiz/train'                   : '../all_data/mqa/vizwiz/train',
    'VisWiz/val'                     : '../all_data/mqa/vizwiz/val',
    'VQAv2-Full/train'               : '../all_data/mqa/VQAv2/train2014',
    'VQAv2-Full/val'                 : '../all_data/mqa/VQAv2/val2014',
    'VQAv2/train'                    : '../all_data/mqa/VQAv2/train2014',
    'VQAv2/val'                      : '../all_data/mqa/VQAv2/val2014',
}