import torch
from dataclasses import dataclass
from torch.utils.data import DataLoader
import os
from util.datasets import All
import argparse
from models.build import create_model
from models.tokenizer import Tokenizer

from util.misc import setup_for_distributed
import logging
from datetime import datetime

setup_for_distributed(True)

class Evaluator:
    def __init__(self):
        pass
    def evaluate(self, resAns, gts):
        assert isinstance(gts, list), "Wrong gts type"
        for ans in gts:
            flag = False
            for pred_slice in resAns.lower().split(' '):
                if pred_slice in ans.lower().split(' '):
                    flag = True
                    break
            if flag:
                return True
        return False


# @dataclass
# class ModelArgs_7B:
#     llama_model_path = '/ai/teacher/dkc/Assets/weights/'
#     llm_model = '7B'
#     max_seq_len = 512
#     hidden_proj = 128
#     emb = 320
#     cpu_load = False
#     adapter_scale = 0.1
#     adapter_dim = 12
#     gradient_checkpointing = False
#     is_train = False
#     dataset=''
#     compression_level = 0


def collate_fn(batch):
    format_qs, answers, imgs, hf_imgs, indicators = [], [], [], [], []
    for item in batch:
        _format_q, answer, img, hf_img, indicator = item
        format_qs.append(_format_q)
        answers.append(answer)
        imgs.append(img)
        hf_imgs.append(hf_img)
        indicators.append(indicator)
    imgs = torch.stack(imgs, 0)
    hf_imgs = torch.stack(hf_imgs, 0)
    return format_qs, answers, imgs, hf_imgs, indicators

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--compression_level', type=int, required=True)

def main(args):
    model_args.dataset = args.dataset
    model_args.compression_level = args.compression_level
    # model_args = ModelArgs_7B()
    llama = create_model(model_args)

    adapter = torch.load(f'./ckpts/{args.dataset}-level{args.compression_level}checkpoint-19.pth')['model'] # OKVQA训练了20个epoch

    sd = {}
    for k in adapter:
        print(k)
        sd[k.replace('module.', '')] = adapter[k]

    llama.load_state_dict(sd, False)

    tokenizer = Tokenizer(model_path=os.path.join(model_args.llama_model_path, 'tokenizer.model'))

    eval_set = All(args.dataset, 'val')
    evaluator = Evaluator()
    dataloader = DataLoader(dataset=eval_set, batch_size=32, collate_fn=collate_fn)
    count = 0
    correct = 0
    flag = False
    type_names = []
    type_counter_correct = []
    type_counter_all = []

    for qs, answsers, imgs, hf_imgs, indicators, types in dataloader:
        preds = llama.generate(qs, imgs, hf_imgs, indicators, 20, tokenizer)
        for idx, pred in enumerate(preds):
            if types[idx] not in type_names:
                type_names.append(types[idx])
                type_counter_correct.append(0)
                type_counter_all.append(0)
            count += 1
            type_counter_all[type_names.index(types[idx])] += 1
            if count == 999999:
                flag = True
            if evaluator.evaluate(pred, answsers[idx]):
                correct += 1
                type_counter_correct[type_names.index(types[idx])] += 1
            print(f'type: {types[idx]:<10} {qs[idx]:<30}\npred: {pred:<20} gt: {str(answsers[idx]):<20}\n {correct:>5}/{count:<5} acc: {float(correct)/count * 100:2.2f}% {eval_set.__len__()}')
            print('|' + ''.join([f'{item:^10}|' for item in type_names]))
            print('|' + ''.join([f'{(type_counter_correct[idx] * 100 /type_counter_all[idx]):^10.2f}|' for idx, _ in enumerate(type_names)]))

        if flag:
            break
        print(f'{correct}/{count}')


if __name__ == '__main__':
    args = get_args_parser()
    dt = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    args = args.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filemode='w',
        filename=f'./{dt}-Val-{args.dataset}-level{args.level}.log' 
    )
    main(args)