import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from memvp.build import create_model
import torch
from dataclasses import dataclass
from memvp.tokenizer import Tokenizer
import json
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import transforms as T
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import warnings

from tqdm import tqdm

transform  = T.Compose(
            [T.Resize((224, 224), interpolation=Image.BICUBIC), 
             T.ToTensor(),
             T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])

@dataclass
class ModelArgs_7B:
    llama_model_path = './data/weights/'
    llm_model = '7B'
    max_seq_len = 512
    hidden_proj = 128
    emb = 320
    cpu_load = False
    adapter_scale = 0.1
    adapter_dim = 12
    gradient_checkpointing = False
    is_train = False
    image_root='/ai/teacher/dkc/Assets/GQA/images'

args = ModelArgs_7B()
llama = create_model(args)


adapter = torch.load('/ai/teacher/dkc/sub/MemVP-G-I-dual/GQA9M-I-dual/checkpoint-11.pth')['model'] #  


sd = {}
for k in adapter:
    sd[k.replace('module.', '')] = adapter[k]
llama.load_state_dict(sd, False)

tokenizer = Tokenizer(model_path=os.path.join(args.llama_model_path, 'tokenizer.model'))


correct = 0


class EvalSet(Dataset):
    def __init__(self):
        self.data = json.load(open('/ai/teacher/dkc/Assets/GQA/jsons/balanced_evalset_list.json'))
        # self.data = json.load(open('/ai/teacher/dkc/Assets/GQA/jsons/random_20000.json'))
        self.img_root = '/ai/teacher/dkc/Assets/GQA/images'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item['question']
        answer = item['answer']
        img_id = item['imageId']
        indicator = 1 if img_id is not None else 0
        img = Image.open(f'{self.img_root}/{img_id}.jpg').convert('RGB')

        w, h = img.size
        half_size = (h//2, w//2)
        hf_img = T.Resize(half_size, interpolation=Image.BICUBIC)(img)

        hf_img = transform(hf_img)
        img = transform(img)
        _format_q = f"Question: {question}?\nResponse:The answer is"
        return _format_q, answer, img, hf_img, indicator

eval_set = EvalSet()
dataloader = DataLoader(dataset=eval_set, batch_size=64)
count = 0
correct = 0
flag = False
for qs, answsers, imgs, hf_imgs, indicators in tqdm(dataloader):
    preds = llama.generate(qs, imgs, hf_imgs, indicators, 20, tokenizer)
    for idx, pred in enumerate(preds):
        count += 1
        if count == 2000:
            flag = True
        print(f'pred: {pred:<20} gt: {answsers[idx]:<20} {correct:>5}/{count:<5} acc: {float(correct)/count * 100:2.2f}% {eval_set.__len__()}')
        if pred == answsers[idx]:
            correct += 1
    if flag:
        break
    print(f'{correct}/{count}')