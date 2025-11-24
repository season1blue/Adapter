import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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
import cv2
import pywt
import numpy as np

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


adapter = torch.load('./ckpts/checkpoint-0.pth')['model'] #  

sd = {}
for k in adapter:
    print(k)
    sd[k.replace('module.', '')] = adapter[k]

llama.load_state_dict(sd, False)

tokenizer = Tokenizer(model_path=os.path.join(args.llama_model_path, 'tokenizer.model'))


class EvalSet(Dataset):
    def __init__(self):
        self.data = json.load(open('/ai/teacher/dkc/Assets/GQA/jsons/balanced_evalset_list.json'))
        # self.data = json.load(open('/ai/teacher/dkc/Assets/GQA/jsons/random_20000.json')) # 用于查看在训练集上的输出, 检查错误
        self.img_root = '/ai/teacher/dkc/Assets/GQA/images'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item['question']
        answer = item['answer']
        img_id = item['imageId']
        indicator = 1 if img_id is not None else 0
        img_path = f'{self.img_root}/{img_id}.jpg'
        img = Image.open(img_path).convert('RGB')
        ndarr = self.low(img_path)
        hf_img = Image.fromarray(ndarr)
        # w, h = img.size
        # half_size = (h//2, w//2)
        # hf_img = T.Resize(half_size, interpolation=Image.BICUBIC)(img)
        hf_img = transform(hf_img)
        img = transform(img)
        _format_q = f"Question: {question}?\nResponse:The answer is"
        return _format_q, answer, img, hf_img, indicator
    
    def low(self, img_path):
        image = cv2.imread(img_path)

        # 检查图像是否加载成功
        if image is None:
            print("图像加载失败，请检查图像路径")
        else:
            # 将图像从 BGR 转换为 RGB 以便展示
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 初始化 LL 子带
            LL_channels = []

            # 对每个颜色通道进行小波分解
            for i in range(3):  # 对 R, G, B 通道进行处理
                # 获取单个颜色通道
                channel = image_rgb[:, :, i]

                # 小波分解（两级）
                coeffs = pywt.wavedec2(channel, 'haar', level=2)  # 两级分解
                LL = coeffs[0]  # 获取 LL 子带

                # 对 LL 子带进行动态范围压缩，保留轮廓
                LL_scaled = LL / np.max(LL) * 255  # 将 LL 子带缩放到 0-255 范围内
                LL_channels.append(LL_scaled)

            # 将 LL 子带合并成彩色图像
            LL_image = cv2.merge(LL_channels)

            # 确保 LL 图像的像素值在合理范围内，并转换为 uint8
            return np.clip(LL_image, 0, 255).astype(np.uint8)

eval_set = EvalSet()
dataloader = DataLoader(dataset=eval_set, batch_size=16) #NOTE: 这里设置推理的批次大小
count = 0
correct = 0
flag = False
for qs, answsers, imgs, hf_imgs, indicators in dataloader:
    preds = llama.generate(qs, imgs, hf_imgs, indicators, 20, tokenizer)
    for idx, pred in enumerate(preds):
        count += 1
        if count == 999999:
            flag = True
        if pred == answsers[idx]:
            correct += 1
        print(f'pred: {pred:<20} gt: {answsers[idx]:<20} {correct:>5}/{count:<5} acc: {float(correct)/count * 100:2.2f}% {eval_set.__len__()}')
    if flag:
        break
    print(f'{correct}/{count}')