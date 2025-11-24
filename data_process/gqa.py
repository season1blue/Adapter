import json, random
import torch.utils.data as Data
from torchvision.transforms import transforms
import os
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from PIL import Image
from util.base_prompt import *
import torch
from memvp import Tokenizer
import copy
from dataclasses import dataclass
import cv2
import pywt
import numpy as np

class GQA(Data.Dataset):
    @dataclass
    class Args:
        question_file = "/ai/teacher/dkc/Assets/GQA/jsons/balanced_trainset.json" #全量数据集 94W
        image_path = "/ai/teacher/dkc/Assets/GQA/images" # 用全局的dataset
        model_path = "/ai/teacher/dkc/Assets/weights" # 要加载tokenizer的
        max_words = 512
        max_image_feats = 1

    def __init__(self):
        super(GQA, self).__init__()
        self.args = self.Args
        self.tokenizer = Tokenizer(model_path=self.args.model_path + '/tokenizer.model')
        self.max_words = self.args.max_words
        self.max_image_feats = self.args.max_image_feats

        self.image_path = self.args.image_path
        self.data = json.load(open(self.args.question_file))
        print(f"number of problems:  {len(self.data)}\n")

        self.transforms = transforms.Compose(
            [transforms.Resize((224, 224), interpolation=Image.BICUBIC), 
             transforms.ToTensor(),
             transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])

    def tokenize(self, prompt, answer):
        example = prompt + answer
        prompt = torch.tensor(self.tokenizer.encode(prompt, bos=True, eos=False), dtype=torch.int64)
        example = torch.tensor(self.tokenizer.encode(example, bos=True, eos=True), dtype=torch.int64)
        padding = self.max_words - example.shape[0]
        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            example = example[:self.max_words]
        labels = copy.deepcopy(example)
        labels[:len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = 0
        example_mask = example_mask.float()
        label_mask = label_mask.float()
        return example, labels, example_mask, label_mask

    def build_prompt(self, question, answer):
        _format_question = f'Question: {question}\nResponse:'
        _format_answer = f"The answer is {answer}"
        return _format_question, _format_answer

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item['question']
        imageId = item['imageId']
        answer = item['answer']

        prompt_question, prompt_answer = self.build_prompt(question, answer)
        
        if imageId is not None:
            image_path = os.path.join(self.args.image_path, f"{imageId}.jpg")
            image = Image.open(image_path).convert('RGB')
            half_image_ndarr = self.low(image_path)
            half_image = Image.fromarray(half_image_ndarr)
            half_image = self.transforms(half_image)
            image = self.transforms(image)
            indicator = 1
        else:
            image = torch.Tensor(torch.zeros(3, 224, 224).float())
            half_image = torch.Tensor(torch.zeros(3, 224, 224).float())
            indicator = 0

        example, labels, example_mask, label_mask = self.tokenize(prompt_question, prompt_answer)
        return example, labels, example_mask, image, half_image, indicator

    def __len__(self):
        return len(self.data)

    def shuffle_list(self, list):
        random.shuffle(list)

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



