import json, random
import torch.utils.data as Data
from torchvision.transforms import transforms
import os
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from PIL import Image
from util.base_prompt import *
import torch
from models import Tokenizer
import copy
from dataclasses import dataclass
import cv2
import pywt
import numpy as np

class All(Data.Dataset):
    @dataclass
    class Args:
        assets_path = "/ai/teacher/dkc/HZIP/Assets"
        question_file = ""
        image_path = ""
        model_path = "/ai/teacher/dkc/Assets/weights" # 要加载tokenizer的
        max_words = 512
        max_image_feats = 1

    def __init__(self, dataset_name: str, task: str, mem_type: str):
        super(All, self).__init__()
        self.dataset_names = ['GQA', 'OKVQA', 'TextVQA', 'VisWiz', 'VQAv2']
        assert dataset_name in self.dataset_names, f"dataset_name must be one of {self.dataset_names}"
        assert task in ['train', 'val'] , "task must be one of ['train', 'val']"
        assert mem_type in ['dual', 'single']
        
        img_path_map = {
            'GQA/train': '/ai/teacher/dkc/Assets/GQA/images',
            'GQA/val': '/ai/teacher/dkc/Assets/GQA/images',
            'OKVQA/train': '/ai/teacher/dkc/Assets/OKVQA/train2014',
            'OKVQA/val': '/ai/teacher/dkc/Assets/OKVQA/val2014',
            'TextVQA/train': '/ai/teacher/dkc/Assets/TextVQA/train_val',
            'TextVQA/val': '/ai/teacher/dkc/Assets/TextVQA/train_val',
            'VisWiz/train': '/ai/teacher/dkc/Assets/vizwiz/train',
            'VisWiz/val': '/ai/teacher/dkc/Assets/vizwiz/val',
            'VQAv2/train': '/ai/teacher/dkc/Assets/VQAv2/train2014',
            'VQAv2/val' : '/ai/teacher/dkc/Assets/VQAv2/val2014'
        }
        
        self.args = self.Args
        self.args.task = task
        self.dataset_name = dataset_name
        self.args.mem_type = mem_type 
        self.args.image_path = img_path_map[f'{dataset_name}/{task}']

        self.args.question_file = os.path.join(self.args.assets_path, dataset_name, f'{task}.json')

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
        if self.args.task == 'train':
            item = self.data[idx]
            question = item['question']
            imageId = item['image']
            answer = item['answer']

            prompt_question, prompt_answer = self.build_prompt(question, answer)
            example, labels, example_mask, label_mask = self.tokenize(prompt_question, prompt_answer)
            
            if imageId is not None:
                indicator = 1
                image_path = os.path.join(self.args.image_path, f"{imageId}.jpg")
                image = Image.open(image_path).convert('RGB')
                image = self.transforms(image)
                if self.args.mem_type == 'dual':
                    half_image_ndarr = self.low(image_path)
                    half_image = Image.fromarray(half_image_ndarr)
                    half_image = self.transforms(half_image)
                    return example, labels, example_mask, image, half_image, indicator
                else:
                    return example, labels, example_mask, image, indicator
            else:
                raise RuntimeError("Image is None")

        elif self.args.task == 'val':
            item = self.data[idx]
            question = item['question']
            imageId = item['image']
            answer = item['answer']

            formatted_question = f"Question: {question}?\nResponse:The answer is"
            
            if imageId is not None:
                indicator = 1
                image_path = os.path.join(self.args.image_path, f"{imageId}.jpg")
                image = Image.open(image_path).convert('RGB')
                image = self.transforms(image)
                if self.args.mem_type == 'dual':
                    half_image_ndarr = self.low(image_path)
                    half_image = Image.fromarray(half_image_ndarr)
                    half_image = self.transforms(half_image)
                    return formatted_question, answer, image, half_image, 1
                else:
                    return formatted_question, answer, image, 1
            else:
                raise RuntimeError("Image is None")

    def __len__(self):
        return len(self.data)

    def shuffle_list(self, list):
        random.shuffle(list)
    def low(self, img_path):
        image = cv2.imread(img_path)
        assert image is not None, "Image is None"
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        LL_channels = []
        for i in range(3):
            channel = image_rgb[:, :, i]
            coeffs = pywt.wavedec2(channel, 'haar', level=2)
            LL = coeffs[0]
            LL_scaled = LL / np.max(LL) * 255
            LL_channels.append(LL_scaled)
        LL_image = cv2.merge(LL_channels)
        return np.clip(LL_image, 0, 255).astype(np.uint8)
