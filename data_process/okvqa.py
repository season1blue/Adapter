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
from tqdm import tqdm
import cv2
import pywt
import numpy as np

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

class OKVQA(Data.Dataset):
    @dataclass
    class Args:
        _type = 'train'
        question_file = f"/ai/teacher/dkc/Assets/OKVQA/processed_datasets/okvqa_{_type}.json" # OKVQA trian: 9k, eval 5k
        image_path = f"/ai/teacher/dkc/Assets/OKVQA/{_type}2014" # 用全局的dataset
        model_path = "/ai/teacher/dkc/Assets/weights" # 要加载tokenizer的
        max_words = 512
        max_image_feats = 1

    def __init__(self):
        super(OKVQA, self).__init__()
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
        imageId = f'COCO_{self.args._type}2014_' + str(item['image_id']).zfill(12)
        answer = item['answer']
        prompt_question, prompt_answer = self.build_prompt(question, answer)
        
        if imageId is not None:
            image_path = os.path.join(self.args.image_path, f"{imageId}.jpg")
            image = Image.open(image_path).convert('RGB')
            image = self.transforms(image)
            indicator = 1
        else:
            image = torch.Tensor(torch.zeros(3, 224, 224).float())
            indicator = 0

        example, labels, example_mask, label_mask = self.tokenize(prompt_question, prompt_answer)
        return example, labels, example_mask, image, indicator

    def __len__(self):
        return len(self.data)

    def shuffle_list(self, list):
        random.shuffle(list)