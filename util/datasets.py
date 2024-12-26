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

class GQA(Data.Dataset):
    @dataclass
    class Args:
        question_file = "/ai/teacher/dkc/Assets/GQA/jsons/balanced_trainset.json" #全量数据集 94W
        # question_file = "/ai/teacher/dkc/Assets/GQA/jsons/random_20000.json" # 随机2w条训练
        image_path = "/ai/teacher/dkc/Assets/GQA/images" # 用全局的dataset
        max_words = 512
        max_image_feats = 1
        model_path = "./data/weights" # 要加载tokenizer的

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
            
            w, h = image.size
            half_size = (h//2, w//2)
            half_image = transforms.Resize(half_size, interpolation=Image.BICUBIC)(image)
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



