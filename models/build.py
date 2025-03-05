import torch

import json
# from memvp import ModelArgs, Tokenizer, Transformer
# from memvp.adapter import set_Llama_Adapter, set_Clip_Adapter

from models import Tokenizer


from pathlib import Path
import importlib
import logging

model_names = ['0_ours', '1_zipped', '2_full', '3_hierachical', '4_origin']

class_mapper = {
    '0_ours': 'models.model_0_ours',
    '1_zipped': 'models.model_1_zipped',
    '2_full': 'models.model_2_full',
    '3_hierachical': 'models.model_3_hierachical',
    '4_origin': 'models.model_4_origin',
}

mem_type_mapper = {
    'single': 'models.adapter_single',
    'dual': 'models.adapter_dual'
}

def _load_and_redistribute_checkpoint(llama_model_path, model_name):
    with open(Path(llama_model_path) / model_name / 'params.json') as f:
        params = json.load(f)
    tokenizer = Tokenizer(model_path=str(Path(llama_model_path) / 'tokenizer.model'))
    print('Using model path: %s, model_name: %s' % (llama_model_path, model_name))
    if model_name == '7B':
        checkpoint = torch.load(llama_model_path + model_name + '/consolidated.00.pth', map_location="cpu")
        return checkpoint, tokenizer, params

    checkpoints = (Path(llama_model_path) / model_name).glob('*.pth')
    checkpoints = sorted(checkpoints)

    loaded = []
    for x in checkpoints:
        print('loading from', x)
        loaded.append(torch.load(x, map_location='cpu'))

    full_state_dict = {}
    split_dims = {}

    def add_weight_with_split_dim(name, dim):
        if dim < 0:  # bcast without split
            full_state_dict[name] = loaded[0][name].clone()
        else:
            full_state_dict[name] = torch.cat([x[name] for x in loaded], dim=dim)
        for x in loaded:
            del x[name]
        split_dims[name] = dim

    add_weight_with_split_dim('tok_embeddings.weight', 1)
    add_weight_with_split_dim('norm.weight', -1)
    add_weight_with_split_dim('output.weight', 0)
    for i in range(params['n_layers']):
        # print('gathering layer %d of %d' % (i, params['n_layers']))
        layer_prefix = f'layers.{i}.'
        bcast_names = [
            'attention_norm.weight',
            'ffn_norm.weight',
        ]
        column_parallel_names = [
            'attention.wq.weight',
            'attention.wk.weight',
            'attention.wv.weight',
            'feed_forward.w1.weight',
            'feed_forward.w3.weight',
        ]
        row_parallel_names = [
            'attention.wo.weight',
            'feed_forward.w2.weight',
        ]
        for key in bcast_names:
            add_weight_with_split_dim(layer_prefix + key, -1)
        for key in column_parallel_names:
            add_weight_with_split_dim(layer_prefix + key, 0)
        for key in row_parallel_names:
            add_weight_with_split_dim(layer_prefix + key, 1)

    checkpoint = full_state_dict

    return checkpoint, tokenizer, params


def create_model(args):
    logging.info(f'Atempt to create model: {args.model_name}')
    assert args.model_name in model_names, f'model_name must be in {model_names}'
    model_lib = importlib.import_module(class_mapper[args.model_name])
    logging.info(f'{args.model_name} imported from {class_mapper[args.model_name]}')
    
    ModelArgs = model_lib.ModelArgs
    Transformer = model_lib.Transformer
    llama_model_path = args.llama_model_path
    model_name = args.llm_model


    checkpoint, tokenizer, params = _load_and_redistribute_checkpoint(llama_model_path, model_name)

    model_args = ModelArgs(
        max_seq_len=args.max_seq_len, max_batch_size=64, hidden_proj=args.hidden_proj,
        emb=args.emb, is_train=args.is_train, **params
    )

    model_args.vocab_size = tokenizer.n_words

    if args.cpu_load:
        # cpu load is slow, but is freindly for GPU with limited memory.
        torch.set_default_tensor_type(torch.HalfTensor)
    else:
        torch.set_default_tensor_type(torch.cuda.HalfTensor)

    llama = Transformer(model_args)

    # delete language encoder
    del llama.backbone.transformer

    torch.set_default_tensor_type(torch.FloatTensor)

    llama.load_state_dict(checkpoint, strict=False)

    adapter_module = importlib.import_module(mem_type_mapper[args.mem_type])
    
    adapter_module.set_Llama_Adapter(llama, model_lib.TransformerBlock, s=args.adapter_scale, gradient_checkpointing=args.gradient_checkpointing)
    adapter_module.set_Clip_Adapter(llama.backbone.visual, dim=args.adapter_dim, s=0.1) # 对视觉编码器增加了adapter

    learnable_keys = ['adapter']
    total = 0.
    trainable_names = []
    print('-----------------------------All Keys-----------------------------')
    for name, param in llama.named_parameters():
        print(f'name: {name}')
        for key in learnable_keys:
            if key in name:
                param.requires_grad = True
                param.data = param.data.float()
                total += param.nelement()
                trainable_names.append((name, param.nelement()))
            else:
                param.requires_grad = False
    print('-----------------------------Learnable Keys-----------------------------')
    for i in trainable_names:
        print(i)
    # print(trainable_names)
    print('  + Number of trainable params: %.2fM' % (total / 1e6))
    return llama