"""
This script is used to directly load model to compute attention sink
"""

import glob
import math
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Union
import math
import lightning as L
import torch
import torch.nn as nn
from lightning.fabric.strategies import FSDPStrategy, XLAStrategy
from torch.utils.data import DataLoader
from functools import partial
# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))
# from apex.optimizers import FusedAdam #torch optimizer has a cuda backend, which is faster actually
from lit_gpt.inference.model_infer import GPT, Config
from lit_gpt.inference.model_postln_infer import GPTpostln
from lit_gpt.inference.model_kv_bias_infer import GPTkvbias
from lit_gpt.inference.model_pos_infer import GPTpos
from lit_gpt.inference.model_relpos_infer import GPTrelpos
from lit_gpt.inference.model_attn_sim_infer import GPTattnsim
from lit_gpt.inference.model_attn_kernel_infer import GPTattnkernel
from lit_gpt.inference.model_prefix_infer import GPTprefix
from lit_gpt.inference.model_window_infer import GPTwindow
from lit_gpt.packed_dataset import CombinedDataset, PackedDataset
from lit_gpt.speed_monitor import SpeedMonitorFabric as Monitor
from lit_gpt.speed_monitor import estimate_flops, measure_flops
from lit_gpt.utils import chunked_cross_entropy, get_default_supported_precision, num_parameters, step_csv_logger, lazy_load
from pytorch_lightning.loggers import WandbLogger
from lit_gpt import FusedCrossEntropyLoss
import random
import yaml
import os
import json
import numpy as np
from lit_gpt import Tokenizer
import argparse


def load_data():
    tokenizer_path = Path("./preprocess/tokenizer/gptneox")
    tokenizer = Tokenizer(tokenizer_path)
    

    with open("datasets/probe_valid.jsonl", 'r') as f:
        prompts = [json.loads(line)["text"] for line in f]
     

    print(f"Tokenizer: BOS token {tokenizer.bos_id}, EOS token {tokenizer.eos_id}, vocabulary size {tokenizer.vocab_size}")
    print(len(prompts))
    print(prompts[0])
    # print(tokenizer.encode(prompts[0], bos=True, eos=False))
    
    all_inputs = []
    for prompt in prompts:
        all_inputs.append(tokenizer.encode(prompt, eos=False).unsqueeze(dim=0).to(torch.long))
    return all_inputs



def evaluate_model(model_name, mode, load_from, fixed_position=0):
    device = torch.device('cuda')
    all_inputs = load_data()
    config = Config.from_name(model_name)
    # define the model
    if mode == "prefix":
        model = GPTprefix(config)
    elif mode == "window":
        model = GPTwindow(config)
    elif mode == "pos":
        model = GPTpos(config)
    elif mode == "relpos":
        model = GPTrelpos(config)
    elif mode == "postln":
        model = GPTpostln(config)
    elif mode == "kv_bias":
        model = GPTkvbias(config)
    elif mode == "attnsim":
        model = GPTattnsim(config)
    elif mode == "attnkernel":
        model = GPTattnkernel(config)
    else:
        model = GPT(config)
    # model.apply(partial(model._init_weights, n_layer=model_config.n_layer))
    if load_from is not None:
        # use torch.load to load the model
        print("loading model from {}".format(load_from))
        state_dict = torch.load(load_from, map_location=device)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        model.load_state_dict(state_dict, strict=True, assign=True)
    

    model.to(torch.bfloat16)
    token_length = 64
    size = 100
    count = 0
    num_layers = config.n_layer
    loss_func = FusedCrossEntropyLoss()
    attention_scores_all_sample = []
    hidden_states_all_sample = []
    all_loss = []
    for data in all_inputs:
        inputs = data[:, :token_length].to(device).contiguous()
        if mode == "fix":
            inputs[:, fixed_position] = 1  # pad token
        labels = data[:, 1:token_length+1].to(device).contiguous()
        outputs, all_attns, all_hiddens = model(
            inputs,
            output_attention=True,
        )
        # print(all_attns)
        
        loss = loss_func(outputs, labels)
        all_loss.append(loss.item())
        # print(f"loss: {loss.item()}")
        attention_scores_all_layer = []
        for l in range(num_layers):
            attentions_layer = all_attns[l] #.cpu()
            attention_scores_all_layer.append(attentions_layer)
            # print(attentions_layer.mean(dim=(0,1)))
        # break
        attention_scores_all_layer = torch.cat(attention_scores_all_layer, dim=0)
        attention_scores_all_sample.append(attention_scores_all_layer.unsqueeze(dim=0))

        count += data.shape[0]
        if count >= size:
            break
        
        hidden_states_all_layer = torch.cat(all_hiddens, dim=0)
        hidden_states_all_sample.append(hidden_states_all_layer.unsqueeze(dim=0))
    print(f"averaged loss: {sum(all_loss) / size}")
    attention_scores_all_sample = torch.cat(attention_scores_all_sample, dim=0)  # (num_samples, num_layers, num_heads, num_tokens)
    # print(attention_scores_all_sample.mean(dim=(0,2,3)))
    
    # data_list = np.concatenate(data_list, axis=0)
    attention_scores = attention_scores_all_sample.detach().cpu() #torch.from_numpy(attention_scores)
    num_samples, num_layers, num_heads, num_tokens1, num_tokens2 = attention_scores.shape
    ratios = torch.arange(num_tokens1, 0, -1)[None, None, None, :].expand(num_samples, num_layers, num_heads, num_tokens1, num_tokens2).to(attention_scores)

    epsilon = 0.3
    importance_scores = (attention_scores / ratios).sum(dim=-2)  # (num_samples, num_layers, num_heads, num_tokens)
    metric1 = (importance_scores > epsilon).to(torch.float).mean(dim=(1,2))

    print(metric1[:, :10].mean(dim=0) * 100)

    # norm
    hidden_states_all_sample = torch.cat(hidden_states_all_sample, dim=0)  # sample, layer, seqence, dim
    hidden_states_all_sample_norm = hidden_states_all_sample.norm(p=2, dim=-1)
    for l in range(num_layers+1):
        print(f"layer: {l}, norm for each token: {hidden_states_all_sample_norm[:, l].mean(dim=0)}") # \pm {hidden_states_all_sample_norm[:, l].std(dim=0)}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--mode", type=str, default=None)
    parser.add_argument("--load_from", type=str, default=None)
    parser.add_argument("--fix_position", type=int, default=0)
    args = parser.parse_args()
    evaluate_model(model_name=args.model_name, mode=args.mode, load_from=args.load_from, fixed_position=args.fix_position)