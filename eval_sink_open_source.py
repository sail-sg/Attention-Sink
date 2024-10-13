import os
import time
import json
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer



def measure_attention_sink(model, tokenizer, prompts, score_path, token_length=50, device=torch.device("cuda")):
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    attention_scores_all_sample = []
    for prompt in tqdm(prompts):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        for key in inputs.keys():
            assert inputs[key].shape[1] >= token_length
            inputs[key] = inputs[key][:, :token_length]
        outputs = model.generate(
            **inputs,
            output_attentions=True,
            return_dict_in_generate=True,
            max_new_tokens=1
        )
        
        attentions = outputs['attentions']
        assert len(attentions) == 1
        attention_scores_all_layer = []
        for l in range(num_layers):
            attentions_layer = attentions[0][l]
            attention_scores_all_layer.append(attentions_layer)
        attention_scores_all_layer = torch.cat(attention_scores_all_layer, dim=0)
        attention_scores_all_sample.append(attention_scores_all_layer.unsqueeze(dim=0))
    attention_scores_all_sample = torch.cat(attention_scores_all_sample, dim=0)  # (num_samples, num_layers, num_heads, num_tokens, num_tokens)
    np.save(score_path, attention_scores_all_sample.cpu().numpy())


def compute_attention_sink(score_path, epsilon):
    attention_scores = np.load(score_path)
    num_samples, num_layers, num_heads, num_tokens1, num_tokens2 = attention_scores.shape
    assert num_tokens1 == num_tokens2
    attention_scores = torch.from_numpy(attention_scores)
    ratios = torch.arange(num_tokens1, 0, -1)[None, None, None, :].expand(num_samples, num_layers, num_heads, num_tokens1, num_tokens2).to(attention_scores)
    importance_scores = (attention_scores / ratios).sum(dim=-2) # (num_samples, num_layers, num_heads, num_tokens)
    metric1 = (importance_scores > epsilon).to(torch.float).mean(dim=(0,1,2))
    return metric1 * 100


def measure_open_sourced_lms():
    # load model family
    device = torch.device("cuda")
    os.makedirs("results", exist_ok=True)
    ########################################
    gpt_family = ["openai-community/gpt2", "openai-community/gpt2-medium", "openai-community/gpt2-large", "openai-community/gpt2-xl"] 
    llama2_family = ["meta-llama/Llama-2-7b-hf",  "meta-llama/Llama-2-13b-hf", "meta-llama/Llama-2-7b-chat-hf",  "meta-llama/Llama-2-13b-chat-hf"]
    llama3_family = ["meta-llama/Meta-Llama-3-8B", "meta-llama/Meta-Llama-3.1-8B", "meta-llama/Meta-Llama-3-8B-Instruct", "meta-llama/Meta-Llama-3.1-8B-Instruct"]
    pythia_family = [f"EleutherAI/pythia-{size}" for size in ["14m", "31m", "70m", "160m", "410m", "1b", "1.4b", "2.8b", "6.9b", "12b"]] 
    opt_family = [f"facebook/opt-{size}" for size in ["125m", "350m", "1.3b", "2.7b", "6.7b", "13b"]] 
    mistral_family = [f"mistralai/Mistral-7B-v0.1", f"mistralai/Mistral-7B-Instruct-v0.1"]
    model_pool = gpt_family + llama2_family + llama3_family + pythia_family + opt_family + mistral_family
    ########################################
    for model_path in tqdm(model_pool):
        model_name = model_path.split("/")[-1]
        os.makedirs(f"results/{model_name}", exist_ok=True)

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            attn_implementation="eager",
            # torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_path
        )
        #########################################
        
        #########################################

        # load data and feed them into LLMs
        file_path = "datasets/probe_valid_natural.jsonl"
        token_length = 64
            
        score_path = f"results/{model_name}/token{token_length}.npy"
        with open(file_path, 'r') as f:
            prompts = [json.loads(line)["text"] for line in f]
        measure_attention_sink(model, tokenizer, prompts, score_path, token_length, device)

        metric1 = compute_attention_sink(score_path, epsilon=0.3)
        print(f"Load model checkpoints from {model_path}.")
        print(metric1)


        

if __name__ == "__main__":
    measure_open_sourced_lms()