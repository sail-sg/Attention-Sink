import os
import time
import json
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def measure_activations(model, tokenizer, prompts, states_path, keys_path, values_path, token_length=50, device=torch.device("cuda")):
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    hidden_states_all_sample = []
    values_all_sample = []
    keys_all_sample = []
    for prompt in tqdm(prompts):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        for key in inputs.keys():
            assert inputs[key].shape[1] >= token_length
            inputs[key] = inputs[key][:, :token_length]

        outputs = model.generate(
            **inputs,
            output_attentions=True,
            output_hidden_states=True,
            use_cache=True,
            return_dict_in_generate=True,
            max_new_tokens=1
        )
        
        attentions = outputs['attentions']
        hidden_states = outputs["hidden_states"]
        past_key_values = outputs["past_key_values"]
        assert len(attentions) == 1
        assert len(past_key_values) == num_layers and len(past_key_values[0]) == 2
        values_all_layer = []
        keys_all_layer = []
        for l in range(num_layers):
            # keys
            keys_all_layer.append(past_key_values[l][0])
            # values
            values_all_layer.append(past_key_values[l][1])
        
        # keys
        keys_all_layer = torch.cat(keys_all_layer, dim=0)
        keys_all_sample.append(keys_all_layer.unsqueeze(dim=0))     
        # values
        values_all_layer = torch.cat(values_all_layer, dim=0)
        values_all_sample.append(values_all_layer.unsqueeze(dim=0)) 
        
        assert len(hidden_states) == 1
        hidden_states_all_layer = []
        for l in range(num_layers+1):
            hidden_states_layer = hidden_states[0][l]  # (num_samples, num_tokens, hidden_dim)
            hidden_states_all_layer.append(hidden_states_layer)
        hidden_states_all_layer = torch.cat(hidden_states_all_layer, dim=0)
        hidden_states_all_sample.append(hidden_states_all_layer.unsqueeze(dim=0))

    # attention_scores_all_sample = torch.cat(attention_scores_all_sample, dim=0)  # (num_samples, num_layers, num_heads, num_tokens)
    hidden_states_all_sample = torch.cat(hidden_states_all_sample, dim=0)  # (num_samples, num_layers, num_tokens, hidden_dim)
    keys_all_sample = torch.cat(keys_all_sample, dim=0)
    values_all_sample = torch.cat(values_all_sample, dim=0)
    # np.save(score_path, attention_scores_all_sample.cpu().numpy())
    np.save(states_path, hidden_states_all_sample.cpu().numpy())
    np.save(values_path, values_all_sample.cpu().numpy())
    np.save(keys_path, keys_all_sample.cpu().numpy())


def compute_norm(states_path, device):
    hidden_states = np.load(states_path)
    num_samples, num_layers, num_tokens, dim = hidden_states.shape
    hidden_states = torch.from_numpy(hidden_states).to(device)
    split_size = 5
    hidden_states_split = torch.split(hidden_states, split_size)
    all_norms = []
    for hidden_states in hidden_states_split:
        norm = hidden_states.norm(p=2, dim=-1)
        all_norms.append(norm)
    all_norms = torch.cat(all_norms, dim=0)
    return all_norms.mean(dim=0)  # (num_samples, num_layers, num_tokens) -> (num_layers, num_tokens)


def compute_kv_norm(states_path, device):
    hidden_states = np.load(states_path)
    num_samples, num_layers, num_heads, num_tokens, dim = hidden_states.shape
    hidden_states = torch.from_numpy(hidden_states).to(device)
    split_size = 5
    hidden_states_split = torch.split(hidden_states, split_size)
    all_norms = []
    for hidden_states in hidden_states_split:
        norm = hidden_states.norm(p=2, dim=-1)
        all_norms.append(norm)
    all_norms = torch.cat(all_norms, dim=0)
    return all_norms.mean(dim=(0, 2))  # (num_samples, num_layers, num_heads, num_tokens) -> (num_layers, num_tokens)


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
            
        states_path = f"results/{model_name}/states_token{token_length}.npy"
        values_path = f"results/{model_name}/values_token{token_length}.npy"
        keys_path = f"results/{model_name}/keys_token{token_length}.npy"
        with open(file_path, 'r') as f:
            prompts = [json.loads(line)["text"] for line in f]
        measure_activations(model, tokenizer, prompts, states_path, keys_path, values_path, token_length, device)

        
        # analysis norm
        all_norms = compute_norm(states_path, device)
        print(f"model name: {model_name}, numerical norm: {all_norms}.")  # (layer, token)
        
        all_k_norms = compute_kv_norm(keys_path, device)
        print(f"model name: {model_name}, numerical norm for keys: {all_k_norms}.")  # (layer, token)

        all_v_norms = compute_kv_norm(values_path, device)
        print(f"model name: {model_name}, numerical norm for values: {all_v_norms}.")  # (layer, token)
        

if __name__ == "__main__":
    measure_open_sourced_lms()