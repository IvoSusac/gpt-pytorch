import os
import json
import urllib.request
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import torch
import numpy as np


def load_params_from_tf_ckpt(tf_ckpt_path, settings):
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    for name, _ in tf.train.list_variables(tf_ckpt_path):
        variable_array = np.squeeze(tf.train.load_variable(tf_ckpt_path, name))
        variable_name_parts = name.split("/")[1:] # skip the model name
    
        target = params
        if variable_name_parts[0].startswith("h"):
            layer_num = int(variable_name_parts[0][1:])
            target = params["blocks"][layer_num]
        
        for key in variable_name_parts[1:-1]:
            target = target.setdefault(key, {})
        
        target[variable_name_parts[-1]] = variable_array
    
    return params

def download_and_load_gpt2_weights(model_name, save_dir):
    if model_name not in ["124M", "355M", "774M", "1558M"]:
        raise ValueError("Invalid model name. Choose one of '124M', '355M', '762M', '1558M'.")

    model_dir = os.path.join(save_dir, model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    filenames = [
        "checkpoint", "encoder.json", "hparams.json", "model.ckpt.data-00000-of-00001",
        "model.ckpt.index", "model.ckpt.meta", "vocab.bpe"
    ]

    for filename in filenames:
        file_path = os.path.join(model_dir, filename)
        url_path = f"{url}/{model_name}/{filename}"
        with urllib.request.urlopen(url_path) as response:
            file_size = int(response.headers.get("Content-Length", 0))
            if os.path.exists(file_path) and os.path.getsize(file_path) == file_size:
                print(f"{file_path} already exists, skipping download.")
                continue
            
            block_size = 1024 # 1 KB
            with tqdm(total=file_size, unit="B", unit_scale=True, unit_divisor=1024) as pbar:
                with open(file_path, "wb") as f:
                    while True:
                        buffer = response.read(block_size)
                        if not buffer:
                            break
                        f.write(buffer)
                        pbar.update(len(buffer))
    
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    settings = json.load(open(os.path.join(model_dir, "hparams.json")))
    params = load_params_from_tf_ckpt(tf_ckpt_path, settings)
    print(params.keys())

    return settings, params

def assign_with_check(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shapes do not match: {left.shape} and {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))

def load_weights_to_model(gpt, params):
    gpt.embedding.weight = assign_with_check(gpt.embedding.weight, params["wte"])
    gpt.pos_embedding.weight = assign_with_check(gpt.pos_embedding.weight, params["wpe"])
    
    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(params["blocks"][b]["attn"]["c_attn"]["w"], 3, axis=-1)
        gpt.trf_blocks[b].attention.W_q.weight = assign_with_check(gpt.trf_blocks[b].attention.W_q.weight, q_w.T)
        gpt.trf_blocks[b].attention.W_k.weight = assign_with_check(gpt.trf_blocks[b].attention.W_k.weight, k_w.T)
        gpt.trf_blocks[b].attention.W_v.weight = assign_with_check(gpt.trf_blocks[b].attention.W_v.weight, v_w.T)

        q_b, k_b, v_b = np.split(params["blocks"][b]["attn"]["c_attn"]["b"], 3, axis=-1)
        gpt.trf_blocks[b].attention.W_q.bias = assign_with_check(gpt.trf_blocks[b].attention.W_q.bias, q_b)
        gpt.trf_blocks[b].attention.W_k.bias = assign_with_check(gpt.trf_blocks[b].attention.W_k.bias, k_b)
        gpt.trf_blocks[b].attention.W_v.bias = assign_with_check(gpt.trf_blocks[b].attention.W_v.bias, v_b)

        gpt.trf_blocks[b].attention.W_o.weight = assign_with_check(gpt.trf_blocks[b].attention.W_o.weight, params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].attention.W_o.bias = assign_with_check(gpt.trf_blocks[b].attention.W_o.bias, params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.trf_blocks[b].ff.layers[0].weight = assign_with_check(gpt.trf_blocks[b].ff.layers[0].weight, params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign_with_check(gpt.trf_blocks[b].ff.layers[0].bias, params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign_with_check(gpt.trf_blocks[b].ff.layers[2].weight, params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign_with_check(gpt.trf_blocks[b].ff.layers[2].bias, params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.trf_blocks[b].ln1.gamma = assign_with_check(gpt.trf_blocks[b].ln1.gamma, params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].ln1.beta = assign_with_check(gpt.trf_blocks[b].ln1.beta, params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].ln2.gamma = assign_with_check(gpt.trf_blocks[b].ln2.gamma, params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].ln2.beta = assign_with_check(gpt.trf_blocks[b].ln2.beta, params["blocks"][b]["ln_2"]["b"])

        gpt.ln.gamma = assign_with_check(gpt.ln.gamma, params["g"])
        gpt.ln.beta = assign_with_check(gpt.ln.beta, params["b"])

        gpt.output_head.weight = assign_with_check(gpt.output_head.weight, params["wte"])
