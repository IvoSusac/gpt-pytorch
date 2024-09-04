import torch
from modules.model import MyGPTModel
from config import *
from get_gpt_weights import download_and_load_gpt2_weights, load_weights_to_model
import json
from utils import collate_batch, InstructionDataset
from functools import partial
from torch.utils.data import DataLoader
from train import calculate_epoch_loss, train_model
import os
import torch.nn as nn
from modules.lora import *

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def prepare_dataset(batch_size):
    with open("alpaca_data_10k.json", "r") as f:
        fine_tuning_data = json.load(f)

    tokenizer = tiktoken.get_encoding("gpt2") 

    train_split = int(len(fine_tuning_data) * 0.8)
    test_split = int(len(fine_tuning_data) * 0.1)
    val_split = len(fine_tuning_data) - train_split - test_split

    train_set = fine_tuning_data[:train_split]
    test_set = fine_tuning_data[train_split:train_split + test_split]
    val_set = fine_tuning_data[train_split + test_split:]

    device = torch.device("cuda")
    collate_on_device = partial(collate_batch, device=device)

    train_dataset = InstructionDataset(train_set, tokenizer)
    val_dataset = InstructionDataset(val_set, tokenizer)
    test_dataset = InstructionDataset(test_set, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_on_device, shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_on_device, shuffle=False, drop_last=False, num_workers=0)
    test_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_on_device, shuffle=False, drop_last=False, num_workers=0)

    return train_loader, val_loader, test_loader

   
if __name__ == "__main__":
    print("CUDA: ", torch.cuda.is_available())
    print("Num of GPUs: ", torch.cuda.device_count())

    train_loader, val_loader, test_loader = prepare_dataset(2)
    model = MyGPTModel(GPT_774_CONFIG)
    model.eval()

    device = torch.device("cuda")

    settings, params = download_and_load_gpt2_weights(model_name="774M", save_dir="gpt2")
    load_weights_to_model(model, params)
    # for more than 1 GPU:
    #model = nn.DataParallel(model)

    # for fine tuning with LoRA:
    #for param in model.parameters():
    #    param.requires_grad = False

    #replace_with_lora(model, rank=256, alpha=256)
    #total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #print("Trainable: ", total_params)
    
    model.to(device)
    
    # if we want to fine tune more comment out the loading of the weights and uncomment this
    #checkpoint = torch.load("model_and_optimizer_3.pt")
    #model.load_state_dict(checkpoint['model_state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.1)
    num_epochs = 5

    train_losses, val_losses, tokens_seen = train_model(model, train_loader, val_loader, optimizer, num_epochs, eval_freq=50, eval_iter=50, device=device)


