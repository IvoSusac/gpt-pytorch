import torch
import tiktoken
from model import MyGPTModel
from config import *
from get_gpt_weights import download_and_load_gpt2_weights, load_weights_to_model
import json
from utils import format_input, collate_batch, InstructionDataset
from functools import partial
from torch.utils.data import DataLoader
from utils import generate_text_v2

if __name__ == "__main__":
    tokenizer = tiktoken.get_encoding("gpt2")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MyGPTModel(GPT_774_CONFIG)

    model.to(device)

    model.eval() # disables components used only in training like dropout

    checkpoint = torch.load("model_and_optimizer_3.pt")

    model.load_state_dict(checkpoint['model_state_dict'])

    with open("alpaca_data_10k.json", "r") as f:
        fine_tuning_data = json.load(f)

    train_split = int(len(fine_tuning_data) * 0.8)
    test_split = int(len(fine_tuning_data) * 0.1)
    val_split = len(fine_tuning_data) - train_split - test_split

    train_set = fine_tuning_data[:train_split]
    test_set = fine_tuning_data[train_split:train_split + test_split]
    val_set = fine_tuning_data[train_split + test_split:]

    collate_on_device = partial(collate_batch, device=device)

    num_workers = 0
    batch_size = 2

    train_dataset = InstructionDataset(train_set, tokenizer)
    val_dataset = InstructionDataset(val_set, tokenizer)
    test_dataset = InstructionDataset(test_set, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_on_device, shuffle=True, drop_last=True, num_workers=num_workers)
    val_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_on_device, shuffle=False, drop_last=False, num_workers=num_workers)
    test_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_on_device, shuffle=False, drop_last=False, num_workers=num_workers)


    # write the first 200 responses in a json
    model_responses = []

    for entry in test_set[0:200]:
        input_text = format_input(entry)
        input_ids = tokenizer.encode(input_text, allowed_special={"<|endoftext|>"})
        input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)
        token_ids = generate_text_v2(model, input_ids, max_generated_tokens=256, context_size=GPT_774_CONFIG["context_size"], eos_token_id=50256)
        decoded_output = tokenizer.decode(token_ids.squeeze(0).tolist())
        response = decoded_output[len(input_text):].replace("### Response:", "").strip()

        model_responses.append({
         "instruction": entry['instruction'],
         "input": entry["input"],
         "output": entry["output"],
         "model response": response.strip()
        })

    with open('model_responses_2.json', 'w') as json_file:
        json.dump(model_responses, json_file, indent=4)




