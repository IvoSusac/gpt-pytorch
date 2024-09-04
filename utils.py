import torch
import torch.nn as nn

from torch.utils.data import Dataset

class InstructionDataset(Dataset):
  def __init__(self, data, tokenizer):
    self.data = data
    self.encoded_texts = []
    for entry in data:
      instruction_format = format_input(entry)
      response = f"\n\n### Response:\n{entry['output']}"
      instruction_response_paired = instruction_format + response
      self.encoded_texts.append(tokenizer.encode(instruction_response_paired))

  def __getitem__(self, idx):
    return self.encoded_texts[idx]

  def __len__(self):
    return len(self.data)

class LayerNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-6):
        super().__init__()
        # the model can learn to scale or shift the normalized values by using the gamma and beta parameters if it needs to (if it improves the performance)
        self.gamma = nn.Parameter(torch.ones(emb_dim))
        self.beta = nn.Parameter(torch.zeros(emb_dim))
        self.eps = eps # small constant added to the variance to prevent division by zero
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False) # we use the biased variance (dividing by n instead of n - 1) because it's compatible to the pretrained weights we will load
        # for large embedding dimensions, the difference between n and n - 1 is negligible
        normalized = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * normalized + self.beta
    
# gelu activation function - a smooth approximation of the relu activation function
# computationally efficient approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
# advantages over relu: differentiable everywhere, non-zero gradients for negative values so less neurons are inactive during training
class GELU (nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * x**3)))

def format_input(entry):
  instruction_txt = (
      f"Below is an instruction that describes a requested task. "
      f"Write a response that appropriately completes the request."
      f"\n\n### Instruction:\n{entry['instruction']}"
  )
  input = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
  return instruction_txt + input

def collate_batch(batch, padding_token_id=50256, allowed_max_length=1024, ignore_index=-100, device="cpu"):
  longest_sequence_len = max(len(item) for item in batch)
  inputs_list, targets_list = [], []

  for item in batch:
    new_item = item.copy()

    padded_item = new_item + [padding_token_id] * (longest_sequence_len - len(new_item))

    inputs = torch.tensor(padded_item)
    targets = torch.tensor(padded_item[1:] + [padding_token_id])

    mask = targets == padding_token_id
    indices = torch.nonzero(mask).squeeze()
    if indices.numel() >= 2:
      targets[indices[1:]] = ignore_index

    if allowed_max_length:
      inputs = inputs[:allowed_max_length]
      targets = targets[:allowed_max_length]

    inputs_list.append(inputs)
    targets_list.append(targets)

  inputs_tensor = torch.stack(inputs_list).to(device)
  targets_tensor = torch.stack(targets_list).to(device)
  return inputs_tensor, targets_tensor

# generate text without temperature or top k sampling
def generate_text(model, input_ids, max_generated_tokens, context_size):
    for _ in range(max_generated_tokens):
        input_ids_cond = input_ids[:, -context_size:] # crop the input_ids to the context size, only keep the last context_size tokens
        with torch.no_grad():
            logits = model(input_ids_cond)
        
        logits = logits[:, -1, :] # we only care about the logits for the last token as it represents the token that needs to be generated
        probs = torch.softmax(logits, dim=-1) # convert the logits to probabilities (not really needed because softmax is a monotonic function but whatever)
        next_token_id = torch.argmax(probs, dim=-1, keepdim=True)
        input_ids = torch.cat((input_ids, next_token_id), dim=1)
    
    return input_ids

# generate text with temperature and top k sampling
def generate_text_v2(model, input_ids, max_generated_tokens, context_size, temperature=1.0, top_k=None, eos_token_id=None):
    for _ in range(max_generated_tokens):
        input_ids_cond = input_ids[:, -context_size:] # crop the input_ids to the context size, only keep the last context_size tokens
        with torch.no_grad():
            logits = model(input_ids_cond)
        
        logits = logits[:, -1, :] # we only care about the logits for the last token as it represents the token that needs to be generated
        if top_k is not None:
            top_logits, top_indices = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits[logits < min_val] = -float("inf")
        
        if temperature > 0.0:
            logits /= temperature
            probs = torch.softmax(logits, dim=-1) # convert the logits to probabilities (not really needed because softmax is a monotonic function but whatever)
            next_token_id = torch.multinomial(probs, num_samples=1)
        else:
            next_token_id = torch.argmax(logits, dim=-1, keepdim=True)
        if eos_token_id is not None and next_token_id == eos_token_id:
            break
        input_ids = torch.cat((input_ids, next_token_id), dim=1)
    
    return input_ids
