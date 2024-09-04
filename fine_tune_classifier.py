import urllib.request
import zipfile
import os
from pathlib import Path
import pandas as pd
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
from config import *
from utils import generate_text
from get_gpt_weights import download_and_load_gpt2_weights, load_weights_to_model
from modules.model import MyGPTModel

url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
zip_path = "sms_spam_collection.zip"
extracted_path = "sms_spam_collection"
data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)

        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]
        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            self.encoded_texts = [
                encoded_text[:self.max_length]
                for encoded_text in self.encoded_texts
            ]

        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )
    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length


def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path):
    if data_file_path.exists():
        print(f"{data_file_path} already exists. Skipping download and extraction.")
        df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])
        return df
    with urllib.request.urlopen(url) as response:
         with open(zip_path, "wb") as out_file:
            out_file.write(response.read())
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extracted_path)
    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)
    print(f"File downloaded and saved as {data_file_path}")
    df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])
    return df

def create_balanced_dataset(df):
    num_spam = df[df["Label"] == "spam"].shape[0]
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])
    return balanced_df

def random_split(df, train_frac, validation_frac):
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)

    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)
    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]
    return train_df, validation_df, test_df


def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    model.to(device)
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)

            with torch.no_grad():
                logits = model(input_batch)[:, -1, :]

            predicted_labels = torch.argmax(logits, dim=-1)
            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            break

    return correct_predictions / num_examples

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)[:, -1, :]
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss

def calc_loader_loss(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval() # disables components used only in training like dropout
    with torch.no_grad():
        train_loss = calc_loader_loss(train_loader, model, device, eval_iter)
        val_loss = calc_loader_loss(val_loader, model, device, eval_iter)
    model.train()
    return train_loss, val_loss



def train_model(model, train_loader, val_loader, optimizer, num_epochs, eval_freq, eval_iter, device):
    model.to(device)
    train_losses, val_losses, track_seen_tokens = [], [], []
    tokens_seen = 0
    global_step = -1
    #best_val_loss = 0.451
    best_val_loss = 1e9

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            global_step += 1
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
            tokens_seen += input_batch.numel()

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_seen_tokens.append(tokens_seen)
                print(f"Epoch: {epoch + 1}, Global step: {global_step}, Tokens seen: {tokens_seen}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),}, "classifier.pt")
                    print("model saved")

        train_accuracy = calc_accuracy_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_accuracy = calc_accuracy_loader(
            val_loader, model, device, num_batches=eval_iter
        )
        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")


    return train_losses, val_losses, track_seen_tokens

def classify_review(text, model, tokenizer, device, max_length=None, pad_token_id=50256):
    model.to(device)
    model.eval()
    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_embedding.weight.shape[1]
    input_ids = input_ids[:min(max_length, supported_context_length)]
    input_ids += [pad_token_id] * (max_length - len(input_ids))
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]
    predicted_label = torch.argmax(logits, dim=-1).item()

    return "spam" if predicted_label == 1 else "not spam" 

if __name__ == "__main__":
    df = download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)
    balanced_df = create_balanced_dataset(df)
    #print(balanced_df["Label"].value_counts())
    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})
    train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)
    train_df.to_csv("train.csv", index=None)
    validation_df.to_csv("validation.csv", index=None)
    test_df.to_csv("test.csv", index=None)

    tokenizer = tiktoken.get_encoding("gpt2")
    
    train_dataset = SpamDataset(csv_file="train.csv", max_length=None, tokenizer=tokenizer)
    val_dataset = SpamDataset(csv_file="validation.csv", max_length=None, tokenizer=tokenizer)
    test_dataset = SpamDataset(csv_file="test.csv", max_length=None, tokenizer=tokenizer)

    num_workers = 0
    batch_size = 8

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )

    
    model = MyGPTModel(GPT_774_CONFIG)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #settings, params = download_and_load_gpt2_weights(model_name="774M", save_dir="gpt2")

    #load_weights_to_model(model, params)
    model.to(device)
    
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    num_classes = 2
    model.output_head = torch.nn.Linear(in_features=GPT_774_CONFIG["embedding_dim"], out_features=num_classes)

    for param in model.trf_blocks[-1].parameters():
        param.requires_grad = True
    for param in model.ln.parameters():
        param.requires_grad = True

    checkpoint = torch.load("classifier.pt")
    model.load_state_dict(checkpoint['model_state_dict'])


    #train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=10)
    #val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=10)
    test_accuracy = calc_accuracy_loader(test_loader, model, device)

    #print(f"Training accuracy: {train_accuracy*100:.2f}%")
    #print(f"Validation accuracy: {val_accuracy*100:.2f}%")
    print(f"Test accuracy: {test_accuracy*100:.2f}%")


    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
    num_epochs = 5
    

    train_losses, val_losses, seen_tokens = train_model(model, train_loader, val_loader, optimizer, num_epochs, eval_freq=50, eval_iter=5, device=device)

   # print(classify_review("Click this link to obtain your prize!", model, tokenizer, max_length=train_dataset.max_length, device=device))

