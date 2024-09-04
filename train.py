import torch


def calculate_batch_loss(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

def calculate_epoch_loss(dataloader, model, device, max_batches=None):
    total_loss = 0
    if len(dataloader) == 0:
        return float("nan")
    elif max_batches is None:
        max_batches = len(dataloader)
    else:
        max_batches = min(max_batches, len(dataloader))
    for i, (input_batch, target_batch) in enumerate(dataloader):
        if i < max_batches:
            loss = calculate_batch_loss(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / max_batches

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval() # disables components used only in training like dropout
    with torch.no_grad():
        train_loss = calculate_epoch_loss(train_loader, model, device, eval_iter)
        val_loss = calculate_epoch_loss(val_loader, model, device, eval_iter)
    model.train()
    return train_loss, val_loss


def train_model(model, train_loader, val_loader, optimizer, num_epochs, eval_freq, eval_iter, device):
    train_losses, val_losses, track_seen_tokens = [], [], []
    tokens_seen = 0
    global_step = -1
    #best_val_loss = 0.451
    best_val_loss = 0.267

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            global_step += 1
            optimizer.zero_grad()
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)
            logits = model(input_batch)
            loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
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
                    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),}, "model_and_optimizer_3.pt")
                    print("model saved")

    return train_losses, val_losses, track_seen_tokens
