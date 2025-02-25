import os
import torch
import torch.nn as nn
import torch.optim as optim
from tokenizers import ByteLevelBPETokenizer
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.gpt1 import GPT1


class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, seq_length=128):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        with open(self.file_path, "r", encoding="utf-8") as f:
            text = f.read()
        self.tokens = self.tokenizer.encode(text).ids  # .ids로 토큰화된 리스트를 얻음

    def __len__(self):
        return (len(self.tokens) - self.seq_length) // self.seq_length

    def __getitem__(self, idx):
        start_idx = idx * self.seq_length
        end_idx = start_idx + self.seq_length

        tokens_chunk = self.tokens[start_idx:end_idx]

        input_seq = torch.tensor(tokens_chunk[:-1], dtype=torch.long)
        target_seq = torch.tensor(tokens_chunk[1:], dtype=torch.long)
        return input_seq, target_seq


def main():
    checkpoint_dir = "models/saved"
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_file = "data/raw/train.txt"
    val_file = "data/raw/valid.txt"
    test_file = "data/raw/test.txt"

    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training data file not found: {train_file}")
    if not os.path.exists(val_file):
        raise FileNotFoundError(f"Validation data file not found: {val_file}")
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test data file not found: {test_file}")

    # Initialize the tokenizer
    with open(train_file, "r", encoding="utf-8") as f:
        data_iterator = iter(f)

        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train_from_iterator(
            data_iterator,
            vocab_size=50257,  # GPT-1 typically uses a vocab size of 50,257
            min_frequency=2,
            special_tokens=["<unk>", "<s>", "</s>", "<pad>", "<mask>"]
        )
        tokenizer.save(os.path.join(checkpoint_dir, "tokenizer.json"))

    # Load the datasets
    train_dataset = TextDataset(train_file, tokenizer, seq_length=128)
    val_dataset = TextDataset(val_file, tokenizer, seq_length=128)
    test_dataset = TextDataset(test_file, tokenizer, seq_length=128)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Model configuration for GPT-1
    vocab_size = 50257  # GPT-1 uses 50,257 tokens
    embed_dim = 768  # GPT-1 uses 768 for the embedding size
    num_layers = 12  # GPT-1 has 12 layers
    num_heads = 12  # GPT-1 has 12 attention heads
    model = GPT1(vocab_size, embed_dim, num_layers, num_heads, dropout=0.1, max_seq_length=1024)  # Max seq length is 1024
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Resume from checkpoint if it exists
    start_epoch = 0

    # Resume from the most recent checkpoint in the checkpoint directory
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]

    if checkpoint_files:
        # Sort checkpoint files by epoch (assuming filenames are in the format 'model_epoch_#.pth')
        checkpoint_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]), reverse=True)
        latest_checkpoint = os.path.join(checkpoint_dir, checkpoint_files[0])

        checkpoint = torch.load(latest_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resumed training from epoch {start_epoch + 1}")
    else:
        start_epoch = 0
        print("No checkpoint found, starting training from scratch.")

    writer = SummaryWriter(log_dir='runs/gpt1_experiment')

    # Add the model graph to TensorBoard
    dummy_input = torch.randint(0, vocab_size, (1, 128)).to(device)
    writer.add_graph(model, dummy_input)

    # Early stopping parameters
    num_epochs = 5
    patience = 3
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    model.train()
    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": epoch_loss / (pbar.n + 1)})
        print(f"Epoch {epoch + 1} Training Loss: {epoch_loss / len(train_dataloader)}")

        writer.add_scalar('Loss/train', epoch_loss / len(train_dataloader), epoch)

        # Validation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                logits = model(inputs)
                loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
                val_loss += loss.item()
        val_loss /= len(val_dataloader)
        print(f"Epoch {epoch + 1} Validation Loss: {val_loss}")

        writer.add_scalar('Loss/val', val_loss, epoch)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch + 1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)
            print(f"Model saved at {checkpoint_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        # Revert to training mode before starting the next epoch
        model.train()

    # Test loop
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for inputs, targets in test_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
            test_loss += loss.item()
    test_loss /= len(test_dataloader)
    print(f"Test Loss: {test_loss}")
    writer.add_scalar('Loss/test', test_loss)

    writer.close()

if __name__ == "__main__":
    main()
