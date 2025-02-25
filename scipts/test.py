import os
import torch
import torch.nn.functional as F
from models.gpt1 import GPT1
from tokenizers import ByteLevelBPETokenizer


def load_model(model_path, vocab_size, embed_dim, num_layers, num_heads, device):
    """
    Loads the trained model from the checkpoint path.
    """
    model = GPT1(vocab_size, embed_dim, num_layers, num_heads, dropout=0.1, max_seq_length=512)
    checkpoint = torch.load(model_path, map_location=device)  # Ensure it's loaded on the correct device
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def load_tokenizer(tokenizer_path):
    """
    Loads the tokenizer from the saved tokenizer file.
    """
    tokenizer = ByteLevelBPETokenizer().from_file(
        vocab_filename=os.path.join(tokenizer_path, "vocab.json"),
        merges_filename=os.path.join(tokenizer_path, "merges.txt")
    )
    return tokenizer


def generate_text(model, tokenizer, prompt, max_length=128, temperature=1.0, top_k=50, device='cpu'):
    """
    Generate text using the trained model based on the given prompt.
    """
    # Encode the prompt to tokens
    input_ids = tokenizer.encode(prompt).ids
    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)

    # Ensure that input_tensor is not empty
    if input_tensor.numel() == 0:
        raise ValueError("Input tensor is empty!")

    generated_ids = input_tensor

    # Generate text
    with torch.no_grad():
        for _ in range(max_length):
            logits = model(generated_ids)
            logits = logits[:, -1, :] / temperature  # Scale logits by temperature
            probs = F.softmax(logits, dim=-1)

            # Top-k sampling
            top_probs, top_indices = torch.topk(probs, top_k)
            top_probs = top_probs.squeeze(0)
            top_indices = top_indices.squeeze(0)

            next_token_id = top_indices[torch.multinomial(top_probs, 1)].item()
            generated_ids = torch.cat((generated_ids, torch.tensor([[next_token_id]], dtype=torch.long).to(device)),
                                      dim=1)

            # Stop generation if end-of-sequence token is generated
            if next_token_id == tokenizer.token_to_id("</s>"):
                break

    # Decode the generated token IDs back to text
    generated_text = tokenizer.decode(generated_ids.squeeze().tolist())
    return generated_text


def evaluate_model(model, valid_file, tokenizer, device):
    """
    Evaluate the model on the validation file and return a performance metric (e.g., loss).
    For simplicity, this function will return the loss on the validation file.
    """
    model.eval()

    # Load validation data
    with open(valid_file, "r", encoding="utf-8") as f:
        valid_data = f.read()

    input_ids = tokenizer.encode(valid_data).ids
    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)

    # Ensure that input_tensor is not empty
    if input_tensor.numel() == 0:
        raise ValueError("Input tensor is empty!")

    # Forward pass
    with torch.no_grad():
        logits = model(input_tensor)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), input_tensor.view(-1))

    return loss.item()


def save_checkpoint(model, optimizer, epoch, checkpoint_dir):
    """
    Save the model checkpoint including the model and optimizer states.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pth")
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


def get_latest_model_path(checkpoint_dir):
    """
    Returns the path to the latest model checkpoint file.
    """
    # Get all model files in the checkpoint directory
    model_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
    if not model_files:
        raise FileNotFoundError("No model checkpoint files found.")

    # Sort models based on their file modification time (latest first)
    model_files.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True)

    # Return the latest model file
    return os.path.join(checkpoint_dir, model_files[0])


def main():
    # Paths to the model and tokenizer
    checkpoint_dir = "models/saved"
    val_file = "data/raw/valid.txt"
    tokenizer_dir = os.path.join(checkpoint_dir, "tokenizer")

    # Model configuration
    vocab_size = 50257  # GPT-1 uses 50,257 tokens
    embed_dim = 768  # GPT-1 uses 768 for the embedding size
    num_layers = 12  # GPT-1 has 12 layers
    num_heads = 12  # GPT-1 has 12 attention heads

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer
    tokenizer = load_tokenizer(tokenizer_dir)

    # Get the latest model checkpoint
    model_path = get_latest_model_path(checkpoint_dir)
    print(f"Latest model: {model_path}")

    # Load the latest model
    model = load_model(model_path, vocab_size, embed_dim, num_layers, num_heads, device)

    # Get the prompt from the user or a sample prompt
    prompt = input("Enter a prompt: ") or "Once upon a time"

    # Generate text
    generated_text = generate_text(model, tokenizer, prompt, device=device)
    print("\nGenerated Text:")
    print(generated_text)


if __name__ == "__main__":
    main()
