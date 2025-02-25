import os
import torch
import torch.nn.functional as F
from models.gpt1 import GPT1
from tokenizers import ByteLevelBPETokenizer

def load_model(model_path, vocab_size, embed_dim, num_layers, num_heads, device):
    """
    Loads the trained model from the checkpoint path.
    """
    model = GPT1(vocab_size, embed_dim, num_layers, num_heads, dropout=0.1, max_seq_length=128)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    return model

def load_tokenizer(tokenizer_path):
    """
    Loads the tokenizer from the saved tokenizer file.
    """
    tokenizer = ByteLevelBPETokenizer(tokenizer_path)
    return tokenizer

def generate_text(model, tokenizer, prompt, max_length=128, temperature=1.0, top_k=50, device='cpu'):
    """
    Generate text using the trained model based on the given prompt.
    """
    # Encode the prompt to tokens
    input_ids = tokenizer.encode(prompt).ids
    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)

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
            generated_ids = torch.cat((generated_ids, torch.tensor([[next_token_id]], dtype=torch.long).to(device)), dim=1)

            # Stop generation if end-of-sequence token is generated
            if next_token_id == tokenizer.token_to_id("</s>"):
                break

    # Decode the generated token IDs back to text
    generated_text = tokenizer.decode(generated_ids.squeeze().tolist())
    return generated_text

def get_latest_model_path(checkpoint_dir):
    """
    Finds the most recent model checkpoint in the checkpoint directory.
    """
    # Get all model files in the checkpoint directory
    model_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
    if not model_files:
        raise FileNotFoundError("No model checkpoint files found.")

    # Sort files by modification time and return the latest
    model_files.sort(key=lambda f: os.path.getmtime(os.path.join(checkpoint_dir, f)), reverse=True)
    latest_model = model_files[0]
    return os.path.join(checkpoint_dir, latest_model)

def main():
    # Paths to the model and tokenizer
    checkpoint_dir = "models/saved"
    model_path = get_latest_model_path(checkpoint_dir)  # Get the most recent model
    tokenizer_path = os.path.join(checkpoint_dir, "tokenizer.json")

    # Model configuration
    vocab_size = 50257  # GPT-1 uses 50,257 tokens
    embed_dim = 768  # GPT-1 uses 768 for the embedding size
    num_layers = 12  # GPT-1 has 12 layers
    num_heads = 12  # GPT-1 has 12 attention heads
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    # Load model and tokenizer
    model = load_model(model_path, vocab_size, embed_dim, num_layers, num_heads, device)
    tokenizer = load_tokenizer(tokenizer_path)

    # Get the prompt from the user or a sample prompt
    prompt = input("Enter a prompt: ") or "Once upon a time"

    # Generate text
    generated_text = generate_text(model, tokenizer, prompt)
    print("\nGenerated Text:")
    print(generated_text)

if __name__ == "__main__":
    main()
