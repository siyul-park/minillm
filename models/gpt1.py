import torch
import torch.nn as nn


class GPTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(GPTBlock, self).__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, tgt_mask):
        attn_output, _ = self.self_attn(x, x, x, attn_mask=tgt_mask)
        x = x + attn_output
        x = self.ln1(x)
        mlp_output = self.mlp(x)
        x = x + mlp_output
        x = self.ln2(x)
        return x


class GPT1(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, dropout=0.1, max_seq_length=512):
        super(GPT1, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_length, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            GPTBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, x):
        batch_size, seq_length = x.size()
        positions = torch.arange(0, seq_length, device=x.device).unsqueeze(0).expand(batch_size, seq_length)
        token_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(positions)
        h = self.dropout(token_emb + pos_emb)
        h = h.transpose(0, 1)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_length).to(x.device)

        for layer in self.layers:
            h = layer(h, tgt_mask)
        h = self.ln_f(h)
        h = h.transpose(0, 1)
        logits = self.head(h)
        return logits
