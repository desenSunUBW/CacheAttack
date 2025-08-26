# train.py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn

class ClipPhraseTransformerAugmentor(nn.Module):
    def __init__(self, embed_dim=512, mlp_hidden_dim=1024, transformer_heads=8, transformer_layers=1, logo="Apple"):
        super().__init__()
        # self.logo_emb = torch.load(f"sampled_db/{logo}/logo.pt").to(dtype=torch.float)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.ReLU()
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=transformer_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, text_embed, logo_emb):
        logo_emb = logo_emb.repeat(text_embed.shape[0], 1)
        fused = torch.cat([text_embed, logo_emb], dim=-1)
        fused = self.mlp(fused)
        seq = torch.stack([text_embed, logo_emb, fused], dim=1)
        transformed = self.transformer(seq)
        out = transformed[:, 2]
        return F.normalize(self.output_proj(out), p=2, dim=-1)

def generate_mock_data(logo):
    text_embed = torch.load(f"sampled_db/original.pt").to(dtype=torch.float)
    target_embed = torch.load(f"sampled_db/{logo}/emb_with_logo.pt").to(dtype=torch.float)
    logo_emb = torch.load(f"sampled_db/{logo}/logo.pt").to(dtype=torch.float)
    # logo_emb = logo_emb.repeat(text_embed.shape[0], 1)
    return text_embed, logo_emb, target_embed

def train():
    # logos = ["Chanel", "Nike", "Apple", "Barcelona", "KFC", "Mcdonald"]
    logos = ["blue moon sign", "Mcdonald sign", "Apple sign", "Chanel symbol", "circled triangle symbol", "circled Nike symbol"]
    for logo in logos:

        embed_dim = 768
        batch_size = 64
        num_epochs = 50
        lr = 1e-4
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        text_embed, logo_emb, target_embed = generate_mock_data(logo)
        dataset = TensorDataset(text_embed, target_embed)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model = ClipPhraseTransformerAugmentor(embed_dim=embed_dim, logo=logo).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)

        for epoch in range(num_epochs):
            total_loss = 0
            model.train()
            for batch in dataloader:
                text, target = [x.to(device) for x in batch]
                output = model(text, logo_emb)
                # loss_target = F.mse_loss(output, target).mean()
                loss_target = (1 - F.cosine_similarity(output, target, dim=-1)).pow(2).mean()
                loss_text_preserve = (F.cosine_similarity(target, text, dim=-1) - F.cosine_similarity(output, text, dim=-1)).pow(2).mean()
                loss_text_logo = (F.cosine_similarity(target, logo_emb, dim=-1) - F.cosine_similarity(output, logo_emb, dim=-1)).pow(2).mean()
                loss = loss_target + loss_text_preserve + 0.1 * loss_text_logo
                # loss = 1 - F.cosine_similarity(output, target, dim=-1).mean() - 0.5 * F.cosine_similarity(text, target, dim=-1).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(dataloader)
            print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}")

        torch.save(model.state_dict(), f"sampled_db/{logo}/clip_phrase_model.pt")

if __name__ == "__main__":
    train()