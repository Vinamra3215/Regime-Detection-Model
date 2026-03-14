
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    D_MODEL, N_HEAD, NUM_LAYERS, FF_DIM, DROPOUT,
    NUM_CLASSES, WINDOW_SIZE,
    NUM_STOCKS, STOCK_EMBED_DIM,
)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        x = x + self.pos_embedding(positions)
        return self.dropout(x)

class SentimentFusionPlaceholder(nn.Module):

    def __init__(self, d_model: int, enabled: bool = False):
        super().__init__()
        self.enabled = enabled
        if enabled:
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=d_model, num_heads=4, dropout=0.1, batch_first=True
            )
            self.norm = nn.LayerNorm(d_model)
            self.gate = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.Sigmoid()
            )

    def forward(self, price_features: torch.Tensor,
                sentiment_features: torch.Tensor = None) -> torch.Tensor:
        if not self.enabled or sentiment_features is None:
            return price_features

        attn_out, _ = self.cross_attn(
            query=price_features,
            key=sentiment_features,
            value=sentiment_features
        )

        gate_input = torch.cat([price_features, attn_out], dim=-1)
        gate = self.gate(gate_input)
        fused = price_features + gate * attn_out
        fused = self.norm(fused)

        return fused

class TransformerRegimeModel(nn.Module):

    def __init__(
        self,
        num_features: int,
        d_model: int = D_MODEL,
        n_head: int = N_HEAD,
        num_layers: int = NUM_LAYERS,
        ff_dim: int = FF_DIM,
        dropout: float = DROPOUT,
        num_classes: int = NUM_CLASSES,
        num_stocks: int = NUM_STOCKS,
        stock_embed_dim: int = STOCK_EMBED_DIM,
        enable_fusion: bool = False,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_features = num_features
        self.stock_embed_dim = stock_embed_dim

        self.stock_embedding = nn.Embedding(num_stocks, stock_embed_dim)

        self.input_proj = nn.Sequential(
            nn.Linear(num_features + stock_embed_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.pos_encoder = PositionalEncoding(d_model, max_len=WINDOW_SIZE + 50, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.fusion = SentimentFusionPlaceholder(d_model, enabled=enable_fusion)

        self.pool_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.regime_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

        self.transition_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    def forward(
        self,
        x: torch.Tensor,
        stock_ids: torch.Tensor = None,
        sentiment: torch.Tensor = None,
        return_attention: bool = False,
    ) -> dict[str, torch.Tensor]:
        batch_size, window_size, _ = x.shape

        if stock_ids is not None:
            stock_emb = self.stock_embedding(stock_ids)
            stock_emb = stock_emb.unsqueeze(1).expand(-1, window_size, -1)
            x = torch.cat([x, stock_emb], dim=-1)
        else:
            zero_emb = torch.zeros(batch_size, window_size, self.stock_embed_dim, device=x.device)
            x = torch.cat([x, zero_emb], dim=-1)

        x = self.input_proj(x)

        x = self.pos_encoder(x)

        encoded = self.transformer_encoder(x)

        fused = self.fusion(encoded, sentiment)

        last_token = fused[:, -1, :]
        mean_pool  = fused.mean(dim=1)
        pooled     = torch.cat([last_token, mean_pool], dim=-1)
        pooled     = self.pool_proj(pooled)

        regime_logits      = self.regime_head(pooled)
        regime_probs       = F.softmax(regime_logits, dim=-1)
        transition_logit   = self.transition_head(pooled)
        transition_prob    = torch.sigmoid(transition_logit)

        output = {
            "regime_logits":    regime_logits,
            "regime_probs":     regime_probs,
            "transition_logit": transition_logit,
            "transition_prob":  transition_prob,
            "features":         pooled,
        }

        return output

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def build_model(num_features: int, **kwargs) -> TransformerRegimeModel:
    model = TransformerRegimeModel(num_features=num_features, **kwargs)
    param_count = model.count_parameters()
    print(f"\n{'─'*50}")
    print(f"  Model: TransformerRegimeModel")
    print(f"  Parameters: {param_count:,}")
    print(f"  Input features: {num_features}")
    print(f"  Architecture: d_model={D_MODEL}, heads={N_HEAD}, layers={NUM_LAYERS}")
    print(f"{'─'*50}\n")
    return model

if __name__ == "__main__":
    batch_size = 4
    window_size = WINDOW_SIZE
    num_features = 18

    model = build_model(num_features)

    x = torch.randn(batch_size, window_size, num_features)
    stock_ids = torch.randint(0, NUM_STOCKS, (batch_size,))
    output = model(x, stock_ids=stock_ids)

    print(f"regime_logits:    {output['regime_logits'].shape}")
    print(f"regime_probs:     {output['regime_probs'].shape}")
    print(f"transition_logit: {output['transition_logit'].shape}")
    print(f"transition_prob:  {output['transition_prob'].shape}")
    print(f"features:         {output['features'].shape}")
    print(f"\nSample regime probs: {output['regime_probs'][0].detach()}")
    print(f"Sample transition prob: {output['transition_prob'][0].item():.4f}")
    print(f"\nStock embedding shape: {model.stock_embedding.weight.shape}")
