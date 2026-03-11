
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    D_MODEL, N_HEAD, NUM_LAYERS, FF_DIM, DROPOUT,
    SENT_D_MODEL, SENT_N_HEAD, SENT_NUM_LAYERS, SENT_FF_DIM,
    NUM_CLASSES, WINDOW_SIZE,
    NUM_STOCKS, STOCK_EMBED_DIM,
    FUSION_N_HEADS, FUSION_DROPOUT,
)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=500, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        x = x + self.pos_embedding(positions)
        return self.dropout(x)


class SentimentEncoder(nn.Module):

    def __init__(self, num_sent_features, d_model=SENT_D_MODEL,
                 n_head=SENT_N_HEAD, num_layers=SENT_NUM_LAYERS,
                 ff_dim=SENT_FF_DIM, dropout=DROPOUT, output_dim=D_MODEL):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(num_sent_features, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.pos_encoder = PositionalEncoding(d_model, max_len=WINDOW_SIZE + 50,
                                               dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Sequential(
            nn.Linear(d_model, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = self.output_proj(x)
        return x


class CrossAttentionFusion(nn.Module):

    def __init__(self, d_model=D_MODEL, n_heads=FUSION_N_HEADS,
                 dropout=FUSION_DROPOUT):
        super().__init__()

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)

        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, price_features, sentiment_features):
        attn_out, _ = self.cross_attn(
            query=price_features,
            key=sentiment_features,
            value=sentiment_features,
        )

        gate_input = torch.cat([price_features, attn_out], dim=-1)
        gate = self.gate(gate_input)
        fused = price_features + gate * attn_out
        fused = self.norm2(fused)

        fused = fused + self.ffn(fused)
        fused = self.norm3(fused)

        return fused


class SentimentEnrichedRegimeModel(nn.Module):

    def __init__(
        self,
        num_price_features,
        num_sent_features,
        d_model=D_MODEL,
        n_head=N_HEAD,
        num_layers=NUM_LAYERS,
        ff_dim=FF_DIM,
        dropout=DROPOUT,
        num_classes=NUM_CLASSES,
        num_stocks=NUM_STOCKS,
        stock_embed_dim=STOCK_EMBED_DIM,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_price_features = num_price_features
        self.num_sent_features = num_sent_features
        self.stock_embed_dim = stock_embed_dim

        self.stock_embedding = nn.Embedding(num_stocks, stock_embed_dim)

        self.price_input_proj = nn.Sequential(
            nn.Linear(num_price_features + stock_embed_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.price_pos_encoder = PositionalEncoding(
            d_model, max_len=WINDOW_SIZE + 50, dropout=dropout
        )

        price_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.price_transformer = nn.TransformerEncoder(
            price_encoder_layer, num_layers=num_layers
        )

        self.sentiment_encoder = SentimentEncoder(
            num_sent_features=num_sent_features,
            d_model=SENT_D_MODEL,
            n_head=SENT_N_HEAD,
            num_layers=SENT_NUM_LAYERS,
            ff_dim=SENT_FF_DIM,
            dropout=dropout,
            output_dim=d_model,
        )

        self.fusion = CrossAttentionFusion(
            d_model=d_model,
            n_heads=FUSION_N_HEADS,
            dropout=FUSION_DROPOUT,
        )

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

    def forward(self, x_price, x_sentiment, stock_ids=None):
        batch_size, window_size, _ = x_price.shape

        if stock_ids is not None:
            stock_emb = self.stock_embedding(stock_ids)
            stock_emb = stock_emb.unsqueeze(1).expand(-1, window_size, -1)
            x_price_in = torch.cat([x_price, stock_emb], dim=-1)
        else:
            zero_emb = torch.zeros(batch_size, window_size,
                                   self.stock_embed_dim, device=x_price.device)
            x_price_in = torch.cat([x_price, zero_emb], dim=-1)

        price_proj = self.price_input_proj(x_price_in)
        price_proj = self.price_pos_encoder(price_proj)
        price_encoded = self.price_transformer(price_proj)

        sent_encoded = self.sentiment_encoder(x_sentiment)

        fused = self.fusion(price_encoded, sent_encoded)

        last_token = fused[:, -1, :]
        mean_pool  = fused.mean(dim=1)
        pooled = torch.cat([last_token, mean_pool], dim=-1)
        pooled = self.pool_proj(pooled)

        regime_logits    = self.regime_head(pooled)
        regime_probs     = F.softmax(regime_logits, dim=-1)
        transition_logit = self.transition_head(pooled)
        transition_prob  = torch.sigmoid(transition_logit)

        return {
            "regime_logits":    regime_logits,
            "regime_probs":     regime_probs,
            "transition_logit": transition_logit,
            "transition_prob":  transition_prob,
            "features":         pooled,
        }

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def load_phase2_weights(model, checkpoint_path, strict=False):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    p2_state = checkpoint["model_state"]

    mapping = {
        "stock_embedding.weight":    "stock_embedding.weight",
        "input_proj.":               "price_input_proj.",
        "pos_encoder.":              "price_pos_encoder.",
        "transformer_encoder.":      "price_transformer.",
        "pool_proj.":                "pool_proj.",
        "regime_head.":              "regime_head.",
        "transition_head.":          "transition_head.",
    }

    mapped_state = {}
    for p2_key, p2_val in p2_state.items():
        if "fusion." in p2_key:
            continue

        mapped_key = p2_key
        for old_prefix, new_prefix in mapping.items():
            if p2_key.startswith(old_prefix):
                mapped_key = p2_key.replace(old_prefix, new_prefix, 1)
                break

        if mapped_key in model.state_dict():
            p4_shape = model.state_dict()[mapped_key].shape
            if p2_val.shape == p4_shape:
                mapped_state[mapped_key] = p2_val

    model.load_state_dict(mapped_state, strict=False)

    loaded = len(mapped_state)
    total = len(model.state_dict())
    print(f"  Loaded {loaded}/{total} parameters from Phase 2 checkpoint")

    return model


def build_model(num_price_features, num_sent_features, **kwargs):
    model = SentimentEnrichedRegimeModel(
        num_price_features=num_price_features,
        num_sent_features=num_sent_features,
        **kwargs,
    )
    param_count = model.count_parameters()
    print(f"\n{'─' * 55}")
    print(f"  Model: SentimentEnrichedRegimeModel")
    print(f"  Parameters: {param_count:,}")
    print(f"  Price features: {num_price_features}")
    print(f"  Sentiment features: {num_sent_features}")
    print(f"  Architecture: d_model={D_MODEL}, heads={N_HEAD}, "
          f"layers={NUM_LAYERS}")
    print(f"  Sentiment: d_sent={SENT_D_MODEL}, heads={SENT_N_HEAD}, "
          f"layers={SENT_NUM_LAYERS}")
    print(f"  Fusion: CrossAttention ({FUSION_N_HEADS} heads)")
    print(f"{'─' * 55}\n")
    return model


if __name__ == "__main__":
    batch_size = 4
    window_size = WINDOW_SIZE
    num_price = 18
    num_sent  = 12

    model = build_model(num_price, num_sent)

    x_price = torch.randn(batch_size, window_size, num_price)
    x_sent  = torch.randn(batch_size, window_size, num_sent)
    stock_ids = torch.randint(0, NUM_STOCKS, (batch_size,))

    output = model(x_price, x_sent, stock_ids=stock_ids)
    print(f"Regime logits:    {output['regime_logits'].shape}")
    print(f"Transition logit: {output['transition_logit'].shape}")

