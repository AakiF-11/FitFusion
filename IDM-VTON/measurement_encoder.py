"""
FitFusion — MeasurementEncoder Module (V2)
===========================================
Encodes body measurements + garment size info into conditioning
embeddings that inject into IDM-VTON's cross-attention alongside the
existing IP-Adapter garment tokens.

V2 changes:
  - 12 input dimensions (was 9): added garment_size_index, person_size_index, size_delta
  - Explicit garment↔person size gap signal for true size-awareness

The measurement tokens are APPENDED to encoder_hidden_states before
the IP-Adapter tokens, so the attention processor sees:

    [text_tokens ... | measurement_tokens | ip_adapter_tokens]

This way, the existing IP-Adapter split logic (which takes the LAST
`num_tokens` from encoder_hidden_states) continues to work unchanged.
We just need to increase the split offset by `num_measurement_tokens`.
"""

import torch
import torch.nn as nn
import math


class MeasurementEncoder(nn.Module):
    """
    Encodes body measurements into cross-attention conditioning tokens.

    Input:  (B, num_measurements)  — normalized floats
    Output: (B, num_tokens, output_dim) — tokens for cross-attention

    Architecture:
        MLP → reshape → LayerNorm

    The output_dim must match UNet's cross_attention_dim (2048 for SDXL).
    """

    def __init__(
        self,
        num_measurements: int = 12,
        output_dim: int = 2048,        # SDXL cross_attention_dim
        num_tokens: int = 4,            # number of conditioning tokens
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.num_measurements = num_measurements
        self.output_dim = output_dim
        self.num_tokens = num_tokens

        self.mlp = nn.Sequential(
            nn.Linear(num_measurements, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim * num_tokens),
        )
        self.norm = nn.LayerNorm(output_dim)

        # Initialize with small weights so measurement signal starts near zero
        # (doesn't disrupt pretrained model at start of training)
        self._init_weights()

    def _init_weights(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Scale down the last linear layer significantly
        last_linear = self.mlp[-1]
        nn.init.normal_(last_linear.weight, std=0.002)

    def forward(self, measurements: torch.Tensor) -> torch.Tensor:
        """
        Args:
            measurements: (B, num_measurements) normalized measurement vector:
                [bust_norm, waist_norm, hips_norm, height_norm,
                 person_size_norm, garment_type_idx,
                 whr_norm, bwr_norm, target_size_delta,
                 garment_size_norm, person_size_index_norm, size_gap]
                All values should be roughly in [-1, 1] or [0, 1].

        Returns:
            (B, num_tokens, output_dim) conditioning tokens
        """
        x = self.mlp(measurements)                              # (B, output_dim * num_tokens)
        x = x.reshape(-1, self.num_tokens, self.output_dim)     # (B, num_tokens, output_dim)
        x = self.norm(x)
        return x


# ─── Measurement normalization utilities ──────────────────────────────

# Body measurement ranges for normalization (in cm or inches)
# These cover the range from Universal Standard 4XS to 4XL
MEASUREMENT_STATS = {
    "bust_cm":   {"mean": 100.0, "std": 20.0},   # ~33" to ~52"
    "waist_cm":  {"mean": 85.0,  "std": 20.0},   # ~25" to ~45"
    "hips_cm":   {"mean": 105.0, "std": 20.0},    # ~36" to ~56"
    "height_cm": {"mean": 170.0, "std": 10.0},    # ~5'4" to ~6'0"
}

# Size label to index mapping (for Universal Standard)
SIZE_TO_INDEX = {
    "4XS": 0, "3XS": 1, "2XS": 2, "XS": 3, "S": 4,
    "M": 5, "L": 6, "XL": 7, "2XL": 8, "3XL": 9, "4XL": 10,
}

# Garment type encoding
GARMENT_TYPE_MAP = {
    "top": 0, "tee": 0, "t-shirt": 0, "shirt": 0,
    "dress": 1,
    "pants": 2, "jeans": 2, "trousers": 2,
    "bodysuit": 3, "leotard": 3,
    "jacket": 4, "coat": 4,
    "skirt": 5,
    "tights": 6, "leggings": 6,
    "hoodie": 7, "sweatshirt": 7, "sweater": 7,
}


def inches_to_cm(val_str: str) -> float:
    """Convert measurement string like '33\"' or '5\\' 9.5\"' to cm."""
    if not val_str:
        return 0.0
    val_str = val_str.strip().strip('"').strip("'")

    # Height format: 5' 9.5"
    if "'" in val_str:
        parts = val_str.replace('"', '').split("'")
        feet = float(parts[0].strip())
        inches = float(parts[1].strip()) if len(parts) > 1 and parts[1].strip() else 0
        return (feet * 12 + inches) * 2.54

    # Simple inches
    try:
        return float(val_str) * 2.54
    except ValueError:
        return 0.0


def normalize_measurements(
    bust_cm: float,
    waist_cm: float,
    hips_cm: float,
    height_cm: float,
    size_label: str = "M",
    garment_type: str = "top",
    target_size_label: str = None,
    garment_size_label: str = None,
) -> torch.Tensor:
    """
    Normalize body measurements to a tensor suitable for MeasurementEncoder.

    Returns: (12,) tensor with z-scored measurements, categoricals, derived ratios,
             and garment-vs-person size gap signals.
    """
    stats = MEASUREMENT_STATS
    bust_norm = (bust_cm - stats["bust_cm"]["mean"]) / stats["bust_cm"]["std"]
    waist_norm = (waist_cm - stats["waist_cm"]["mean"]) / stats["waist_cm"]["std"]
    hips_norm = (hips_cm - stats["hips_cm"]["mean"]) / stats["hips_cm"]["std"]
    height_norm = (height_cm - stats["height_cm"]["mean"]) / stats["height_cm"]["std"]

    # Person size index normalized to [-1, 1]
    person_size_idx = SIZE_TO_INDEX.get(size_label.upper(), 5)
    person_size_norm = (person_size_idx - 5.0) / 5.0  # maps 0-10 → -1 to 1

    # Garment type as normalized index
    garment_idx = GARMENT_TYPE_MAP.get(garment_type.lower(), 0)
    garment_norm = garment_idx / 7.0  # maps 0-7 → 0 to 1

    # Derived Ratios
    whr = waist_cm / hips_cm if hips_cm > 0 else 0.8
    bwr = bust_cm / waist_cm if waist_cm > 0 else 1.2
    whr_norm = (whr - 0.8) / 0.1
    bwr_norm = (bwr - 1.2) / 0.15

    # Target size delta (num sizes difference between person and target)
    target_idx = SIZE_TO_INDEX.get(target_size_label.upper(), person_size_idx) if target_size_label else person_size_idx
    target_size_delta = float(target_idx - person_size_idx) / 3.0

    # ── V2: Garment size conditioning ──
    # Garment size index (what size is the garment designed for?)
    garment_size_idx = SIZE_TO_INDEX.get(garment_size_label.upper(), 5) if garment_size_label else person_size_idx
    garment_size_norm = (garment_size_idx - 5.0) / 5.0  # maps 0-10 → -1 to 1

    # Person size index (explicit, separate from the z-scored body measurements)
    person_size_index_norm = (person_size_idx - 5.0) / 5.0

    # Size gap: how many sizes bigger/smaller is the garment vs the person?
    # Positive = garment is larger (baggy), negative = garment is smaller (tight)
    size_gap = float(garment_size_idx - person_size_idx) / 3.0

    return torch.tensor(
        [
            bust_norm, waist_norm, hips_norm, height_norm,
            person_size_norm, garment_norm,
            whr_norm, bwr_norm, target_size_delta,
            garment_size_norm, person_size_index_norm, size_gap,
        ],
        dtype=torch.float32,
    )
