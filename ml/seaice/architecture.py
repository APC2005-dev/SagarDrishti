"""``SmallUNetResidual`` — the exact model class from the sea-ice training notebook.

Copied verbatim from the notebook that produced ``unet_residual_v4_best.pt``;
the layer names (``e1``/``e2``/``e3``/``b``/``u3``/``d3``/``u2``/``d2``/``u1``/
``d1``/``out``) are what the saved ``state_dict`` keys refer to, so this
structure cannot be changed without invalidating the base weights.

``u3`` carries ``output_padding=(1, 0)`` because the production grid is
100 x 720: pooling gives 100 -> 50 -> 25 (odd) in latitude, so the transpose
needs one extra row to line up with the ``e3`` skip connection, while
720 -> 360 -> 180 needs none. That is a property of the grid, not a free
parameter.

The model predicts a *residual* (delta). The forecast is
``clamp(persistence + delta, 0, 1)`` where persistence is the last
concentration entry of the input window — see ``ml.seaice.inference``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ml.seaice.constants import BASE_CHANNELS, IN_CHANNELS, OUT_CHANNELS


def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.ReLU(inplace=True),
    )


class SmallUNetResidual(nn.Module):
    def __init__(self, in_ch: int = IN_CHANNELS, out_ch: int = OUT_CHANNELS, base: int = BASE_CHANNELS) -> None:
        super().__init__()
        self.e1 = conv_block(in_ch, base)
        self.e2 = conv_block(base, base * 2)
        self.e3 = conv_block(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.b = conv_block(base * 4, base * 8)
        self.u3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2, output_padding=(1, 0))
        self.d3 = conv_block(base * 8, base * 4)
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.d2 = conv_block(base * 4, base * 2)
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.d1 = conv_block(base * 2, base)
        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        b = self.b(self.pool(e3))
        d3 = self.d3(torch.cat([self.u3(b), e3], dim=1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], dim=1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], dim=1))
        return self.out(d1)


def build_model(in_ch: int = IN_CHANNELS, out_ch: int = OUT_CHANNELS, base: int = BASE_CHANNELS) -> SmallUNetResidual:
    return SmallUNetResidual(in_ch=in_ch, out_ch=out_ch, base=base)
