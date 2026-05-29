from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn


# ── VGG16 ──────────────────────────────────────────────────────────────────

class VGGBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_convs: int, use_pool: bool = True) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_channels = in_channels
        for _ in range(num_convs):
            layers.extend([
                nn.Conv2d(current_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ])
            current_channels = out_channels
        if use_pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class VGG16Scratch(nn.Module):
    def __init__(self, num_classes: int = 8) -> None:
        super().__init__()
        self.block1 = VGGBlock(3, 64, num_convs=2)
        self.block2 = VGGBlock(64, 128, num_convs=2)
        self.block3 = VGGBlock(128, 256, num_convs=3)
        self.block4 = VGGBlock(256, 512, num_convs=3)
        self.block5 = VGGBlock(512, 512, num_convs=3)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        for block in [self.block1, self.block2, self.block3, self.block4, self.block5]:
            x = block(x)
        x = torch.flatten(self.global_pool(x), 1)
        return self.classifier(x)


# ── ResNet50 ───────────────────────────────────────────────────────────────

class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels: int, planes: int, stride: int = 1, downsample: nn.Module | None = None) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResNet50Scratch(nn.Module):
    def __init__(self, num_classes: int = 8) -> None:
        super().__init__()
        self.stem_conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(64)
        self.stem_relu = nn.ReLU(inplace=True)
        self._in_channels = 64
        self.layer1 = self._make_layer(64, 3, stride=1)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512 * BottleneckBlock.expansion, num_classes)

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        out_channels = planes * BottleneckBlock.expansion
        if stride != 1 or self._in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self._in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        layers = [BottleneckBlock(self._in_channels, planes, stride=stride, downsample=downsample)]
        self._in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(BottleneckBlock(self._in_channels, planes))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem_relu(self.stem_bn(self.stem_conv(x)))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.classifier(torch.flatten(self.global_pool(x), 1))


# ── DenseNet121 ────────────────────────────────────────────────────────────

class DenseLayer(nn.Module):
    def __init__(self, in_channels: int, growth_rate: int) -> None:
        super().__init__()
        bn = growth_rate * 4
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, bn, kernel_size=1, bias=False)
        self.norm2 = nn.BatchNorm2d(bn)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        nf = self.conv1(self.relu1(self.norm1(x)))
        nf = self.conv2(self.relu2(self.norm2(nf)))
        return torch.cat([x, nf], dim=1)


class DenseBlock(nn.Module):
    def __init__(self, num_layers: int, in_channels: int, growth_rate: int) -> None:
        super().__init__()
        layers, ch = [], in_channels
        for _ in range(num_layers):
            layers.append(DenseLayer(ch, growth_rate))
            ch += growth_rate
        self.layers = nn.ModuleList(layers)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class TransitionLayer(nn.Module):
    def __init__(self, in_channels: int, compression: float = 0.5) -> None:
        super().__init__()
        out = int(in_channels * compression)
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: Tensor) -> Tensor:
        return self.pool(self.conv(self.relu(self.norm(x))))


class DenseNet121Scratch(nn.Module):
    def __init__(
        self,
        num_classes: int = 8,
        growth_rate: int = 32,
        compression: float = 0.5,
        block_layers: Sequence[int] = (6, 12, 24, 16),
    ) -> None:
        super().__init__()
        self.growth_rate = growth_rate
        self.compression = compression
        self.block_layers = tuple(block_layers)

        num_features = 64
        self.stem_conv = nn.Conv2d(3, num_features, kernel_size=3, stride=1, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(num_features)
        self.stem_relu = nn.ReLU(inplace=True)

        blocks, transitions = [], []
        for idx, nl in enumerate(self.block_layers):
            blocks.append(DenseBlock(nl, num_features, growth_rate))
            num_features += nl * growth_rate
            if idx != len(self.block_layers) - 1:
                transitions.append(TransitionLayer(num_features, compression))
                num_features = int(num_features * compression)

        self.blocks = nn.ModuleList(blocks)
        self.transitions = nn.ModuleList(transitions)
        self.final_bn = nn.BatchNorm2d(num_features)
        self.final_relu = nn.ReLU(inplace=True)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem_relu(self.stem_bn(self.stem_conv(x)))
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx < len(self.transitions):
                x = self.transitions[idx](x)
        x = self.final_relu(self.final_bn(x))
        return self.classifier(torch.flatten(self.global_pool(x), 1))


# ── Vision Transformer ─────────────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    def __init__(self, img_size: int, patch_size: int, in_channels: int, embed_dim: int) -> None:
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.projection(x).flatten(2).transpose(1, 2)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, attn_drop: float = 0.0, proj_drop: float = 0.0) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        bsz, num_tokens, channels = x.shape
        qkv = self.qkv(x).reshape(bsz, num_tokens, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(bsz, num_tokens, channels)
        return self.proj_drop(self.proj(x))


class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, mlp_dim: int, drop: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, mlp_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(mlp_dim, embed_dim)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        return self.drop2(self.fc2(self.drop1(self.act(self.fc1(x)))))


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int, attn_drop: float = 0.0, drop: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, mlp_dim, drop=drop)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.ff(self.norm2(x))


class ViTScratch(nn.Module):
    def __init__(
        self,
        img_size: int = 48,
        patch_size: int = 6,
        in_channels: int = 3,
        num_classes: int = 8,
        embed_dim: int = 256,
        depth: int = 8,
        num_heads: int = 8,
        mlp_dim: int = 512,
        dropout_rate: float = 0.05,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout_rate)
        self.encoder = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_dim, attn_drop=dropout_rate, drop=dropout_rate)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor:
        bsz = x.shape[0]
        x = torch.cat([self.cls_token.expand(bsz, -1, -1), self.patch_embed(x)], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for block in self.encoder:
            x = block(x)
        return self.classifier(self.norm(x)[:, 0])
