import torch
import torch.nn as nn
from networks.nat import ResNetNAT

class ConvTokenizer(nn.Module):
    def __init__(self, in_chans=3, embed_dim=512, norm_layer=None):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(
                in_chans,
                embed_dim // 2,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
            ),
            nn.Conv2d(
                embed_dim // 2,
                embed_dim,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
            ),
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        return x

model = ConvTokenizer()
model_nat = ResNetNAT(        
        depths=[3, 4, 6, 5],
        num_heads=[2, 4, 8, 16],
        embed_dim=512,
        mlp_ratio=3,
        drop_path_rate=0.2,
        kernel_size=7,
        num_classes=2
    )
model_nat.eval()
inp = torch.rand(1,3,224,224)
print(model_nat(inp).shape)