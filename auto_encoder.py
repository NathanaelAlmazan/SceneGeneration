import os
import math
import torch


class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.group_norm_1 = torch.nn.GroupNorm(32, in_channels)
        self.conv_1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.group_norm_2 = torch.nn.GroupNorm(32, out_channels)
        self.conv_2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = torch.nn.Identity()
        else:
            self.residual_layer = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    
    def forward(self, x):
        residue = x

        x = self.group_norm_1(x)
        x = torch.nn.functional.silu(x)
        x = self.conv_1(x)

        x = self.group_norm_2(x)
        x = torch.nn.functional.silu(x)
        x = self.conv_2(x)

        return x + self.residual_layer(residue)


class SelfAttention(torch.nn.Module):
    def __init__(self, n_heads, d_embed, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.in_proj = torch.nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = torch.nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x, causal_mask=False):
        input_shape = x.shape
        batch_size, sequence_length, _ = input_shape
        interim_shape = (batch_size, sequence_length, self.n_heads, self.d_head)

        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        weight = q @ k.transpose(-1, -2)
        if causal_mask:
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)
        weight /= math.sqrt(self.d_head)
        weight = torch.nn.functional.softmax(weight, dim=-1)

        output = weight @ v
        output = output.transpose(1, 2)
        output = output.reshape(input_shape)
        output = self.out_proj(output)
        return output


class AttentionBlock(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.group_norm = torch.nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)
    
    def forward(self, x):
        residue = x
        x = self.group_norm(x)

        n, c, h, w = x.shape
        x = x.view((n, c, h * w))
        x = x.transpose(-1, -2)
        x = self.attention(x)
        x = x.transpose(-1, -2)
        x = x.view((n, c, h, w))

        x += residue
        return x


class Encoder(torch.nn.Sequential):
    def __init__(self):
        super().__init__(
            torch.nn.Conv2d(3, 64, kernel_size=3, padding=1),
            ResidualBlock(64, 64),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=0),
            ResidualBlock(64, 128),
            torch.nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            ResidualBlock(128, 256),
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
            ResidualBlock(256, 256),
            AttentionBlock(256),
            ResidualBlock(256, 256),
            torch.nn.GroupNorm(32, 256),
            torch.nn.SiLU(),
            torch.nn.Conv2d(256, 8, kernel_size=3, padding=1),
            torch.nn.Conv2d(8, 8, kernel_size=1, padding=0),
        )

    def forward(self, x):
        for module in self:
            if getattr(module, 'stride', None) == (2, 2):
                x = torch.nn.functional.pad(x, (0, 1, 0, 1))
            x = module(x)

        mean, log_variance = torch.chunk(x, 2, dim=1)
        log_variance = torch.clamp(log_variance, -30, 20)
        variance = log_variance.exp()
        stdev = variance.sqrt()
        noise = torch.randn(*mean.size()).float().cuda()
        x = mean + stdev * noise

        x *= 0.18215
        return mean, log_variance, x


class Decoder(torch.nn.Sequential):
    def __init__(self):
        super().__init__(
            torch.nn.Conv2d(4, 4, kernel_size=1, padding=0),
            torch.nn.Conv2d(4, 256, kernel_size=3, padding=1),
            ResidualBlock(256, 256),
            AttentionBlock(256),
            ResidualBlock(256, 256),
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            ResidualBlock(256, 256),
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            ResidualBlock(256, 128),
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
            ResidualBlock(128, 64),
            torch.nn.GroupNorm(32, 64),
            torch.nn.SiLU(),
            torch.nn.Conv2d(64, 3, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x /= 0.18215
        for module in self:
            x = module(x)
        return x


# load weights
encoder = Encoder().float().cuda()
decoder = Decoder().float().cuda()

encoder.load_state_dict(torch.load(os.path.join("diffusion_vae", "encoder_ckpt.pt")))
decoder.load_state_dict(torch.load(os.path.join("diffusion_vae", "decoder_ckpt.pt")))

encoder.eval()
decoder.eval()

def encode_images(images: torch.Tensor):
    images = images.cuda()
    _, _, z = encoder(images)
    return z

def decode_images(x: torch.Tensor):
    x = decoder(x)
    return x

print("Loaded VAE!")