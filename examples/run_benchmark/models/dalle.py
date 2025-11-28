#!/usr/bin/env python
#
# This is a varient of DALLE-pytorch.
# https://github.com/lucidrains/DALLE-pytorch

import os
from collections import deque
from typing import Optional, Literal, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

from transformers import GPT2Tokenizer
from transformers import PretrainedConfig, PreTrainedModel


def load_model(path):
    with open(path, 'rb') as f:
        return torch.load(f, map_location = torch.device('cpu'), weights_only=False)


def download(url, filename = None, root = os.path.expanduser("~/.cache/dalle")):
    from tqdm import tqdm
    import urllib.request
    
    os.makedirs(root, exist_ok = True)
    filename = filename or os.path.basename(url)
    download_target = os.path.join(root, filename)
    
    if os.path.isfile(download_target):
        return download_target
    
    download_target_tmp = os.path.join(root, f'tmp.{filename}')
    
    with urllib.request.urlopen(url) as source, open(download_target_tmp, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break
                output.write(buffer)
                loop.update(len(buffer))
    
    os.rename(download_target_tmp, download_target)
    return download_target


class OpenAIDiscreteVAE(nn.Module):
    def __init__(self):
        super().__init__()
        # assert version.parse(get_pkg_version('torch')) < version.parse('1.11.0'), 'torch version must be <= 1.10 in order to use OpenAI discrete vae'

        self.enc = load_model(download('https://cdn.openai.com/dall-e/encoder.pkl'))
        self.dec = load_model(download('https://cdn.openai.com/dall-e/decoder.pkl'))

        with torch.no_grad():
            for param in self.parameters():
                param.set_(param.contiguous())

        # Extract num_layers and num_tokens from loaded models
        self.num_layers = 3
        self.num_tokens = self.dec.vocab_size

    @torch.no_grad()
    def get_codebook_indices(self, img):
        img = (1 - 2 * 0.1) * img + 0.1
        z_logits = self.enc.blocks(img)
        z = torch.argmax(z_logits, dim = 1)
        # rearrange(z, 'b h w -> b (h w)') equivalent:
        return z.view(z.size(0), -1)

    def decode(self, img_seq):
        b, n = img_seq.shape
        # rearrange(img_seq, 'b (h w) -> b h w', h = int(sqrt(n))) equivalent:
        h = int(n**0.5)
        img_seq = img_seq.view(b, h, h)

        z = F.one_hot(img_seq, num_classes = self.num_tokens)
        # rearrange(z, 'b h w c -> b c h w') equivalent:
        z = z.permute(0, 3, 1, 2).float()
        x_stats = self.dec(z).float()
        x_rec = torch.sigmoid(x_stats[:, :3])
        x_rec = torch.clamp((x_rec - 0.1) / (1 - 2 * 0.1), 0, 1)
        return x_rec

    def forward(self, img):
        raise NotImplemented


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        custom_freqs: torch.Tensor | None = None,
        freqs_for:  Literal['lang', 'pixel', 'constant'] = 'lang',
        theta = 10000,
        max_freq = 10,
        num_freqs = 1,
        use_xpos = False,
        xpos_scale_base = 512,
        interpolate_factor = 1.,
        theta_rescale_factor = 1.,
        seq_before_head_dim = False,
        cache_if_possible = True,
        cache_max_seq_len = 8192
    ):
        super().__init__()
        # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
        # has some connection to NTK literature
        # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/

        theta *= theta_rescale_factor ** (dim / (dim - 2))

        self.freqs_for = freqs_for

        if custom_freqs is not None:
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * torch.pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()

        self.cache_if_possible = cache_if_possible
        self.cache_max_seq_len = cache_max_seq_len

        self.register_buffer('cached_freqs', torch.zeros(cache_max_seq_len, dim), persistent = False)
        self.cached_freqs_seq_len = 0

        self.freqs = freqs

        # dummy for device

        self.register_buffer('dummy', torch.tensor(0), persistent = False)

        # default sequence dimension

        self.seq_before_head_dim = seq_before_head_dim
        self.default_seq_dim = -3 if seq_before_head_dim else -2

        # interpolation factors

        assert interpolate_factor >= 1.
        self.interpolate_factor = interpolate_factor

        # xpos

        self.use_xpos = use_xpos

        if not use_xpos:
            return

        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.scale_base = xpos_scale_base

        self.register_buffer('scale', scale, persistent = False)
        self.register_buffer('cached_scales', torch.zeros(cache_max_seq_len, dim), persistent = False)
        self.cached_scales_seq_len = 0

        # add apply_rotary_emb as static method

        self.apply_rotary_emb = staticmethod(apply_rotary_emb)

    @property
    def device(self):
        return self.dummy.device

    def get_seq_pos(self, seq_len, device = None, dtype = None, offset = 0):
        device = device if device is not None else self.device
        dtype = dtype if dtype is not None else self.cached_freqs.dtype

        return (torch.arange(seq_len, device = device, dtype = dtype) + offset) / self.interpolate_factor

    def rotate_queries_or_keys(self, t, seq_dim = None, offset = 0, scale = None):
        seq_dim = seq_dim if seq_dim is not None else self.default_seq_dim

        assert not self.use_xpos or scale is not None, 'you must use `.rotate_queries_and_keys` method instead and pass in both queries and keys, for length extrapolatable rotary embeddings'

        device, dtype, seq_len = t.device, t.dtype, t.shape[seq_dim]

        seq = self.get_seq_pos(seq_len, device = device, dtype = dtype, offset = offset)

        freqs = self.forward(seq, seq_len = seq_len, offset = offset)

        if seq_dim == -3:
            freqs = freqs.unsqueeze(-2)  # 'n d -> n 1 d'

        return apply_rotary_emb(freqs, t, scale = scale if scale is not None else 1., seq_dim = seq_dim)

    def rotate_queries_with_cached_keys(self, q, k, seq_dim = None, offset = 0):
        dtype, device, seq_dim = q.dtype, q.device, seq_dim if seq_dim is not None else self.default_seq_dim

        q_len, k_len = q.shape[seq_dim], k.shape[seq_dim]
        assert q_len <= k_len

        q_scale = k_scale = 1.

        if self.use_xpos:
            seq = self.get_seq_pos(k_len, dtype = dtype, device = device)

            q_scale = self.get_scale(seq[-q_len:]).type(dtype)
            k_scale = self.get_scale(seq).type(dtype)

        rotated_q = self.rotate_queries_or_keys(q, seq_dim = seq_dim, scale = q_scale, offset = k_len - q_len + offset)
        rotated_k = self.rotate_queries_or_keys(k, seq_dim = seq_dim, scale = k_scale ** -1)

        rotated_q = rotated_q.type(q.dtype)
        rotated_k = rotated_k.type(k.dtype)

        return rotated_q, rotated_k

    def rotate_queries_and_keys(self, q, k, seq_dim = None):
        seq_dim = seq_dim if seq_dim is not None else self.default_seq_dim

        assert self.use_xpos
        device, dtype, seq_len = q.device, q.dtype, q.shape[seq_dim]

        seq = self.get_seq_pos(seq_len, dtype = dtype, device = device)

        freqs = self.forward(seq, seq_len = seq_len)
        scale = self.get_scale(seq, seq_len = seq_len).to(dtype)

        if seq_dim == -3:
            freqs = freqs.unsqueeze(-2)  # 'n d -> n 1 d'
            scale = scale.unsqueeze(-2)  # 'n d -> n 1 d'

        rotated_q = apply_rotary_emb(freqs, q, scale = scale, seq_dim = seq_dim)
        rotated_k = apply_rotary_emb(freqs, k, scale = scale ** -1, seq_dim = seq_dim)

        rotated_q = rotated_q.type(q.dtype)
        rotated_k = rotated_k.type(k.dtype)

        return rotated_q, rotated_k

    def get_scale(
        self,
        t: torch.Tensor,
        seq_len: int | None = None,
        offset = 0
    ):
        assert self.use_xpos

        should_cache = (
            self.cache_if_possible and
            seq_len is not None and
            (offset + seq_len) <= self.cache_max_seq_len
        )

        if (
            should_cache and \
            self.cached_scales is not None and \
            (seq_len + offset) <= self.cached_scales_seq_len
        ):
            return self.cached_scales[offset:(offset + seq_len)]

        scale = 1.
        if self.use_xpos:
            power = (t - len(t) // 2) / self.scale_base
            scale = self.scale ** power.unsqueeze(-1)  # 'n -> n 1'
            scale = scale.repeat(1, 2)  # 'n d -> n (d r)' where r=2

        if should_cache and offset == 0:
            self.cached_scales[:seq_len] = scale.detach()
            self.cached_scales_seq_len = seq_len

        return scale

    def get_axial_freqs(
        self,
        *dims,
        offsets: (
            tuple[int | float, ...] |
            torch.Tensor |
            None
        ) = None
    ):
        Colon = slice(None)
        all_freqs = []

        # handle offset

        if offsets is not None:
            if not torch.is_tensor(offsets):
                offsets = torch.tensor(offsets)

            assert len(offsets) == len(dims)

        # get frequencies for each axis

        for ind, dim in enumerate(dims):

            offset = 0
            if offsets is not None:
                offset = offsets[ind]

            if self.freqs_for == 'pixel':
                pos = torch.linspace(-1, 1, steps = dim, device = self.device)
            else:
                pos = torch.arange(dim, device = self.device)

            pos = pos + offset

            freqs = self.forward(pos, seq_len = dim)

            all_axis = [None] * len(dims)
            all_axis[ind] = Colon

            new_axis_slice = (Ellipsis, *all_axis, Colon)
            all_freqs.append(freqs[new_axis_slice])

        # concat all freqs

        all_freqs = torch.broadcast_tensors(*all_freqs)
        return torch.cat(all_freqs, dim = -1)

    @autocast('cuda', enabled = False)
    def forward(
        self,
        t: torch.Tensor,
        seq_len: int | None = None,
        offset = 0
    ):
        should_cache = (
            self.cache_if_possible and
            seq_len is not None and
            self.freqs_for != 'pixel' and
            (offset + seq_len) <= self.cache_max_seq_len
        )

        if (
            should_cache and \
            self.cached_freqs is not None and \
            (offset + seq_len) <= self.cached_freqs_seq_len
        ):
            return self.cached_freqs[offset:(offset + seq_len)].detach()

        freqs = self.freqs

        freqs = t.type(freqs.dtype).unsqueeze(-1) * freqs.to(t.device)
        freqs = freqs.repeat_interleave(2, dim=-1)  # '... n -> ... (n r)' where r=2

        if should_cache and offset == 0:
            self.cached_freqs[:seq_len] = freqs.detach()
            self.cached_freqs_seq_len = seq_len

        return freqs


@autocast('cuda', enabled = False)
def apply_rotary_emb(
    freqs,
    t,
    start_index = 0,
    scale = 1.,
    seq_dim = -2,
    freqs_seq_dim = None
):
    dtype = t.dtype

    if freqs_seq_dim is None:
        if freqs.ndim == 2 or t.ndim == 3:
            freqs_seq_dim = 0

    if t.ndim == 3 or freqs_seq_dim is not None:
        seq_len = t.shape[seq_dim]

        def slice_at_dim(t, dim_slice: slice, *, dim):
            dim += (t.ndim if dim < 0 else 0)
            colons = [slice(None)] * t.ndim
            colons[dim] = dim_slice
            return t[tuple(colons)]

        freqs = slice_at_dim(freqs, slice(-seq_len, None), dim = freqs_seq_dim)

    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim

    assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'

    # Split t into three parts: left, middle (to be transformed), and right
    t_left = t[..., :start_index]
    t_middle = t[..., start_index:end_index]
    t_right = t[..., end_index:]

    # Apply rotary embeddings without modifying t in place
    def rotate_half(x):
        d = x.shape[-1]
        x = x.view(*x.shape[:-1], d // 2, 2)
        x1, x2 = x.unbind(dim = -1)
        x = torch.stack((-x2, x1), dim = -1)
        return x.view(*x.shape[:-2], d)

    t_transformed = (t_middle * freqs.cos() * scale) + (rotate_half(t_middle) * freqs.sin() * scale)
        
    out = torch.cat((t_left, t_transformed, t_right), dim=-1)

    return out.type(dtype)


class PreShiftToken(nn.Module):
    def __init__(self, fn, image_size, seq_len):
        super().__init__()
        self.fn = fn
        self.image_size = image_size
        self.seq_len = seq_len
        self.img_seq_len = image_size ** 2
        self.text_len = seq_len - self.img_seq_len + 1

    def forward(self, x, cache=None, cache_key=None, **kwargs):
        seq_len, image_size, text_len = self.seq_len, self.image_size, self.text_len

        if cache is not None and cache_key in cache:
            offset = cache['offset']
            assert offset >= text_len, "cached inference for text is not supported"
            q = cache[cache_key]
            assert isinstance(q, deque) and len(q) == image_size

            x_top, x_left, *x_pass = x[:, -1].chunk(4, dim=-1)

            q.append((x_top, x_left))
            x_top = q.popleft()[0]
            x_left = q[-2][1]
            if (offset - text_len) % image_size == 0:
                x_left = torch.zeros_like(x_left)

            x = torch.cat((x_top, x_left, *x_pass), dim=-1)
            return self.fn(x[:, None], cache=cache, **kwargs)

        n = x.shape[1]
        padding = seq_len - n + 1

        # if sequence is shorter than the text length, no image tokens to shift

        if n < text_len:
            return self.fn(x, **kwargs)

        # get text and image tokens

        x_text, x_img = x[:, :text_len], x[:, text_len:]
        x_img = F.pad(x_img, (0, 0, 0, padding))
        x_img = x_img.view(x_img.shape[0], image_size, image_size, x_img.shape[-1])

        # shift 1 from the left for text tokens

        x_text_shift, x_text_pass = x_text.chunk(2, dim = -1)
        x_text_shift = F.pad(x_text_shift, (0, 0, 1, -1))
        x_text = torch.cat((x_text_shift, x_text_pass), dim = -1)

        # shift from top, left for image tokens

        x_img_shift_top, x_img_shift_left, *x_img_pass = x_img.chunk(4, dim = -1)
        x_img_shift_left = F.pad(x_img_shift_left, (0, 0, 1, -1))
        x_img_shift_top = F.pad(x_img_shift_top, (0, 0, 0, 0, 1, -1))
        x_img = torch.cat((x_img_shift_top, x_img_shift_left, *x_img_pass), dim = -1)

        # merge text and image sequence back together

        x_img = x_img.view(x_img.shape[0], -1, x_img.shape[-1])
        x_img = x_img[:, :-padding]
        x = torch.cat((x_text, x_img), dim = 1)

        if cache is not None:
            dummy_top, dummy_left, *_ = x[:, -1].chunk(4, dim=-1)
            dummy_top, dummy_left = torch.zeros_like(dummy_top), torch.zeros_like(dummy_left)

            q = deque()
            x_img = x_img[:, -image_size:]
            for _ in range(image_size - x_img.shape[1]):
                q.append((dummy_top, dummy_left))
            for i in range(x_img.shape[1]):
                q.append(x_img[:, i].chunk(4, dim=-1)[:2])
            cache[cache_key] = q

        return self.fn(x, cache=cache, **kwargs)


class Attention(nn.Module):
    def __init__(self, dim, seq_len, causal = True, heads = 8, dim_head = 64, dropout = 0.,
                 static_mask = None):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.seq_len = seq_len
        self.scale = dim_head ** -0.5

        self.causal = causal
        self.register_buffer('static_mask', static_mask, persistent=False)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None, rotary_pos_emb = None, cache = None, cache_key = None):
        b, n, _, h, device = *x.shape, self.heads, x.device
        offset = cache.get('offset', 0) if cache is not None else 0

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: t.view(b, n, h, -1).transpose(1, 2), qkv)

        if rotary_pos_emb is not None:
            def apply_pos_emb(pos_emb, qkv):
                n = qkv[0].shape[-2]
                pos_emb = pos_emb[..., :n, :]
                return tuple(map(lambda t: apply_rotary_emb(pos_emb, t), qkv))

            q, k, v = apply_pos_emb(rotary_pos_emb[..., offset:, :], (q, k, v))

        q = q * self.scale

        if offset > 0:
            k_top, v_top = cache[cache_key]
            k = torch.cat([k_top, k], dim=-2)
            v = torch.cat([v_top, v], dim=-2)
        if cache is not None:
            cache[cache_key] = k, v

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k)
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)
            dots.masked_fill_(~mask, mask_value)
            del mask

        if self.causal and offset == 0:  # causality is naturally enforced for the cached inference
            i, j = dots.shape[-2:]
            mask = torch.ones(i, j, device = device).triu_(j - i + 1).bool()
            dots.masked_fill_(mask, mask_value)

        if self.static_mask is not None:
            dots.masked_fill_(~self.static_mask[offset:offset + n, :offset + n], mask_value)

        attn = torch.softmax(dots, dim=-1)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = out.transpose(1, 2).contiguous().view(b, n, -1)
        out =  self.to_out(out)
        return out


class LayerScale(nn.Module):
    def __init__(self, dim, depth, fn):
        super().__init__()
        if depth <= 18:
            init_eps = 0.1
        elif depth > 18 and depth <= 24:
            init_eps = 1e-5
        else:
            init_eps = 1e-6

        scale = torch.zeros(1, 1, dim).fill_(init_eps)
        self.scale = nn.Parameter(scale)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale


class PreNorm(nn.Module):
    def __init__(self, dim, fn, sandwich=False):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.norm_out = nn.LayerNorm(dim) if sandwich else nn.Identity()
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        x = self.fn(x, **kwargs)
        return self.norm_out(x)


class SequentialSequence(nn.Module):
    def __init__(self, layers, args_route = {}, layer_dropout = 0.):
        super().__init__()
        assert all(len(route) == len(layers) for route in args_route.values()), 'each argument route map must have the same depth as the number of sequential layers'
        self.layers = layers
        self.args_route = args_route
        self.layer_dropout = layer_dropout

    def forward(self, x, **kwargs):
        def route_args(router, args, depth):
            routed_args = [(dict(), dict()) for _ in range(depth)]
            matched_keys = [key for key in args.keys() if key in router]

            for key in matched_keys:
                val = args[key]
                for depth, ((f_args, g_args), routes) in enumerate(zip(routed_args, router[key])):
                    new_f_args, new_g_args = map(lambda route: ({key: val} if route else {}), routes)
                    routed_args[depth] = ({**f_args, **new_f_args}, {**g_args, **new_g_args})
            return routed_args
        
        args = route_args(self.args_route, kwargs, len(self.layers))
        layers_and_args = list(zip(self.layers, args))

        for (f, g), (f_args, g_args) in layers_and_args:
            x = x + f(x, **f_args)
            x = x + g(x, **g_args)
        return x


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, dropout = 0., mult = 4.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x, cache=None, cache_key=None):
        return self.net(x)


class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        seq_len,
        causal=True,
        heads=8,
        dim_head=64,
        ff_mult=4,
        attn_dropout=0.0,
        ff_dropout=0.0,
        image_fmap_size=None,
        shift_tokens=False,
        rotary_emb=True,
    ):
        super().__init__()
        layers = nn.ModuleList()
        for ind in range(depth):
            attn = Attention(
                dim, causal=causal, seq_len=seq_len, heads=heads,
                dim_head=dim_head, dropout=attn_dropout
            )
            ff = FeedForward(dim, mult=ff_mult, dropout=ff_dropout)

            if shift_tokens:
                attn = PreShiftToken(attn, image_size=image_fmap_size, seq_len=seq_len)
                ff = PreShiftToken(ff, image_size=image_fmap_size, seq_len=seq_len)

            layers.append(nn.ModuleList([
                LayerScale(dim, ind + 1, PreNorm(dim, attn)),
                LayerScale(dim, ind + 1, PreNorm(dim, ff)),
            ]))

        route_attn = ((True, False),) * depth
        route_all = ((True, True),) * depth
        attn_route_map = {
            "mask": route_attn,
            "rotary_pos_emb": route_attn,
            "cache": route_all,
        }

        self.layers = SequentialSequence(layers, args_route=attn_route_map)

        pos_emb = None
        if rotary_emb:
            rot_dim = dim_head // 3
            img_seq_len = image_fmap_size ** 2
            text_len = seq_len - img_seq_len + 1

            text_pos_emb = RotaryEmbedding(dim=rot_dim)
            img_axial_pos_emb = RotaryEmbedding(dim=rot_dim, freqs_for='pixel')

            text_freqs = text_pos_emb(torch.arange(text_len))
            img_to_text_freqs = text_pos_emb(torch.full((img_seq_len,), 8192))
            text_freqs = torch.cat((text_freqs, img_to_text_freqs), dim=0)

            img_freqs_axial = img_axial_pos_emb(torch.linspace(-1, 1, steps=image_fmap_size))
            img_freqs = torch.cat([
                img_freqs_axial.unsqueeze(1).expand(-1, image_fmap_size, -1),
                img_freqs_axial.unsqueeze(0).expand(image_fmap_size, -1, -1)
            ], dim=-1)
            img_freqs = img_freqs.view(-1, img_freqs.size(-1))

            text_axial_freqs = img_axial_pos_emb(torch.full((text_len,), -10.0))
            text_axial_freqs = torch.cat((text_axial_freqs, text_axial_freqs), dim=-1)
            img_freqs = torch.cat((text_axial_freqs, img_freqs), dim=0)

            pos_emb = torch.cat((text_freqs, img_freqs), dim=-1)
            pos_emb = pos_emb.unsqueeze(0)

        self.register_buffer('pos_emb', pos_emb)

    def forward(self, x, **kwargs):
        return self.layers(x, rotary_pos_emb=self.pos_emb, **kwargs)


class DALLEConfig(PretrainedConfig):
    """
    Configuration class for DALL-E model.
    """
    model_type = "dalle"
    
    def __init__(
        self,
        image_size = 512,           # image size
        text_seq_len = 24,          # text sequence length
        num_text_tokens = 10000,    # vocab size for text
        dim: int = 512,             # model dimension
        depth = 4,                  # should aim to be 64 (however 4 is good enough)
        heads = 16,                 # attention heads
        dim_head = 64,              # attention head dimension
        attn_dropout = 0.1,         # attention dropout
        ff_dropout = 0.1,           # feedforward dropout
        loss_img_weight = 7,        # weight for image loss
        **kwargs
    ):
        self.image_size = image_size
        self.text_seq_len = text_seq_len
        self.num_text_tokens = num_text_tokens
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.dim_head = dim_head
        self.attn_dropout = attn_dropout
        self.ff_dropout = ff_dropout
        self.loss_img_weight = loss_img_weight
        
        super().__init__(**kwargs)


class DALLEModel(PreTrainedModel):
    """
    Varient of DALLE-pytorch.
    https://github.com/lucidrains/DALLE-pytorch
    """
    config_class = DALLEConfig
    base_model_prefix = "dalle"

    def __init__(self, config: DALLEConfig):
        super().__init__(config)

        # Store config parameters
        self.config = config
        self.image_size = config.image_size
        self.text_seq_len = config.text_seq_len
        self.num_text_tokens = config.num_text_tokens
        self.dim = config.dim
        self.depth = config.depth
        self.heads = config.heads
        self.dim_head = config.dim_head
        self.attn_dropout = config.attn_dropout
        self.ff_dropout = config.ff_dropout
        self.loss_img_weight = config.loss_img_weight

        # Pretrained VAE
        self.vae = OpenAIDiscreteVAE()
        self.vae.requires_grad_(False)

        # Image tokens and sequence length
        self.image_fmap_size = (self.image_size // (2 ** self.vae.num_layers))
        self.image_seq_len = self.image_fmap_size ** 2

        num_text_tokens = self.num_text_tokens + self.text_seq_len

        class always():
            def __init__(self, val):
                self.val = val
            def __call__(self, x, *args, **kwargs):
                return self.val

        self.text_pos_emb = always(0)
        self.image_pos_emb = always(0)

        self.num_text_tokens = num_text_tokens
        self.num_image_tokens = self.vae.num_tokens

        self.total_seq_len = self.text_seq_len + self.image_seq_len
        self.total_tokens = self.num_text_tokens + self.num_image_tokens

        self.transformer = Transformer(
            dim = self.dim,
            seq_len=self.total_seq_len,
            depth = self.depth,
            heads = self.heads,
            dim_head = self.dim_head,
            attn_dropout = self.attn_dropout,
            ff_dropout = self.ff_dropout,
            image_fmap_size = self.image_fmap_size,
            shift_tokens = True,
            rotary_emb = True,
        )

        self.to_logits = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.total_tokens),
        )

        self.text_emb = nn.Embedding(self.num_text_tokens, self.dim)
        self.image_emb = nn.Embedding(self.num_image_tokens, self.dim)

        seq_range = torch.arange(self.total_seq_len)
        logits_range = torch.arange(self.total_tokens)

        seq_range = seq_range.unsqueeze(0).unsqueeze(2)  # (1, n, 1)
        logits_range = logits_range.unsqueeze(0).unsqueeze(0)  # (1, 1, d)

        logits_mask = (
            ((seq_range >= self.text_seq_len) & (logits_range < self.num_text_tokens)) |
            ((seq_range < self.text_seq_len) & (logits_range >= self.num_text_tokens))
        )

        self.register_buffer('logits_mask', logits_mask, persistent=False)

    def forward(
        self,
        inputs: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        **kwargs
    ):
        # Normalize pixel values to [0, 1]
        image = (pixel_values + 1) / 2

        if image.size(2) != self.image_size or image.size(3) != self.image_size:
            image = F.interpolate(
                image,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )

        # Numbering tokenizer
        hist_len = int(self.text_seq_len / inputs.size(1))
        assert hist_len > 0, \
            f"text_seq_len ({self.text_seq_len}) must be >= number of sensors {inputs.size(1)}"

        # Convert to text ids
        text_ids = inputs[..., :hist_len].contiguous().view(inputs.size(0), -1)
        text_ids = (text_ids * 1E4).long()
        text_ids = text_ids.clamp(0, self.num_text_tokens - 1) + 1 # ensure within vocab range, reserve 0 for padding

        # Convert to image ids
        image_ids = self.vae.get_codebook_indices(image)

        # Forward pass
        logits = self._forward(text_ids, image_ids)
        labels = torch.cat((text_ids, image_ids + self.num_text_tokens), dim = 1)

        # Compute loss
        logits = logits.transpose(1, 2)
        loss_text = F.cross_entropy(logits[:, :, :self.text_seq_len], labels[:, :self.text_seq_len])
        loss_img = F.cross_entropy(logits[:, :, self.text_seq_len:], labels[:, self.text_seq_len:])
        loss = (loss_text + self.loss_img_weight * loss_img) / (self.loss_img_weight + 1)

        # Generate images during inference
        if not self.training:
            pred_images = self._generate_images(text_ids)
            # Convert back to [-1, 1] range and resize to original image size
            pred_images = (pred_images * 2) - 1

            if pred_images.size(2) != pixel_values.size(2) or pred_images.size(3) != pixel_values.size(3):
                pred_images = F.interpolate(
                    pred_images,
                    size=(pixel_values.size(2), pixel_values.size(3)),
                    mode="bilinear",
                    align_corners=False,
                )

        return {
            'outputs': None if self.training else pred_images,
            'loss': loss
        }

    @torch.no_grad()
    def _generate_images(
        self,
        text_ids,
        filter_thres=0.5,
        temperature=1.0,
    ):
        """
        Internal method for generating images during inference
        """
        vae, text_seq_len, image_seq_len, num_text_tokens = self.vae, self.text_seq_len, self.image_seq_len, self.num_text_tokens
        total_len = text_seq_len + image_seq_len

        text_ids = text_ids[:, :text_seq_len]  # make sure text is within bounds

        out = text_ids
        for cur_len in range(out.shape[1], total_len):
            is_image = cur_len >= text_seq_len

            text_part_ids, image_part_ids = out[:, :text_seq_len], out[:, text_seq_len:]

            logits = self._forward(text_part_ids, image_part_ids)
            logits = logits[:, -1, :]

            # Use torch.topk and torch.multinomial for sampling
            if filter_thres > 0:
                k = max(int((1 - filter_thres) * logits.shape[-1]), 1)
                topk_logits, topk_indices = torch.topk(logits, k, dim=-1)
                # Create a tensor with -inf for non-topk positions
                filtered_logits = torch.full_like(logits, float('-inf'))
                filtered_logits.scatter_(-1, topk_indices, topk_logits)
            else:
                filtered_logits = logits

            # Use torch.multinomial for sampling instead of gumbel sampling
            probs = F.softmax(filtered_logits / max(temperature, 1e-10), dim=-1)
            sample = torch.multinomial(probs, 1).squeeze(-1)

            # offset sampled token if it is an image token
            sample -= (num_text_tokens if is_image else 0)
            out = torch.cat((out, sample[:, None]), dim=-1)

        img_seq = out[:, -image_seq_len:]
        images = vae.decode(img_seq)

        return images

    def _forward(self, text_ids, image_ids):
        """
        Forward pass with conditional scaling for generation
        """
        device = text_ids.device

        # Text tokens
        text_ids = F.pad(text_ids, (1, 0), value=0)  # add <bos>
        tokens = self.text_emb(text_ids)
        tokens += self.text_pos_emb(torch.arange(text_ids.shape[1], device=device))

        # Image tokens (if any)
        if image_ids.shape[1] > 0:
            image_emb = self.image_emb(image_ids)
            image_emb += self.image_pos_emb(image_emb)
            tokens = torch.cat((tokens, image_emb), dim=1)

        # If the length exceeds the total text + image length
        # remove the last token, since it needs not to be trained
        seq_len = tokens.shape[1]
        if tokens.shape[1] > self.total_seq_len:
            seq_len = seq_len - 1
            tokens = tokens[:, :-1]

        # Send through transformer
        out = self.transformer(tokens)
        logits = self.to_logits(out)

        # Apply logits mask
        logits_mask = self.logits_mask[:, :seq_len]
        logits.masked_fill_(logits_mask, -torch.finfo(logits.dtype).max)

        return logits


# This version is original DALLE-pytorch implementation,
# but it has some errors because of the version mismatch.

# from dalle_pytorch import OpenAIDiscreteVAE, DALLE
# class DALLEModel(PreTrainedModel):
#     """
#     Wrapper class for DALLE-pytorch.
#     https://github.com/lucidrains/DALLE-pytorch
#     """
#     config_class = DALLEConfig
#     base_model_prefix = "dalle"

#     def __init__(self, config: DALLEConfig):
#         super().__init__(config)

#         # Store config parameters
#         self.config = config
#         self.text_seq_len = config.text_seq_len
#         self.num_text_tokens = config.num_text_tokens
#         self.dim = config.dim
#         self.depth = config.depth
#         self.heads = config.heads
#         self.dim_head = config.dim_head
#         self.attn_dropout = config.attn_dropout
#         self.ff_dropout = config.ff_dropout

#         # # Numbering tokenizer is more useful for our case
#         # self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
#         # self.tokenizer.pad_token = self.tokenizer.eos_token

#         # Pretrained VAE on 256x256 images
#         self.vae = OpenAIDiscreteVAE()

#         # Initialize DALL-E model
#         self.dalle = DALLE(
#             dim = self.dim,
#             vae = self.vae,
#             num_text_tokens = self.num_text_tokens,
#             text_seq_len = self.text_seq_len,
#             depth = self.depth,
#             heads = self.heads,
#             dim_head = self.dim_head,
#             attn_dropout = self.attn_dropout,
#             ff_dropout = self.ff_dropout
#         )

#     def forward(
#         self,
#         inputs: torch.Tensor,
#         pixel_values: Optional[torch.Tensor] = None,
#         **kwargs
#     ) -> Union[Tuple[torch.Tensor], dict]:

#         # Normalize pixel values to [0, 1]
#         pixel_values = (pixel_values + 1) / 2

#         # Because pretrained VAE provided is only trained on 256x256 images
#         # we need to downsample to the desired size
#         pixel_values = F.interpolate(pixel_values, size=(256, 256), mode='bilinear', align_corners=False)

#         # Numbering tokenizer
#         text_seq_len = 24
#         hist_len = int(text_seq_len / inputs.size(1))

#         assert hist_len > 0, f"text_seq_len ({text_seq_len}) must be >= number of sensors {inputs.size(1)}"

#         text = inputs[..., :hist_len].contiguous().view(inputs.size(0), -1)
#         text = (text * 1E4).long()

#         # Inference
#         loss = self.dalle(text, pixel_values, return_loss = True)

#         # Image will be None if training to promote efficiency
#         if self.dalle.training:
#             image = None
#         else:
#             image = self.dalle.generate_images(text)
#             image = (image * 2) - 1
#             image = F.interpolate(
#                 image,
#                 size=(pixel_values.size(2), pixel_values.size(3)),
#                 mode="bilinear",
#                 align_corners=False,
#             )

#         return {
#             'outputs': image,
#             'loss': loss
#         }
