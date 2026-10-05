# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math
from functools import partial

import torch
import torch.nn as nn

from lmwm.models import modules
from lmwm.models.modules import (
    MLP,
    Attention,
    DropPath,
    RoPEAttention,
    build_action_block_causal_attention_mask,
)
import logging

from lmwm.models.tensors import trunc_normal_

logger = logging.getLogger(__name__)

BLOCK_SIZE = 64


class FWAdaLNBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        wide_silu=True,
        norm_layer=nn.LayerNorm,
        use_sdpa=True,
        is_causal=False,
        grid_size=16,
        use_rope=False,
        **kwargs,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.grid_size = grid_size
        if use_rope:
            self.attn = RoPEAttention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                use_sdpa=use_sdpa,
                is_causal=is_causal,
                grid_size=grid_size,
                proj_drop=drop,
            )
        else:
            self.attn = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                use_sdpa=use_sdpa,
                is_causal=is_causal,
                proj_drop=drop,
            )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if act_layer is nn.SiLU:
            self.mlp = modules.SwiGLUFFN(
                in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, wide_silu=wide_silu, drop=drop
            )
        else:
            self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))

    def forward(self, x, z, mask=None, attn_mask=None, T=None, H_patches=None, W_patches=None, cond_tokens=0):
        """
        Input:
            x : B, N, C with N = T*H*W
            z : B, T, D
        Returns:
            B, N, D
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(z).repeat_interleave((self.grid_size**2 + cond_tokens), dim=1).chunk(6, dim=2)
        )
        if isinstance(self.attn, RoPEAttention):
            y = self.attn(
                modulate(self.norm1(x), shift_msa, scale_msa),
                mask=mask,
                attn_mask=attn_mask,
                T=T,
                H=H_patches,
                W=W_patches,
                action_tokens=cond_tokens,
            )
        else:
            y = self.attn(
                modulate(self.norm1(x), shift_msa, scale_msa),
                mask=mask,
                attn_mask=attn_mask,
            )
        x = x + self.drop_path(y * gate_msa)
        x = x + self.drop_path(gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp)))
        return x


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class VisionTransformerAdaLN(nn.Module):
    """Vision Transformer"""

    def __init__(
        self,
        img_size=(224, 224),
        patch_size=16,
        num_frames=1,
        tubelet_size=2,
        embed_dim=768,
        predictor_embed_dim=384,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        use_silu=False,
        wide_silu=True,
        is_causal=False,
        use_activation_checkpointing=False,
        local_window=(-1, -1, -1),
        use_rope=True,
        action_dim=20,
        proprio_dim=10,
        use_proprio=True,
        act_mlp=False,
        prop_mlp=False,
        init_scale_factor_adaln=10,
        # AdaLN-predictor specific
        proprio_encoding="feature",  # 'feature' or 'token'
        proprio_emb_dim=0,  # if proprio_encoding='feature', proprio_emb_dim>0 will be used to encode the proprio input
        proprio_encoder_inpred=True,
        proprio_tokens=0,  # if proprio_encoding='token', proprio_tokens>0 will be used to encode the proprio input
        action_encoder_inpred=True,
        action_norm=False,
        action_conditioning="adaln",  # "adaln" | "token" | "both"
        # Language conditioning
        use_language=False,
        language_dim=768,  # T5-base output dimension
        language_fusion="add",  # "add" or "gate"
        language_norm=False,
        **kwargs,
    ):
        super().__init__()
        self.attn_depth, self.attn_height, self.attn_width = local_window
        self.predictor_embed_dim = predictor_embed_dim
        self.proprio_encoder_inpred = proprio_encoder_inpred
        self.action_encoder_inpred = action_encoder_inpred
        self.action_norm = action_norm
        self.action_conditioning = action_conditioning
        
        # Language conditioning
        self.use_language = use_language
        self.language_dim = language_dim
        self.language_fusion = language_fusion
        self.language_norm = language_norm
        self.record_z = False
        self._last_z = None

        # Map input to predictor dimension
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)

        # Determine positional embedding
        if type(img_size) is int:
            img_size = (img_size, img_size)
        self.img_height, self.img_width = img_size
        self.patch_size = patch_size
        # --
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.is_video = num_frames > 1

        self.grid_height = img_size[0] // self.patch_size
        self.grid_width = img_size[1] // self.patch_size
        self.grid_depth = num_frames // self.tubelet_size
        self.use_activation_checkpointing = use_activation_checkpointing

        self.action_dim = action_dim
        self.proprio_dim = proprio_dim
        self.proprio_emb_dim = proprio_emb_dim
        self.proprio_tokens = proprio_tokens
        self.use_proprio = use_proprio
        self.proprio_encoding = proprio_encoding
        self.act_mlp = act_mlp
        self.prop_mlp = prop_mlp

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        # Embed proprio and action
        if self.use_proprio and self.proprio_encoding == "feature":
            self.predictor_total_embed_dim = predictor_embed_dim + proprio_emb_dim
        else:
            self.predictor_total_embed_dim = predictor_embed_dim

        # Initialize encoders
        if self.action_encoder_inpred:
            self.action_encoder = nn.Linear(action_dim, self.predictor_total_embed_dim, bias=True)
        if self.action_norm:
            self.action_norm_layer = nn.LayerNorm(self.predictor_total_embed_dim)
        if self.proprio_encoder_inpred:
            if self.proprio_encoding == "token" and self.proprio_tokens > 0:
                self.proprio_encoder = nn.Linear(proprio_dim, predictor_embed_dim, bias=True)
            elif self.proprio_encoding == "feature" and self.proprio_emb_dim > 0:
                self.proprio_encoder = nn.Linear(proprio_dim, proprio_emb_dim, bias=True)
        
        # Language encoder (projects language embedding to conditioning dimension)
        if self.use_language:
            self.language_encoder = nn.Linear(language_dim, self.predictor_total_embed_dim, bias=True)
            if self.language_norm:
                self.language_norm_layer = nn.LayerNorm(self.predictor_total_embed_dim)
            if self.language_fusion == "gate":
                # Gating mechanism: learn how much language vs action to use
                self.language_gate = nn.Sequential(
                    nn.Linear(self.predictor_total_embed_dim * 2, self.predictor_total_embed_dim),
                    nn.Sigmoid(),
                )

        # ── Attention Blocks ─────────────────────────────────────────────
        # FIX: pass is_causal=False to attention layers — causality is
        # enforced by the explicit block-causal attn_mask we construct
        # below.  Passing is_causal=True together with an explicit
        # attn_mask would cause a PyTorch RuntimeError.
        self.use_rope = use_rope
        self.predictor_blocks = nn.ModuleList(
            [
                FWAdaLNBlock(
                    use_rope=use_rope,
                    grid_size=self.grid_height,
                    grid_depth=self.grid_depth,
                    dim=self.predictor_total_embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    act_layer=nn.SiLU if use_silu else nn.GELU,
                    is_causal=False,  # FIX: was is_causal — use explicit mask instead
                    wide_silu=wide_silu,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        # Normalize & project back to input dimension
        self.predictor_norm = norm_layer(self.predictor_total_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)

        # ── Block-causal attention mask ──────────────────────────────────
        # FIX: ALWAYS build the block-causal mask when is_causal=True,
        # regardless of local_window.  The old code gated mask construction
        # on local_window > 0, which defaulted to (-1,-1,-1) and was never
        # set — so the mask was never built and attention was fully
        # bidirectional, letting the predictor cheat by copying future
        # frames during teacher-forcing.
        self.cond_tokens = 0
        if is_causal:
            if action_conditioning in ("token", "both"):
                self.cond_tokens += 1  # one action token per timestep
            if self.proprio_tokens > 0 and self.proprio_encoding == "token":
                self.cond_tokens += 1
            attn_mask = build_action_block_causal_attention_mask(
                num_frames // tubelet_size,
                self.grid_height,
                self.grid_width,
                add_tokens=self.cond_tokens,
            )
            self.attn_mask = attn_mask
            logger.info(
                f"Built block-causal mask: {attn_mask.shape} "
                f"(T={num_frames // tubelet_size}, "
                f"grid={self.grid_height}x{self.grid_width}, "
                f"cond_tokens={self.cond_tokens})"
            )
        else:
            self.attn_mask = None

        # ------ initialize weights
        self.init_std = init_std
        self.init_scale_factor_adaln = init_scale_factor_adaln
        self.apply(self._init_weights)
        self._rescale_blocks()

        # Initialize for better gradient flow
        with torch.no_grad():
            for block in self.predictor_blocks:
                linear_layer = block.adaLN_modulation[1]
                if self.init_scale_factor_adaln == 0:
                    nn.init.constant_(linear_layer.weight, 0)
                else:
                    trunc_normal_(linear_layer.weight, std=self.init_std * self.init_scale_factor_adaln)
                nn.init.constant_(linear_layer.bias, 0)
            # Log once after initializing all blocks
            if self.init_scale_factor_adaln == 0:
                logger.info(f"🔧 Initialized {len(self.predictor_blocks)} AdaLN-zero blocks")
            else:
                logger.info(
                    f"🔧 Initialized {len(self.predictor_blocks)} AdaLN blocks (scale_factor={self.init_scale_factor_adaln})"
                )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def concat_obs(self, z_vis, proprio):
        """
        input :
            z_vis: B T H*W D
            proprio: B T 1 P
        output:    z (tensor): (b, num_frames, num_patches, emb_dim)
        """
        z = torch.cat([z_vis, proprio], dim=3)
        return z

    def forward(self, x, actions, proprio=None, language=None, override_z=None):
        """
        Input:
            x: B T V H W D
            actions: B T A
            proprio: B T P
            language: B L (language embedding, where L is language_dim)
        Returns:
            x: B T H*W D
            proprio: B T 1 P (P=D if proprio_encoding='token' else P=proprio_dim)
        """
        # Map context tokens to pedictor dimensions
        x = self.predictor_embed(x)
        x = x.flatten(2, 4)  # [B, T, H*W, D]
        B, T, N, D = x.shape

        # Encode action once — reused for AdaLN conditioning and/or as a sequence token
        if self.action_encoder_inpred:
            act_encoded = self.action_encoder(actions)   # (B, T, D)
        else:
            act_encoded = actions.squeeze(2)             # (B, T, 1, A) -> (B, T, A)
        if self.action_norm:
            act_encoded = self.action_norm_layer(act_encoded)

        # Build z (AdaLN conditioning signal)
        # "adaln" / "both": action contributes to z; "token": action is a sequence token only
        if self.action_conditioning in ("adaln", "both"):
            z = act_encoded
        else:  # "token"
            z = torch.zeros_like(act_encoded)  # language will fill this below

        # Fuse language into z (all modes)
        if self.use_language and language is not None:
            lang_emb = self.language_encoder(language)          # (B, language_dim) -> (B, D)
            if self.language_norm:
                lang_emb = self.language_norm_layer(lang_emb)
            lang_emb = lang_emb.unsqueeze(1).expand(-1, T, -1)  # (B, D) -> (B, T, D)
            if self.language_fusion == "add":
                z = z + lang_emb
            elif self.language_fusion == "gate":
                gate = self.language_gate(torch.cat([z, lang_emb], dim=-1))
                z = gate * z + (1 - gate) * lang_emb

        if override_z is not None:
            z = override_z

        if self.record_z:
            self._last_z = z
            if z.requires_grad:
                z.retain_grad()

        # Build the token sequence x, optionally prepending action token
        if self.action_conditioning in ("token", "both"):
            act_tok = act_encoded.unsqueeze(2)  # (B, T, 1, D)
            if self.use_proprio and proprio is not None:
                if self.proprio_encoder_inpred:
                    proprio = self.proprio_encoder(proprio).unsqueeze(2)
                if self.proprio_encoding == "token":
                    # order: [action, proprio, visual patches]
                    x = torch.cat([act_tok, proprio, x], dim=2).flatten(1, 2)
                elif self.proprio_encoding == "feature":
                    x = torch.cat([act_tok, self.concat_obs(x, proprio)], dim=2).flatten(1, 2)
            else:
                x = torch.cat([act_tok, x], dim=2).flatten(1, 2)  # [B, T*(1+H*W), D]
        else:  # "adaln": action only in z, no action token in sequence
            if self.use_proprio and proprio is not None:
                if self.proprio_encoder_inpred:
                    proprio = self.proprio_encoder(proprio).unsqueeze(2)
                if self.proprio_encoding == "token":
                    x = torch.cat([proprio, x], dim=2).flatten(1, 2)
                elif self.proprio_encoding == "feature":
                    x = self.concat_obs(x, proprio).flatten(1, 2)
            else:
                x = x.flatten(1, 2)  # [B, T*(H*W), D]
        attn_mask = (
            self.attn_mask[: x.size(1), : x.size(1)].to(x.device, non_blocking=True)
            if self.attn_mask is not None
            else None
        )

        # Fwd prop
        for i, blk in enumerate(self.predictor_blocks):
            if self.use_activation_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    blk,
                    x,
                    z,
                    None,
                    attn_mask,
                    T=T,
                    H_patches=self.grid_height,
                    W_patches=self.grid_width,
                    use_reentrant=False,
                    cond_tokens=self.cond_tokens,
                )
            else:
                x = blk(
                    x,
                    z,
                    mask=None,
                    attn_mask=attn_mask,
                    T=T,
                    H_patches=self.grid_height,
                    W_patches=self.grid_width,
                    cond_tokens=self.cond_tokens,
                )
        x = self.predictor_norm(x)

        n_vis = self.grid_height * self.grid_width
        if self.use_proprio and proprio is not None:
            if self.proprio_encoding == "token":
                x = x.view(B, T, self.cond_tokens + n_vis, D)
                x, proprio_features = x[:, :, self.cond_tokens:, :], x[:, :, :self.cond_tokens, :]
            elif self.proprio_encoding == "feature":
                x = x.view(B, T, n_vis, self.predictor_total_embed_dim)
                x, proprio_features = x[:, :, :, :-self.proprio_emb_dim], x[:, :, :, -self.proprio_emb_dim:]
        else:
            if self.action_conditioning in ("token", "both"):
                # Strip the prepended action token(s) before projecting
                x = x.view(B, T, self.cond_tokens + n_vis, self.predictor_total_embed_dim)
                x = x[:, :, self.cond_tokens:, :]
            else:
                x = x.view(B, T, n_vis, self.predictor_total_embed_dim)
            proprio_features = None

        x = self.predictor_proj(x)
        return x, None, proprio_features


def vit_predictor_AdaLN(**kwargs):
    model = VisionTransformerAdaLN(qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model