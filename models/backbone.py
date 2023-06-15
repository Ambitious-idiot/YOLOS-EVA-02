from typing import Callable, Optional, Tuple, Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import math
from timm.layers import Mlp, GluMlp, SwiGLU, LayerNorm, DropPath, PatchDropout, RotaryEmbeddingCat, \
    apply_rot_embed_cat, apply_keep_indices_nlc, trunc_normal_, to_2tuple, use_fused_attn


__all__ = ['tiny', 'small', 'base', 'large']


class EvaAttention(nn.Module):
    fused_attn: torch.jit.Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = True,
            qkv_fused: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            attn_head_dim: Optional[int] = None,
            norm_layer: Optional[Callable] = None,
            det_num: int = 0
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = head_dim ** -0.5
        self.fused_attn = use_fused_attn()
        self.det_num = det_num

        if qkv_fused:
            self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
            self.q_proj = self.k_proj = self.v_proj = None
            if qkv_bias:
                self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
                self.register_buffer('k_bias', torch.zeros(all_head_dim), persistent=False)
                self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
            else:
                self.q_bias = self.k_bias = self.v_bias = None
        else:
            self.q_proj = nn.Linear(dim, all_head_dim, bias=qkv_bias)
            self.k_proj = nn.Linear(dim, all_head_dim, bias=False)
            self.v_proj = nn.Linear(dim, all_head_dim, bias=qkv_bias)
            self.qkv = None
            self.q_bias = self.k_bias = self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.norm = norm_layer(all_head_dim) if norm_layer is not None else nn.Identity()
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
            self,
            x,
            rope: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        B, N, C = x.shape

        if self.qkv is not None:
            qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
            qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
            qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
        else:
            q = self.q_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)  # B, num_heads, N, C
            k = self.k_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)
            v = self.v_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)

        if rope is not None:

            q = torch.cat([q[:, :, :1, :], apply_rot_embed_cat(q[:, :, 1:-self.det_num, :], rope),
                           q[:, :, -self.det_num:, :]], 2).type_as(v)
            k = torch.cat([k[:, :, :1, :], apply_rot_embed_cat(k[:, :, 1:-self.det_num, :], rope),
                           k[:, :, -self.det_num:, :]], 2).type_as(v)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))
            attn = attn.softmax(dim=-1)
            if attn_mask is not None:
                attn_mask = attn_mask.to(torch.bool)
                attn = attn.masked_fill(~attn_mask[:, None, None, :], float("-inf"))
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class EvaBlock(nn.Module):

    def __init__(
            self,
            dim: int,
            num_heads: int,
            qkv_bias: bool = True,
            qkv_fused: bool = True,
            mlp_ratio: float = 4.,
            swiglu_mlp: bool = False,
            scale_mlp: bool = False,
            scale_attn_inner: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float = 0.,
            init_values: Optional[float] = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            attn_head_dim: Optional[int] = None,
            det_num: int = 0
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = EvaAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qkv_fused=qkv_fused,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            attn_head_dim=attn_head_dim,
            norm_layer=norm_layer if scale_attn_inner else None,
            det_num=det_num,
        )
        self.gamma_1 = nn.Parameter(init_values * torch.ones(dim)) if init_values is not None else None
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        hidden_features = int(dim * mlp_ratio)
        if swiglu_mlp:
            if scale_mlp:
                # when norm in SwiGLU used, an impl with separate fc for gate & x is used
                self.mlp = SwiGLU(
                    in_features=dim,
                    hidden_features=hidden_features,
                    norm_layer=norm_layer if scale_mlp else None,
                    drop=proj_drop,
                )
            else:
                # w/o any extra norm, an impl with packed weights is used, matches existing GluMLP
                self.mlp = GluMlp(
                    in_features=dim,
                    hidden_features=hidden_features * 2,
                    norm_layer=norm_layer if scale_mlp else None,
                    act_layer=nn.SiLU,
                    gate_last=False,
                    drop=proj_drop,
                )
        else:
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=hidden_features,
                act_layer=act_layer,
                norm_layer=norm_layer if scale_mlp else None,
                drop=proj_drop,
            )
        self.gamma_2 = nn.Parameter(init_values * torch.ones(dim)) if init_values is not None else None
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, rope: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None):
        if self.gamma_1 is None:
            x = x + self.drop_path1(self.attn(self.norm1(x), rope=rope, attn_mask=attn_mask))
            x = x + self.drop_path2(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), rope=rope, attn_mask=attn_mask))
            x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class EvaBlockPostNorm(nn.Module):
    """ EVA block w/ post-norm and support for swiglu, MLP norm scale, ROPE. """
    def __init__(
            self,
            dim: int,
            num_heads: int,
            qkv_bias: bool = True,
            qkv_fused: bool = True,
            mlp_ratio: float = 4.,
            swiglu_mlp: bool = False,
            scale_mlp: bool = False,
            scale_attn_inner: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float = 0.,
            init_values: Optional[float] = None,  # ignore for post-norm
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = nn.LayerNorm,
            attn_head_dim: Optional[int] = None,
            det_num: int = 0
    ):
        super().__init__()
        self.attn = EvaAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qkv_fused=qkv_fused,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            attn_head_dim=attn_head_dim,
            norm_layer=norm_layer if scale_attn_inner else None,
            det_num=det_num
        )
        self.norm1 = norm_layer(dim)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        hidden_features = int(dim * mlp_ratio)
        if swiglu_mlp:
            if scale_mlp:
                # when norm in SwiGLU used, an impl with separate fc for gate & x is used
                self.mlp = SwiGLU(
                    in_features=dim,
                    hidden_features=hidden_features,
                    norm_layer=norm_layer if scale_mlp else None,
                    drop=proj_drop,
                )
            else:
                # w/o any extra norm, an impl with packed fc1 weights is used, matches existing GluMLP
                self.mlp = GluMlp(
                    in_features=dim,
                    hidden_features=hidden_features * 2,
                    norm_layer=norm_layer if scale_mlp else None,
                    act_layer=nn.SiLU,
                    gate_last=False,
                    drop=proj_drop,
                )
        else:
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=hidden_features,
                act_layer=act_layer,
                norm_layer=norm_layer if scale_mlp else None,
                drop=proj_drop,
            )
        self.norm2 = norm_layer(dim)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, rope: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None):
        x = x + self.drop_path1(self.norm1(self.attn(x, rope=rope, attn_mask=attn_mask)))
        x = x + self.drop_path2(self.norm2(self.mlp(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.grid_size = tuple([s // p for s, p in zip(self.img_size, self.patch_size)])
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class Eva(nn.Module):
    """ Eva Vision Transformer w/ Abs & Rotary Pos Embed

    This class implements the EVA and EVA02 models that were based on the BEiT ViT variant
      * EVA - abs pos embed, global avg pool
      * EVA02 - abs + rope pos embed, global avg pool, SwiGLU, scale Norm in MLP (ala normformer)
    """

    def __init__(
            self,
            img_size: Union[int, Tuple[int, int]] = 224,
            patch_size: Union[int, Tuple[int, int]] = 16,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: str = 'avg',
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            qkv_bias: bool = True,
            qkv_fused: bool = True,
            mlp_ratio: float = 4.,
            swiglu_mlp: bool = False,
            scale_mlp: bool = False,
            scale_attn_inner: bool = False,
            drop_rate: float = 0.,
            pos_drop_rate: float = 0.,
            patch_drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            norm_layer: Callable = LayerNorm,
            init_values: Optional[float] = None,
            class_token: bool = True,
            use_abs_pos_emb: bool = True,
            use_rot_pos_emb: bool = False,
            use_post_norm: bool = False,
            ref_feat_shape: Optional[Union[Tuple[int, int], int]] = None,
            head_init_scale: float = 0.001,
            det_num: int = 100
    ):
        super().__init__()

        if isinstance(img_size, tuple):
            self.img_size = img_size
        else:
            self.img_size = to_2tuple(img_size)

        self.depth = depth
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_prefix_tokens = 1 if class_token else 0
        self.grad_checkpointing = False

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None

        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + self.num_prefix_tokens, embed_dim)) if use_abs_pos_emb else None
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(
                patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
                return_indices=True,
            )
        else:
            self.patch_drop = None

        if use_rot_pos_emb:
            ref_feat_shape = to_2tuple(ref_feat_shape) if ref_feat_shape is not None else None
            self.rope = RotaryEmbeddingCat(
                embed_dim // num_heads,
                in_pixels=False,
                feat_shape=None,
                ref_feat_shape=ref_feat_shape,
            )
        else:
            self.rope = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        block_fn = EvaBlockPostNorm if use_post_norm else EvaBlock
        self.blocks = nn.ModuleList([
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qkv_fused=qkv_fused,
                mlp_ratio=mlp_ratio,
                swiglu_mlp=swiglu_mlp,
                scale_mlp=scale_mlp,
                scale_attn_inner=scale_attn_inner,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                init_values=init_values,
                det_num=det_num,
            )
            for i in range(depth)])

        use_fc_norm = self.global_pool == 'avg'
        self.norm = norm_layer(embed_dim)
        # self.norm = nn.Identity() if use_fc_norm else norm_layer(embed_dim)

        # Classifier head
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if isinstance(self.head, nn.Linear):
            trunc_normal_(self.head.weight, std=.02)
            self.head.weight.data.mul_(head_init_scale)
            self.head.bias.data.mul_(head_init_scale)

        self.apply(self._init_weights)
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)

        self.fix_init_weight()

        # set finetune flag
        self.has_mid_pe = False

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def finetune_det(self, img_size=[800, 1344], det_token_num=100, mid_pe_size=None, use_checkpoint=False):
        g = math.pow(self.pos_embed.size(1) - 1, 0.5)
        if int(g) - g != 0:
            self.pos_embed = torch.nn.Parameter(self.pos_embed[:, 1:, :])

        self.det_token_num = det_token_num
        self.det_token = nn.Parameter(torch.zeros(1, det_token_num, self.embed_dim))
        self.det_token = trunc_normal_(self.det_token, std=.02)
        cls_pos_embed = self.pos_embed[:, 0, :]
        cls_pos_embed = cls_pos_embed[:, None]
        det_pos_embed = torch.zeros(1, det_token_num, self.embed_dim)
        det_pos_embed = trunc_normal_(det_pos_embed, std=.02)
        patch_pos_embed = self.pos_embed[:, 1:, :]
        patch_pos_embed = patch_pos_embed.transpose(1, 2)
        B, E, Q = patch_pos_embed.shape
        P_H, P_W = self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size
        patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)
        H, W = img_size
        new_P_H, new_P_W = H // self.patch_size, W // self.patch_size
        patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H, new_P_W), mode='bicubic',
                                                    align_corners=False)
        patch_pos_embed = patch_pos_embed.flatten(2).transpose(1, 2)
        self.pos_embed = torch.nn.Parameter(torch.cat((cls_pos_embed, patch_pos_embed, det_pos_embed), dim=1))
        self.img_size = img_size
        if mid_pe_size == None:
            self.has_mid_pe = False
            print('No mid pe')
        else:
            print('Has mid pe')
            self.mid_pos_embed = nn.Parameter(
                torch.zeros(self.depth - 1, 1, 1 + (mid_pe_size[0] * mid_pe_size[1] // self.patch_size ** 2) + 100,
                            self.embed_dim))
            trunc_normal_(self.mid_pos_embed, std=.02)
            self.has_mid_pe = True
            self.mid_pe_size = mid_pe_size
        self.use_checkpoint = use_checkpoint


    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'det_token'}

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(
            stem=r'^cls_token|pos_embed|patch_embed',  # stem and embed
            blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))],
        )
        return matcher

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def InterpolateInitPosEmbed(self, pos_embed, img_size=(800, 1344)):
        cls_pos_embed = pos_embed[:, 0, :]
        cls_pos_embed = cls_pos_embed[:, None]
        det_pos_embed = pos_embed[:, -self.det_token_num:, :]
        patch_pos_embed = pos_embed[:, 1:-self.det_token_num, :]
        patch_pos_embed = patch_pos_embed.transpose(1, 2)
        B, E, Q = patch_pos_embed.shape

        P_H, P_W = self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size
        patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)

        # P_H, P_W = self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size
        # patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)

        H, W = img_size
        new_P_H, new_P_W = H // self.patch_size, W // self.patch_size
        patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H, new_P_W), mode='bicubic',
                                                    align_corners=False)
        patch_pos_embed = patch_pos_embed.flatten(2).transpose(1, 2)
        scale_pos_embed = torch.cat((cls_pos_embed, patch_pos_embed, det_pos_embed), dim=1)
        return scale_pos_embed

    def InterpolateMidPosEmbed(self, pos_embed, img_size=(800, 1344)):
        cls_pos_embed = pos_embed[:, :, 0, :]
        cls_pos_embed = cls_pos_embed[:, None]
        det_pos_embed = pos_embed[:, :, -self.det_token_num:, :]
        patch_pos_embed = pos_embed[:, :, 1:-self.det_token_num, :]
        patch_pos_embed = patch_pos_embed.transpose(2, 3)
        D, B, E, Q = patch_pos_embed.shape

        P_H, P_W = self.mid_pe_size[0] // self.patch_size, self.mid_pe_size[1] // self.patch_size
        patch_pos_embed = patch_pos_embed.view(D * B, E, P_H, P_W)
        H, W = img_size
        new_P_H, new_P_W = H // self.patch_size, W // self.patch_size
        patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H, new_P_W), mode='bicubic', align_corners=False)
        patch_pos_embed = patch_pos_embed.flatten(2).transpose(1, 2).contiguous().view(D, B, new_P_H * new_P_W, E)
        scale_pos_embed = torch.cat((cls_pos_embed, patch_pos_embed, det_pos_embed), dim=2)
        return scale_pos_embed

    def forward_features(self, x):
        B, H, W = x.shape[0], x.shape[2], x.shape[3]

        x = self.patch_embed(x)

        # interpolate init pe
        if (self.pos_embed.shape[1] - 1 - self.det_token_num) != x.shape[1]:
            temp_pos_embed = self.InterpolateInitPosEmbed(self.pos_embed, img_size=(H, W))
        else:
            temp_pos_embed = self.pos_embed
        # interpolate mid pe
        if self.has_mid_pe:
            # temp_mid_pos_embed = []
            if (self.mid_pos_embed.shape[2] - 1 - self.det_token_num) != x.shape[1]:
                temp_mid_pos_embed = self.InterpolateMidPosEmbed(self.mid_pos_embed, img_size=(H, W))
            else:
                temp_mid_pos_embed = self.mid_pos_embed

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        det_token = self.det_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x, det_token), dim=1)
        x = x + temp_pos_embed
        x = self.pos_drop(x)

        # obtain shared rotary position embedding and apply patch dropout
        grid_shape = tuple([s // p for s, p in zip((H, W), to_2tuple(self.patch_size))])
        rot_pos_embed = self.rope.get_embed(grid_shape) if self.rope is not None else None
        if self.patch_drop is not None:
            x, keep_indices = self.patch_drop(x)
            if rot_pos_embed is not None and keep_indices is not None:
                rot_pos_embed = apply_keep_indices_nlc(x, rot_pos_embed, keep_indices)

        for i, blk in enumerate(self.blocks):
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(blk, x, rope=rot_pos_embed)
            else:
                x = blk(x, rope=rot_pos_embed)
            if self.has_mid_pe:
                if i < (self.depth - 1):
                    x = x + temp_mid_pos_embed[i]
        x = self.norm(x)
        return x[:, -self.det_token_num:, :]

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.forward_head(x)
        return x


def resample_abs_pos_embed(
        posemb,
        new_size: List[int],
        old_size: Optional[List[int]] = None,
        num_prefix_tokens: int = 1,
        interpolation: str = 'bicubic'
):
    # sort out sizes, assume square if old size not provided
    num_pos_tokens = posemb.shape[1]
    num_new_tokens = new_size[0] * new_size[1] + num_prefix_tokens
    if num_new_tokens == num_pos_tokens and new_size[0] == new_size[1]:
        return posemb

    if not old_size:
        hw = int(math.sqrt(num_pos_tokens - num_prefix_tokens))
        old_size = hw, hw

    if num_prefix_tokens:
        posemb_prefix, posemb = posemb[:, :num_prefix_tokens], posemb[:, num_prefix_tokens:]
    else:
        posemb_prefix, posemb = None, posemb

    # do the interpolation
    embed_dim = posemb.shape[-1]
    posemb = posemb.reshape(1, old_size[0], old_size[1], -1).permute(0, 3, 1, 2)
    posemb = F.interpolate(posemb, size=new_size, mode=interpolation)
    posemb = posemb.permute(0, 2, 3, 1).reshape(1, -1, embed_dim)

    # add back extra (class, etc) prefix tokens
    if posemb_prefix is not None:
        posemb = torch.cat([posemb_prefix, posemb], dim=1)

    return posemb


def resample_patch_embed(
        patch_embed,
        new_size: List[int],
        interpolation: str = 'bicubic'
):
    """Resample the weights of the patch embedding kernel to target resolution.
    We resample the patch embedding kernel by approximately inverting the effect
    of patch resizing.

    Code based on:
      https://github.com/google-research/big_vision/blob/b00544b81f8694488d5f36295aeb7972f3755ffe/big_vision/models/proj/flexi/vit.py

    With this resizing, we can for example load a B/8 filter into a B/16 model
    and, on 2x larger input image, the result will match.

    Args:
        patch_embed: original parameter to be resized.
        new_size (tuple(int, int): target shape (height, width)-only.
        interpolation (str): interpolation for resize
        antialias (bool): use anti-aliasing filter in resize
        verbose (bool): log operation
    Returns:
        Resized patch embedding kernel.
    """
    import numpy as np

    assert len(patch_embed.shape) == 4, "Four dimensions expected"
    assert len(new_size) == 2, "New shape should only be hw"
    old_size = patch_embed.shape[-2:]
    if tuple(old_size) == tuple(new_size):
        return patch_embed

    def resize(x_np, _new_size):
        x_tf = torch.Tensor(x_np)[None, None, ...]
        x_upsampled = F.interpolate(
            x_tf, size=_new_size, mode=interpolation)[0, 0, ...].numpy()
        return x_upsampled

    def get_resize_mat(_old_size, _new_size):
        mat = []
        for i in range(np.prod(_old_size)):
            basis_vec = np.zeros(_old_size)
            basis_vec[np.unravel_index(i, _old_size)] = 1.
            mat.append(resize(basis_vec, _new_size).reshape(-1))
        return np.stack(mat).T

    resize_mat = get_resize_mat(old_size, new_size)
    resize_mat_pinv = torch.Tensor(np.linalg.pinv(resize_mat.T))

    def resample_kernel(kernel):
        resampled_kernel = resize_mat_pinv @ kernel.reshape(-1)
        return resampled_kernel.reshape(new_size)

    ret_mat = torch.zeros(patch_embed[0], patch_embed[1], *new_size)
    for i in range(patch_embed.shape[0]):
        for j in range(patch_embed.shape[1]):
            ret_mat[i, j] = resample_kernel(patch_embed[i, j])

    return ret_mat


def checkpoint_filter_fn(
        state_dict,
        model,
        interpolation='bicubic'
):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    state_dict = state_dict.get('model_ema', state_dict)
    state_dict = state_dict.get('model', state_dict)
    state_dict = state_dict.get('module', state_dict)
    state_dict = state_dict.get('state_dict', state_dict)
    # prefix for loading OpenCLIP compatible weights
    if 'visual.trunk.pos_embed' in state_dict:
        prefix = 'visual.trunk.'
    elif 'visual.pos_embed' in state_dict:
        prefix = 'visual.'
    else:
        prefix = ''
    mim_weights = prefix + 'mask_token' in state_dict
    no_qkv = prefix + 'blocks.0.attn.q_proj.weight' in state_dict

    len_prefix = len(prefix)
    for k, v in state_dict.items():
        if prefix:
            if k.startswith(prefix):
                k = k[len_prefix:]
            else:
                continue

        if 'rope' in k:
            continue

        if 'patch_embed.proj.weight' in k:
            _, _, H, W = model.patch_embed.proj.weight.shape
            if v.shape[-1] != W or v.shape[-2] != H:
                v = resample_patch_embed(
                    v,
                    (H, W),
                    interpolation=interpolation
                )
        elif k == 'pos_embed' and v.shape[1] != model.pos_embed.shape[1]:
            # To resize pos embedding when using model at different size from pretrained weights
            num_prefix_tokens = 0 if getattr(model, 'no_embed_class', False) else getattr(model, 'num_prefix_tokens', 1)
            v = resample_abs_pos_embed(
                v,
                new_size=model.patch_embed.grid_size,
                num_prefix_tokens=num_prefix_tokens,
                interpolation=interpolation
            )

        k = k.replace('mlp.ffn_ln', 'mlp.norm')
        k = k.replace('attn.inner_attn_ln', 'attn.norm')
        k = k.replace('mlp.w12', 'mlp.fc1')
        k = k.replace('mlp.w1', 'mlp.fc1_g')
        k = k.replace('mlp.w2', 'mlp.fc1_x')
        k = k.replace('mlp.w3', 'mlp.fc2')
        if no_qkv:
            k = k.replace('q_bias', 'q_proj.bias')
            k = k.replace('v_bias', 'v_proj.bias')

        if mim_weights and k in ('mask_token', 'lm_head.weight', 'lm_head.bias', 'norm.weight', 'norm.bias'):
            if k == 'norm.weight' or k == 'norm.bias':
                # try moving norm -> fc norm on fine-tune, probably a better starting point than new init
                k = k.replace('norm', 'fc_norm')
            else:
                # skip pretrain mask token & head weights
                continue

        out_dict[k] = v

    return out_dict


def tiny(pretrained=None, **kwargs):
    model = Eva(
        img_size=224,
        patch_size=14,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4 * 2 / 3,
        swiglu_mlp=True,
        use_rot_pos_emb=True,
        ref_feat_shape=(16, 16),  # 224/14
        **kwargs
    )
    if pretrained:
        checkpoint = torch.load(pretrained, map_location="cpu")
        checkpoint = checkpoint_filter_fn(checkpoint, model)
        model.load_state_dict(checkpoint, strict=False)
    return model, 192


def small(pretrained=False, **kwargs):
    model = Eva(
        img_size=224,
        patch_size=14,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4 * 2 / 3,
        swiglu_mlp=True,
        use_rot_pos_emb=True,
        ref_feat_shape=(16, 16),  # 224/14
        **kwargs
    )
    if pretrained:
        checkpoint = torch.load(pretrained, map_location="cpu")
        checkpoint = checkpoint_filter_fn(checkpoint, model)
        model.load_state_dict(checkpoint, strict=False)
    return model, 384


def base(pretrained=False, **kwargs):
    model = Eva(
        img_size=224,
        patch_size=14,
        embed_dim=768,
        depth=12,
        num_heads=12,
        qkv_fused=False,
        mlp_ratio=4 * 2 / 3,
        swiglu_mlp=True,
        scale_mlp=True,
        use_rot_pos_emb=True,
        ref_feat_shape=(16, 16),  # 224/14
        **kwargs
    )
    if pretrained:
        checkpoint = torch.load(pretrained, map_location="cpu")
        checkpoint = checkpoint_filter_fn(checkpoint, model)
        model.load_state_dict(checkpoint, strict=False)
    return model, 768


def large(pretrained=False, **kwargs):
    model = Eva(
        img_size=224,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4 * 2 / 3,
        qkv_fused=False,
        swiglu_mlp=True,
        scale_mlp=True,
        use_rot_pos_emb=True,
        ref_feat_shape=(16, 16),  # 224/14
        **kwargs
    )
    if pretrained:
        checkpoint = torch.load(pretrained, map_location="cpu")
        checkpoint = checkpoint_filter_fn(checkpoint, model)
        model.load_state_dict(checkpoint, strict=False)
    return model, 1024
