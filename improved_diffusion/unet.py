from abc import abstractmethod

import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from typing import Dict, List, Tuple, Optional

from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import (
    SiLU,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
    checkpoint,
    PositionEmbeddingLearned,
)

from . import logger


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class SpatialAttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    """

    @abstractmethod
    def forward(self, x, pos_emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock, SpatialAttentionBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, query_pos=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialAttentionBlock):
                x = layer(x, query_pos)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, channels, channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest")
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims, channels, channels, 3, stride=stride, padding=1)
        else:
            self.op = avg_pool_nd(stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            SiLU(),
            nn.Dropout(p=dropout),
            zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(self._forward, (x, emb), self.parameters(), self.use_checkpoint)

    def _forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(SpatialAttentionBlock):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(self,
                 channels,
                 num_heads=1,
                 use_checkpoint=False,
                 pos_emb_layers=None,
                 pos_emb_in_channels=None,
                 pos_emb_out_channels=None):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint

        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = QKVAttention()
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))
        self.pos_emb_layers = pos_emb_layers
        if pos_emb_layers is not None:
            self.pos_emb_proj = nn.Sequential(
                SiLU(),
                linear(
                    pos_emb_in_channels,
                    pos_emb_out_channels,
                ),
            )

    def with_pos_embed(self, tensor, pos: Optional[th.Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, x, query_pos=None):
        return checkpoint(self._forward, (x, query_pos), self.parameters(), self.use_checkpoint)

    def _forward(self, x, query_pos=None):
        pos_emb = None
        if (query_pos is not None) and (self.pos_emb_layers is not None):
            # Bx128x16
            pos_emb = self.pos_emb_layers(query_pos)
            logger.debug(f"AttentionBlock::_forward pos_emb: {pos_emb.shape}")
            # Bx128x128, project position embedding to match the feature dimension
            pos_emb = self.pos_emb_proj(pos_emb)
            logger.debug(f"AttentionBlock::_forward projected_pos_emb: {pos_emb.shape}")

        b, c, *spatial = x.shape
        logger.debug(f"AttentionBlock::_forward {b, c, spatial}")
        x = x.reshape(b, c, -1)
        x = self.norm(x)
        x = self.with_pos_embed(x, pos_emb)

        # logger.debug(f"AttentionBlock::_forward reshaped x: {x.shape}")
        qkv = self.qkv(x)
        # logger.debug(f"AttentionBlock::_forward qkv: {qkv.shape}")
        qkv = qkv.reshape(b * self.num_heads, -1, qkv.shape[2])
        # logger.debug(f"AttentionBlock::_forward reshaped qkv: {qkv.shape}")

        h = self.attention(qkv)
        h = h.reshape(b, -1, h.shape[-1])
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention.
    """

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (C * 3) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x C x T] tensor after attention.
        """
        ch = qkv.shape[1] // 3
        q, k, v = th.split(qkv, ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum("bct,bcs->bts", q * scale, k * scale)  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        return th.einsum("bts,bcs->bct", weight, v)

    @staticmethod
    def count_flops(model, _x, y):
        """
        A counter for the `thop` package to count the operations in an
        attention operation.

        Meant to be used like:

            macs, params = thop.profile(
                model,
                inputs=(inputs, timestamps),
                custom_ops={QKVAttention: QKVAttention.count_flops},
            )

        """
        b, c, *spatial = y[0].shape
        num_spatial = int(np.prod(spatial))
        # We perform two matmuls with the same number of ops.
        # The first computes the weight matrix, the second computes
        # the combination of the value vectors.
        matmul_ops = 2 * b * (num_spatial**2) * c
        model.total_ops += th.DoubleTensor([matmul_ops])


class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=1,
        num_classes=None,
        use_checkpoint=False,
        num_heads=1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        feature_size=64,
        bbox_centroid_idx=22,
        bbox_centroid_dim=3,
        bbox_size_idx=25,
        bbox_size_dim=3,
        use_position_embedding=True,
        use_centroid_and_size_pos_embedding=True,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        logger.debug(f"channel_mult: {channel_mult}")
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_heads_upsample = num_heads_upsample

        self.use_position_embedding = use_position_embedding
        self.use_centroid_and_size_pos_embedding = use_centroid_and_size_pos_embedding
        self.bbox_centroid_idx = bbox_centroid_idx
        self.bbox_centroid_dim = bbox_centroid_dim
        self.bbox_size_idx = bbox_size_idx
        self.bbox_size_dim = bbox_size_dim

        # time embedding block
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        # label embedding block
        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        # position embedding block
        self.input_pos_emb_layers = nn.ModuleList([])
        self.middle_pos_emb_layers = nn.ModuleList([])
        self.output_pos_emb_layers = nn.ModuleList([])

        # position dimension is 6 if using centroid and size for position embedding,
        # otherwise it is 3
        position_dim = 6 if use_centroid_and_size_pos_embedding else 3

        # self attention block
        self.self_attn_input_blocks = nn.ModuleList([])
        self.self_attn_output_blocks = nn.ModuleList([])

        # input block
        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(
                OrderedDict([('input_conv', conv_nd(dims, in_channels, model_channels, 3, padding=1))]))
        ])
        input_block_chans = [model_channels]
        ch = model_channels
        # downsample rates for the attention layers
        downsample_ratio = 1
        for level, mult in enumerate(channel_mult):
            for idx in range(num_res_blocks):
                # each resblock consists of a resblock, an optional attention block
                idx_str = f'{level}_{idx}'
                layers = [(
                    'resblock_' + idx_str,
                    ResBlock(
                        channels=ch,  # input channels
                        emb_channels=time_embed_dim,  # embedding channels
                        dropout=dropout,  # dropout probability
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    ))]
                # increase minput channels for next resblock
                ch = mult * model_channels
                # insert attention block if needed
                if downsample_ratio in attention_resolutions:
                    pos_emb_layer = PositionEmbeddingLearned(input_channel=position_dim, out_channel=ch)
                    layers.append(('attnblock_' + idx_str,
                                   AttentionBlock(channels=ch,
                                                  use_checkpoint=use_checkpoint,
                                                  num_heads=num_heads,
                                                  pos_emb_layers=pos_emb_layer,
                                                  pos_emb_in_channels=in_channels,
                                                  pos_emb_out_channels=feature_size)))

                self.input_blocks.append(TimestepEmbedSequential(OrderedDict(layers)))
                input_block_chans.append(ch)
            # create downsampling block at each level
            if level != len(channel_mult) - 1:
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        OrderedDict([('down_' + idx_str, Downsample(channels=ch, use_conv=conv_resample, dims=dims))])))
                input_block_chans.append(ch)
                downsample_ratio *= 2
                feature_size = feature_size // 2

        # out_channel is same as input_channel in moddle block
        pos_emb_layer = PositionEmbeddingLearned(input_channel=position_dim, out_channel=ch)
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(ch,
                           use_checkpoint=use_checkpoint,
                           num_heads=num_heads,
                           pos_emb_layers=pos_emb_layer,
                           pos_emb_in_channels=in_channels,
                           pos_emb_out_channels=feature_size),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                idx_str = f'{level}_{i}'
                layers = [('resblock_' + idx_str,
                           ResBlock(
                               ch + input_block_chans.pop(),
                               time_embed_dim,
                               dropout,
                               out_channels=model_channels * mult,
                               dims=dims,
                               use_checkpoint=use_checkpoint,
                               use_scale_shift_norm=use_scale_shift_norm,
                           ))]
                ch = model_channels * mult
                if downsample_ratio in attention_resolutions:
                    pos_emb_layer = PositionEmbeddingLearned(input_channel=position_dim, out_channel=ch)
                    layers.append(('attnblock_' + idx_str,
                                   AttentionBlock(ch,
                                                  use_checkpoint=use_checkpoint,
                                                  num_heads=num_heads_upsample,
                                                  pos_emb_layers=pos_emb_layer,
                                                  pos_emb_in_channels=in_channels,
                                                  pos_emb_out_channels=feature_size)))
                if level and i == num_res_blocks:
                    layers.append(('up_' + idx_str, Upsample(channels=ch, use_conv=conv_resample, dims=dims)))
                    downsample_ratio //= 2
                    feature_size *= 2
                self.output_blocks.append(TimestepEmbedSequential(OrderedDict(layers)))

        self.out = nn.Sequential(
            normalization(ch),
            SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    @property
    def inner_dtype(self):
        """
        Get the dtype used by the torso of the model.
        """
        return next(self.input_blocks.parameters()).dtype

    def normalized_room_layout_bbox_position(self, bbox_position: th.Tensor):
        b, c, bbox_dim = bbox_position.shape
        room_layout_position = th.tensor([[0, 0, 0], [1, 1, 1], [-1, -1, -1]],
                                         dtype=bbox_position.dtype,
                                         device=bbox_position.device)
        room_layout_position = room_layout_position.unsqueeze(0).repeat(b, 1, 1)
        bbox_position[:, :3, :] = room_layout_position
        return bbox_position

    def normalized_room_layout_bbox_size(self, bbox_size: th.Tensor):
        b, c, size_dim = bbox_size.shape
        room_layout_size = th.tensor([[2, 2, 2], [2, 2, 2], [2, 2, 2]], dtype=bbox_size.dtype, device=bbox_size.device)
        room_layout_size = room_layout_size.unsqueeze(0).repeat(b, 1, 1)
        bbox_size[:, :3, :] = room_layout_size
        return bbox_size

    def forward(self, x, timesteps, y=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (self.num_classes
                                   is not None), "must specify y if and only if the model is class-conditional"

        hs = []
        logger.debug(f"UNetModel::forward: input x shape: {x.shape}")
        logger.debug(f"UNetModel::forward: input timesteps shape: {timesteps.shape}")
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        logger.debug(f"UNetModel::forward: output timesteps embedding shape: {emb.shape}")

        if self.num_classes is not None:
            logger.debug(f"UNetModel::forward: input y shape: {y.shape}")
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)
            logger.debug(f"UNetModel::forward: output y embedding shape: {emb.shape}")

        bbox_pos = x[..., self.bbox_centroid_idx:self.bbox_centroid_idx + self.bbox_centroid_dim]
        bbox_pos = self.normalized_room_layout_bbox_position(bbox_pos)
        bbox_size = (x[..., self.bbox_size_idx:self.bbox_size_idx + self.bbox_size_dim] + 1)
        bbox_size = self.normalized_room_layout_bbox_size(bbox_size)
        bbox_pos = th.cat([bbox_pos, bbox_size], dim=-1) if self.use_centroid_and_size_pos_embedding else bbox_pos
        logger.debug(f'bbox_pos: {bbox_pos.shape}')

        h = x.type(self.inner_dtype)

        for module in self.input_blocks:
            h = module(h, emb, bbox_pos)
            logger.debug(f"UNetModel::forward: input_blocks hidden layer output shape: {h.shape}")
            hs.append(h)

        h = self.middle_block(h, emb, bbox_pos)
        logger.debug(f"UNetModel::forward: middle_block hidden layer output shape: {h.shape}")

        for module in self.output_blocks:
            h_in_input = hs.pop()
            # logger.debug(f"UNetModel::forward: output_blocks hs.pop output shape: {h_in_input.shape}")
            cat_in = th.cat([h, h_in_input], dim=1)
            h = module(cat_in, emb, bbox_pos)
            logger.debug(f"UNetModel::forward: output_blocks h output shape: {h.shape}")

        h = h.type(x.dtype)
        return self.out(h)

    def get_feature_vectors(self, x, timesteps, y=None):
        """
        Apply the model and return all of the intermediate tensors.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: a dict with the following keys:
                 - 'down': a list of hidden state tensors from downsampling.
                 - 'middle': the tensor of the output of the lowest-resolution
                             block in the model.
                 - 'up': a list of hidden state tensors from upsampling.
        """
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)
        result = dict(down=[], up=[])
        h = x.type(self.inner_dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
            result["down"].append(h.type(x.dtype))
        h = self.middle_block(h, emb)
        result["middle"] = h.type(x.dtype)
        for module in self.output_blocks:
            cat_in = th.cat([h, hs.pop()], dim=1)
            h = module(cat_in, emb)
            result["up"].append(h.type(x.dtype))
        return result


class SuperResModel(UNetModel):
    """
    A UNetModel that performs super-resolution.

    Expects an extra kwarg `low_res` to condition on a low-resolution image.
    """

    def __init__(self, in_channels, *args, **kwargs):
        super().__init__(in_channels * 2, *args, **kwargs)

    def forward(self, x, timesteps, low_res=None, **kwargs):
        _, _, new_height, new_width = x.shape
        upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear")
        x = th.cat([x, upsampled], dim=1)
        return super().forward(x, timesteps, **kwargs)

    def get_feature_vectors(self, x, timesteps, low_res=None, **kwargs):
        _, new_height, new_width, _ = x.shape
        upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear")
        x = th.cat([x, upsampled], dim=1)
        return super().get_feature_vectors(x, timesteps, **kwargs)
