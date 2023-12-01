# Copyright (c) OpenMMLab. All rights reserved.
import math
import copy
import torch
import torch.nn as nn
from mmcv.cnn import build_conv_layer
from mmcv.runner import BaseModule, force_fp32
from mmdet.core import (build_assigner, build_sampler, multi_apply,
                        reduce_mean, build_bbox_coder)
from mmdet.models.utils import build_transformer
from mmdet.models import HEADS, build_loss
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet3d.models.utils.clip_sigmoid import clip_sigmoid
from mmdet3d.models import builder
from einops import rearrange
import collections

from functools import reduce
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
from mmdet3d.ops import  make_sparse_convmodule
import spconv.pytorch as spconv
from mmcv.cnn import build_conv_layer
import copy
from spconv.core import ConvAlgo

def pos2embed(pos, num_pos_feats=128, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    # pos_z = pos[..., 2, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    # pos_z = torch.stack((pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()), dim=-1).flatten(-2)
    posemb = torch.cat((pos_y, pos_x), dim=-1)
    return posemb


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, groups, eps):
        ctx.groups = groups
        ctx.eps = eps
        N, C, L = x.size()
        x = x.view(N, groups, C // groups, L)
        mu = x.mean(2, keepdim=True)
        var = (x - mu).pow(2).mean(2, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1) * y.view(N, C, L) + bias.view(1, C, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        groups = ctx.groups
        eps = ctx.eps

        N, C, L = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1)
        g = g.view(N, groups, C//groups, L)
        mean_g = g.mean(dim=2, keepdim=True)
        mean_gy = (g * y).mean(dim=2, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx.view(N, C, L), (grad_output * y.view(N, C, L)).sum(dim=2).sum(dim=0), grad_output.sum(dim=2).sum(
            dim=0), None, None


class GroupLayerNorm1d(nn.Module):

    def __init__(self, channels, groups=1, eps=1e-6):
        super(GroupLayerNorm1d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.groups = groups
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.groups, self.eps)


@HEADS.register_module()
class SeparateTaskHead(BaseModule):
    """SeparateHead for CenterHead.

    Args:
        in_channels (int): Input channels for conv_layer.
        heads (dict): Conv information.
        head_conv (int): Output channels.
            Default: 64.
        final_kernal (int): Kernal size for the last conv layer.
            Deafult: 1.
        init_bias (float): Initial bias. Default: -2.19.
        conv_cfg (dict): Config of conv layer.
            Default: dict(type='Conv2d')
        norm_cfg (dict): Config of norm layer.
            Default: dict(type='BN2d').
        bias (str): Type of bias. Default: 'auto'.
    """

    def __init__(self,
                 in_channels,
                 heads,
                 groups=1,
                 head_conv=64,
                 final_kernel=1,
                 init_bias=-2.19,
                 init_cfg=None,
                 **kwargs):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
            'behavior, init_cfg is not allowed to be set'
        super(SeparateTaskHead, self).__init__(init_cfg=init_cfg)
        self.heads = heads
        self.groups = groups
        self.init_bias = init_bias
        for head in self.heads:
            classes, num_conv = self.heads[head]

            conv_layers = []
            c_in = in_channels
            for i in range(num_conv - 1):
                conv_layers.extend([
                    nn.Conv1d(
                        c_in * groups,
                        head_conv * groups,
                        kernel_size=final_kernel,
                        stride=1,
                        padding=final_kernel // 2,
                        groups=groups,
                        bias=False),
                    GroupLayerNorm1d(head_conv * groups, groups=groups),
                    nn.ReLU(inplace=True)
                ])
                c_in = head_conv

            conv_layers.append(
                nn.Conv1d(
                    head_conv * groups,
                    classes * groups,
                    kernel_size=final_kernel,
                    stride=1,
                    padding=final_kernel // 2,
                    groups=groups,
                    bias=True))
            conv_layers = nn.Sequential(*conv_layers)

            self.__setattr__(head, conv_layers)

            if init_cfg is None:
                self.init_cfg = dict(type='Kaiming', layer='Conv1d')

    def init_weights(self):
        """Initialize weights."""
        super().init_weights()
        for head in self.heads:
            if head == 'cls_logits':
                self.__getattr__(head)[-1].bias.data.fill_(self.init_bias)

    def forward(self, x):
        """Forward function for SepHead.

        Args:
            x (torch.Tensor): Input feature map with the shape of
                [N, B, query, C].

        Returns:
            dict[str: torch.Tensor]: contains the following keys:

                -reg （torch.Tensor): 2D regression value with the \
                    shape of [N, B, query, 2].
                -height (torch.Tensor): Height value with the \
                    shape of [N, B, query, 1].
                -dim (torch.Tensor): Size value with the shape \
                    of [N, B, query, 3].
                -rot (torch.Tensor): Rotation value with the \
                    shape of [N, B, query, 2].
                -vel (torch.Tensor): Velocity value with the \
                    shape of [N, B, query, 2].
        """
        N, B, query_num, c1 = x.shape
        x = rearrange(x, "n b q c -> b (n c) q")
        ret_dict = dict()
        
        for head in self.heads:
             head_output = self.__getattr__(head)(x)
             ret_dict[head] = rearrange(head_output, "b (n c) q -> n b q c", n=N)

        return ret_dict



@HEADS.register_module()
class FSTRHead(BaseModule):
    "only init lidar proposal query"
    def __init__(self,
                in_channels,
                num_init_query = 200,
                num_query=900,
                max_sparse_token_per_sample = 10000,
                proposal_head_kernel = 3,
                hidden_dim=128,
                norm_bbox=True,
                downsample_scale=8,
                scalar=10,
                noise_scale=1.0,
                noise_trans=0.0,
                dn_weight=1.0,
                split=0.75,
                depth_num=64,
                nms_kernel_size=3,
                init_dn_query=False,
                init_learnable_query = False,
                init_query_topk = 1,
                init_query_radius = 1,
                gauusian_dn_sampling=False,
                noise_mean = 0.5,
                noise_std = 0.125,
                train_cfg=None,
                test_cfg=None,
                common_heads=dict(
                    center=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)
                ),
                tasks=[
                dict(num_class=1, class_names=['car']),
                dict(num_class=2, class_names=['truck', 'construction_vehicle']),
                dict(num_class=2, class_names=['bus', 'trailer']),
                dict(num_class=1, class_names=['barrier']),
                dict(num_class=2, class_names=['motorcycle', 'bicycle']),
                dict(num_class=2, class_names=['pedestrian', 'traffic_cone']),
                ],
                transformer=None,
                bbox_coder=None,
                loss_cls=dict(
                    type="FocalLoss",
                    use_sigmoid=True,
                    reduction="mean",
                    gamma=2, alpha=0.25, loss_weight=1.0
                ),
                loss_bbox=dict(
                type="L1Loss",
                reduction="mean",
                loss_weight=0.25,
                ),
                loss_heatmap=dict(
                    type="GuassianFocalLoss",
                    reduction="mean"
                ),
                separate_head=dict(
                    type='SeparateMlpHead', init_bias=-2.19, final_kernel=3),
                init_cfg=None,
                **kwargs):
        super(FSTRHead, self).__init__(**kwargs)


        self.num_classes = [len(t["class_names"]) for t in tasks]
        self.class_names = [t["class_names"] for t in tasks]
        self.hidden_dim = hidden_dim
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.num_query = num_query
        self.in_channels = in_channels
        self.norm_bbox = norm_bbox
        self.downsample_scale = downsample_scale
        self.scalar = scalar
        self.bbox_noise_scale = noise_scale
        self.bbox_noise_trans = noise_trans
        self.dn_weight = dn_weight
        self.split = split
        self.depth_num = depth_num
        self.nms_kernel_size = nms_kernel_size
        self.num_proposals = num_query
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_heatmap = build_loss(loss_heatmap)
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self.fp16_enabled = False
        self.init_dn_query = init_dn_query
        self.init_learnable_query = init_learnable_query
        self.gauusian_dn_sampling = gauusian_dn_sampling
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.init_query_topk = init_query_topk
        self.init_query_radius = init_query_radius

        # transformer
        self.transformer = build_transformer(transformer)
        # self.reference_points = nn.Embedding(num_query, 3)
        self.bev_embedding = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # task head
        self.task_heads = nn.ModuleList()
        for num_cls in self.num_classes:
            heads = copy.deepcopy(common_heads)
            heads.update(dict(cls_logits=(num_cls, 2)))
            separate_head.update(
                in_channels=hidden_dim,
                heads=heads, num_cls=num_cls,
                groups=transformer.decoder.num_layers
            )
            self.task_heads.append(builder.build_head(separate_head))

        # assigner
        if train_cfg:
            self.assigner = build_assigner(train_cfg["assigner"])
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)


        self.num_init_query = num_init_query
        assert self.num_init_query < self.num_query, "number of init query must less than number of query"
        self.reference_points = nn.Embedding(self.num_query - self.num_init_query, 3)
        self.class_encoding = nn.Sequential()
        self.shared_conv = make_sparse_convmodule(
                self.in_channels,
                self.hidden_dim,
                (3,3),
                norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                padding=(1,1),
                indice_key='head_spconv_1',
                conv_type='SubMConv2d',
                order=('conv', 'norm', 'act'))
        self.sparse_maxpool_2d = spconv.SparseMaxPool2d(3, 1, 1, subm=True, algo=ConvAlgo.Native, indice_key='max_pool_head_3')
        self.sparse_maxpool_2d_small = spconv.SparseMaxPool2d(1, 1, 0, subm=True, algo=ConvAlgo.Native, indice_key='max_pool_head_3')
        self.max_sparse_token_per_sample = max_sparse_token_per_sample

        # for sparse heatmap
        self.proposal_head_kernel = proposal_head_kernel
        output_channels = sum(self.num_classes)
        num_conv = 2
        self.heatmap_head = nn.Sequential()
        fc_list = []
        for k in range(num_conv - 1):
            fc_list.append(
                make_sparse_convmodule(
                self.hidden_dim,
                self.hidden_dim,
                self.proposal_head_kernel,
                norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                padding=int(self.proposal_head_kernel//2),
                indice_key='head_spconv_1',
                conv_type='SubMConv2d',
                order=('conv', 'norm', 'act')),
            )
        fc_list.append(build_conv_layer(
                        dict(type='SubMConv2d', indice_key='hm_out'),
                        self.hidden_dim,
                        sum(self.num_classes),
                        1,
                        stride=1,
                        padding=0,
                        bias=True))
        
        
        self.sparse_hm_layer = nn.Sequential(*fc_list)
        self.sparse_hm_layer[-1].bias.data.fill_(-2.19)

    @property
    def coords_bev(self):
        cfg = self.train_cfg if self.train_cfg else self.test_cfg
        x_size, y_size = (
            cfg['grid_size'][1] // self.downsample_scale,
            cfg['grid_size'][0] // self.downsample_scale
        )
        meshgrid = [[0, y_size - 1, y_size], [0, x_size - 1, x_size]]
        batch_y, batch_x = torch.meshgrid(*[torch.linspace(it[0], it[1], it[2]) for it in meshgrid])
        batch_x = (batch_x + 0.5) / x_size
        batch_y = (batch_y + 0.5) / y_size
        coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)
        coord_base = coord_base.view(2, -1).transpose(1, 0) # (H*W, 2)
        return coord_base
    def init_weights(self):
        super(FSTRHead, self).init_weights()
        nn.init.uniform_(self.reference_points.weight.data, 0, 1)
        
    def _bev_query_embed(self, ref_points, img_metas):
        bev_embeds = self.bev_embedding(pos2embed(ref_points, num_pos_feats=self.hidden_dim))
        return bev_embeds
    def forward(self, points_feats, img_metas=None):
        """
            list([bs, c, h, w])
        """
        img_metas = [img_metas]
        return multi_apply(self.forward_single, points_feats, img_metas)    

    def forward_single(self, x, img_metas):
        """
            x: [bs c h w]
            return List(dict(head_name: [num_dec x bs x num_query * head_dim]) ) x task_num
        """
        ret_dicts = []
        batch_size = len(img_metas)
        x = self.shared_conv(x)
        x_feature = torch.zeros(*(x.features.shape),device = x.features.device)
        x_feature[:,:] = x.features
        x_batch_indices = torch.zeros(x.indices.shape[0],1,device = x.features.device)
        x_ind = torch.zeros(x.indices.shape[0],2,device = x.features.device)
        x_2dpos = torch.zeros(x.indices.shape[0],2,device = x.features.device)
        x_batch_indices[:,:] = x.indices[:,:1]
        x_ind[:,:] = x.indices[:,-2:]
        x_ind = x_ind.to(torch.float32)
        cfg = self.train_cfg if self.train_cfg else self.test_cfg
        y_size, x_size = x.spatial_shape
        x_2dpos[:,0] = (x_ind[:,1] + 0.5) / x_size
        x_2dpos[:,1] = (x_ind[:,0] + 0.5) / y_size
        batch_size = int(x.batch_size)

        sparse_hm = self.sparse_hm_layer(x)
        sparse_hm_clone = spconv.SparseConvTensor(
                features=sparse_hm.features.clone().detach().sigmoid(),
                indices=sparse_hm.indices.clone(),
                spatial_shape=sparse_hm.spatial_shape,
                batch_size=sparse_hm.batch_size
            )
        x_hm_max = self.sparse_maxpool_2d(sparse_hm_clone, True)
        x_hm_max_small = self.sparse_maxpool_2d_small(sparse_hm_clone, True)


        selected = (x_hm_max.features == sparse_hm_clone.features)
        selected_small = (x_hm_max_small.features == sparse_hm_clone.features)
        selected[:,8] = selected_small[:,8]
        selected[:,9] = selected_small[:,9]

        score = sparse_hm_clone.features * selected
        score, _ = score.topk(1,dim=1)
        proposal_list = []
        proposal_feature = []
        # topk for each sample in batch
        for i in range(batch_size):
            mask = (x_batch_indices == i).squeeze(-1)
            sample_voxel_pos = x_2dpos[mask]
            sample_voxel_hm = score[mask]
            sample_voxel_feature = x_feature[mask]
            _, proposal_ind = sample_voxel_hm.topk(self.num_init_query,dim=0)
            proposal_list.append(sample_voxel_pos.gather(0, proposal_ind.repeat(1,2))[None,...])
            proposal_feature.append(sample_voxel_feature.gather(0, proposal_ind.repeat(1,sample_voxel_feature.shape[1]))[None,...])
        query_pos = torch.cat(proposal_list,dim=0)
        query_init_feature = torch.cat(proposal_feature,dim=0)

        reference_points = self.reference_points.weight
        reference_points = reference_points.unsqueeze(0).repeat(batch_size,1,1)

        init_reference_points = torch.cat([query_pos,0.5*torch.ones([*query_pos.shape[:-1],1]).to(query_pos.device)],dim=-1)

        reference_points = torch.cat([init_reference_points, reference_points],dim=1)

        reference_points, attn_mask, mask_dict = self.prepare_for_dn(batch_size, reference_points, img_metas)
        
        pad_size = mask_dict['pad_size'] if mask_dict is not None else 0
        
        target = self.get_sparse_init_query(reference_points, x_feature , x_2dpos, x_batch_indices, pad_size)
        
        bev_pos_embeds = self.bev_embedding(pos2embed(x_2dpos, num_pos_feats=self.hidden_dim))
        
        bev_query_embeds  = self.query_embed(reference_points, img_metas)
        query_embeds = bev_query_embeds


        # pad or drop 

        batch_feature = torch.zeros(batch_size,self.max_sparse_token_per_sample,self.hidden_dim,device = x.features.device)
        batch_bevemb = torch.zeros(batch_size,self.max_sparse_token_per_sample,self.hidden_dim,device = x.features.device)

        for i in range(batch_size):
            sample_token_num = (x_batch_indices==i).sum()
            batch_token_num = min(sample_token_num,self.max_sparse_token_per_sample)
            mask = (x_batch_indices == i).squeeze(-1)
            sample_voxel_hm = score[mask]
            sample_voxel_feature = x_feature[mask]
            sample_voxel_bev_emb = bev_pos_embeds[mask]
            _, voxel_ind = sample_voxel_hm.topk(batch_token_num,dim=0)
            # a = sample_voxel_feature.gather(0, voxel_ind.repeat(1,sample_voxel_feature.shape[1]))[None,...]
            batch_feature[i][:batch_token_num] = sample_voxel_feature.gather(0, voxel_ind.repeat(1,sample_voxel_feature.shape[1]))
            batch_bevemb[i][:batch_token_num] = sample_voxel_bev_emb.gather(0, voxel_ind.repeat(1,sample_voxel_bev_emb.shape[1]))

        outs_dec, _ = self.transformer(
                            batch_feature, query_embeds,
                            batch_bevemb,
                            attn_masks=attn_mask,
                            target = target
                        )
        outs_dec = torch.nan_to_num(outs_dec)

        reference = inverse_sigmoid(reference_points.clone())
        
        flag = 0
        for task_id, task in enumerate(self.task_heads, 0):
            outs = task(outs_dec)
            center = (outs['center'] + reference[None, :, :, :2]).sigmoid()
            height = (outs['height'] + reference[None, :, :, 2:3]).sigmoid()
            _center, _height = center.new_zeros(center.shape), height.new_zeros(height.shape)
            _center[..., 0:1] = center[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
            _center[..., 1:2] = center[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
            _height[..., 0:1] = height[..., 0:1] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
            outs['center'] = _center
            outs['height'] = _height
            
            if mask_dict and mask_dict['pad_size'] > 0:
                task_mask_dict = copy.deepcopy(mask_dict)
                class_name = self.class_names[task_id]

                known_lbs_bboxes_label =  task_mask_dict['known_lbs_bboxes'][0]
                known_labels_raw = task_mask_dict['known_labels_raw']
                new_lbs_bboxes_label = known_lbs_bboxes_label.new_zeros(known_lbs_bboxes_label.shape)
                new_lbs_bboxes_label[:] = len(class_name)
                new_labels_raw = known_labels_raw.new_zeros(known_labels_raw.shape)
                new_labels_raw[:] = len(class_name)
                task_masks = [
                    torch.where(known_lbs_bboxes_label == class_name.index(i) + flag)
                    for i in class_name
                ]
                task_masks_raw = [
                    torch.where(known_labels_raw == class_name.index(i) + flag)
                    for i in class_name
                ]
                for cname, task_mask, task_mask_raw in zip(class_name, task_masks, task_masks_raw):
                    new_lbs_bboxes_label[task_mask] = class_name.index(cname)
                    new_labels_raw[task_mask_raw] = class_name.index(cname)
                task_mask_dict['known_lbs_bboxes'] = (new_lbs_bboxes_label, task_mask_dict['known_lbs_bboxes'][1])
                task_mask_dict['known_labels_raw'] = new_labels_raw
                flag += len(class_name)
                
                for key in list(outs.keys()):
                    outs['dn_' + key] = outs[key][:, :, :mask_dict['pad_size'], :]
                    outs[key] = outs[key][:, :, mask_dict['pad_size']:, :]
                outs['dn_mask_dict'] = task_mask_dict
            
            ret_dicts.append(outs)
        ret_dicts[0]['sparse_heatmap'] = sparse_hm
        return ret_dicts
    
    
    def get_sparse_init_query(self, ref_points, x_feature, x_2dpos , x_batch_indices, pad_size):

        total_range = self.pc_range[3]-self.pc_range[0]
        radius = self.init_query_radius
        diameter = (2 * radius + 1)/total_range
        sigma = diameter / 6
        # masked_gaussian = torch.exp(- distances / (2 * sigma * sigma))
        query_feature_list = []
        batch_size = ref_points.shape[0]

        for bs in range(batch_size):
            sample_q = ref_points[bs][:,:2]
            sample_mask = x_batch_indices[:,0] == bs
            sample_token = x_feature[sample_mask]
            sample_pos = x_2dpos[sample_mask]
            with torch.no_grad():
                dis_mat = sample_q.unsqueeze(1) - sample_pos.unsqueeze(0)
                dis_mat = -(dis_mat ** 2).sum(-1)
                nearest_dis_topk,nearest_order_topk = dis_mat.topk(self.init_query_topk ,dim=1,sorted= True)
                gaussian_weight = torch.exp( nearest_dis_topk / (2 * sigma * sigma))
                gaussian_weight_sum = torch.clip(gaussian_weight.sum(-1),1)
            
            flatten_order = nearest_order_topk.view(-1,self.init_query_topk)
            flatten_weight = (gaussian_weight/gaussian_weight_sum.unsqueeze(1)).view(-1,self.init_query_topk)
            feature = (sample_token.gather(0, flatten_order.repeat(1,sample_token.shape[1]))*flatten_weight).view(-1,self.init_query_topk,sample_token.shape[1]).sum(1).unsqueeze(0)
            query_feature_list.append(feature)
        
        query_feature = torch.cat(query_feature_list,dim=0)
        if not self.init_dn_query:
           query_feature[:,:pad_size,:] *=0 
        if not self.init_learnable_query:
           query_feature[:,pad_size+self.num_init_query:,:] *=0  
        query_feature = query_feature.permute(1,0,2)


        return query_feature
    

    def prepare_for_dn(self, batch_size, reference_points, img_metas):
        if self.training:
            targets = [torch.cat((img_meta['gt_bboxes_3d']._data.gravity_center, img_meta['gt_bboxes_3d']._data.tensor[:, 3:]),dim=1) for img_meta in img_metas ]
            labels = [img_meta['gt_labels_3d']._data for img_meta in img_metas ]

            known = [(torch.ones_like(t)).cuda() for t in labels]
            know_idx = known
            unmask_bbox = unmask_label = torch.cat(known)
            known_num = [t.size(0) for t in targets]
            labels = torch.cat([t for t in labels])
            boxes = torch.cat([t for t in targets])
            batch_idx = torch.cat([torch.full((t.size(0),), i) for i, t in enumerate(targets)])

            known_indice = torch.nonzero(unmask_label + unmask_bbox)
            known_indice = known_indice.view(-1)
            # add noise
            groups = min(self.scalar, self.num_query // max(known_num))
            known_indice = known_indice.repeat(groups, 1).view(-1)
            known_labels = labels.repeat(groups, 1).view(-1).long().to(reference_points.device)
            known_labels_raw = labels.repeat(groups, 1).view(-1).long().to(reference_points.device)
            known_bid = batch_idx.repeat(groups, 1).view(-1)
            known_bboxs = boxes.repeat(groups, 1).to(reference_points.device)
            known_bbox_center = known_bboxs[:, :3].clone()
            known_bbox_scale = known_bboxs[:, 3:6].clone()

            # known_one_hot = F.one_hot(known_labels, self.num_classes[0]).permute(1,0)
            # known_query_cat_encoding = self.class_encoding(known_one_hot.float().unsqueeze(0))
            if self.bbox_noise_scale > 0:
                diff = known_bbox_scale / 2 + self.bbox_noise_trans
                if self.gauusian_dn_sampling:
                    rand_prob = torch.randn_like(known_bbox_center)*self.noise_std + self.noise_mean
                    rand_pn = torch.rand_like(known_bbox_center)
                    p_mask = rand_pn>0.5
                    n_mask = rand_pn<=0.5
                    rand_prob[n_mask] *= -1
                else:
                    rand_prob = torch.rand_like(known_bbox_center) * 2 - 1.0
                known_bbox_center += torch.mul(rand_prob, diff) * self.bbox_noise_scale
                known_bbox_center[..., 0:1] = (known_bbox_center[..., 0:1] - self.pc_range[0]) / (
                    self.pc_range[3] - self.pc_range[0]
                )
                known_bbox_center[..., 1:2] = (known_bbox_center[..., 1:2] - self.pc_range[1]) / (
                    self.pc_range[4] - self.pc_range[1]
                )
                known_bbox_center[..., 2:3] = (known_bbox_center[..., 2:3] - self.pc_range[2]) / (
                    self.pc_range[5] - self.pc_range[2]
                )
                known_bbox_center = known_bbox_center.clamp(min=0.0, max=1.0)
                mask = torch.norm(rand_prob, 2, 1) > self.split
                known_labels[mask] = sum(self.num_classes)

            single_pad = int(max(known_num))
            pad_size = int(single_pad * groups)
            padding_bbox = torch.zeros(batch_size,pad_size, 3).to(reference_points.device)
            # padding_cls_encoding = torch.zeros(batch_size,query_cat_encoding.shape[1],pad_size).to(reference_points.device)
            padded_reference_points = torch.cat([padding_bbox, reference_points], dim=1)
            # padding_query_cat_encoding = torch.cat([padding_cls_encoding, query_cat_encoding], dim=-1)
            # padding_query_cat_encoding = padding_query_cat_encoding.permute(0,2,1)
            # known_query_cat_encoding = known_query_cat_encoding.permute(0,2,1)

            if len(known_num):
                map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]
                map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(groups)]).long()
            if len(known_bid):
                padded_reference_points[(known_bid.long(), map_known_indice)] = known_bbox_center.to(reference_points.device)
                # padding_query_cat_encoding[(known_bid.long(), map_known_indice)] = known_query_cat_encoding
            
            # padding_query_cat_encoding = padding_query_cat_encoding.permute(0,2,1)
            tgt_size = pad_size + self.num_query
            attn_mask = torch.ones(tgt_size, tgt_size).to(reference_points.device) < 0
            # match query cannot see the reconstruct
            attn_mask[pad_size:, :pad_size] = True
            # reconstruct cannot see each other
            for i in range(groups):
                if i == 0:
                    attn_mask[single_pad * i : single_pad * (i + 1), single_pad * (i + 1) : pad_size] = True
                if i == groups - 1:
                    attn_mask[single_pad * i : single_pad * (i + 1), : single_pad * i] = True
                else:
                    attn_mask[single_pad * i : single_pad * (i + 1), single_pad * (i + 1) : pad_size] = True
                    attn_mask[single_pad * i : single_pad * (i + 1), : single_pad * i] = True

            mask_dict = {
                "known_indice": torch.as_tensor(known_indice).long(),
                "batch_idx": torch.as_tensor(batch_idx).long(),
                "map_known_indice": torch.as_tensor(map_known_indice).long(),
                "known_lbs_bboxes": (known_labels, known_bboxs),
                "known_labels_raw": known_labels_raw,
                "know_idx": know_idx,
                "pad_size": pad_size,
            }

        else:
            padded_reference_points = reference_points
            attn_mask = None
            mask_dict = None
            # padding_query_cat_encoding = query_cat_encoding

        return padded_reference_points, attn_mask, mask_dict
        
    @force_fp32(apply_to=('preds_dicts'))
    def loss(self, gt_bboxes_3d, gt_labels_3d, preds_dicts, **kwargs):
        """"Loss function.
        Args:
            gt_bboxes_3d (list[LiDARInstance3DBoxes]): batch_size * (num_gts, 9)
            gt_labels_3d (list[Tensor]): Ground truth class indices. batch_size * (num_gts, )
            preds_dicts(tuple[list[dict]]): nb_tasks x num_lvl
                center: (num_dec, batch_size, num_query, 2)
                height: (num_dec, batch_size, num_query, 1)
                dim: (num_dec, batch_size, num_query, 3)
                rot: (num_dec, batch_size, num_query, 2)
                vel: (num_dec, batch_size, num_query, 2)
                cls_logits: (num_dec, batch_size, num_query, task_classes)
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        num_decoder = preds_dicts[0][0]['center'].shape[0]
        all_pred_bboxes, all_pred_logits = collections.defaultdict(list), collections.defaultdict(list)

        for task_id, preds_dict in enumerate(preds_dicts, 0):
            for dec_id in range(num_decoder):
                pred_bbox = torch.cat(
                    (preds_dict[0]['center'][dec_id], preds_dict[0]['height'][dec_id],
                    preds_dict[0]['dim'][dec_id], preds_dict[0]['rot'][dec_id],
                    preds_dict[0]['vel'][dec_id]),
                    dim=-1
                )
                all_pred_bboxes[dec_id].append(pred_bbox)
                all_pred_logits[dec_id].append(preds_dict[0]['cls_logits'][dec_id])
        all_pred_bboxes = [all_pred_bboxes[idx] for idx in range(num_decoder)]
        all_pred_logits = [all_pred_logits[idx] for idx in range(num_decoder)]

        loss_cls, loss_bbox = multi_apply(
            self.loss_single, all_pred_bboxes, all_pred_logits,
            [gt_bboxes_3d for _ in range(num_decoder)],
            [gt_labels_3d for _ in range(num_decoder)], 
        )

        loss_dict = dict()
        loss_dict['loss_cls'] = loss_cls[-1]
        loss_dict['loss_bbox'] = loss_bbox[-1]

        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(loss_cls[:-1],
                                           loss_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            num_dec_layer += 1
        
        dn_pred_bboxes, dn_pred_logits = collections.defaultdict(list), collections.defaultdict(list)
        dn_mask_dicts = collections.defaultdict(list)
        for task_id, preds_dict in enumerate(preds_dicts, 0):
            for dec_id in range(num_decoder):
                pred_bbox = torch.cat(
                    (preds_dict[0]['dn_center'][dec_id], preds_dict[0]['dn_height'][dec_id],
                    preds_dict[0]['dn_dim'][dec_id], preds_dict[0]['dn_rot'][dec_id],
                    preds_dict[0]['dn_vel'][dec_id]),
                    dim=-1
                )
                dn_pred_bboxes[dec_id].append(pred_bbox)
                dn_pred_logits[dec_id].append(preds_dict[0]['dn_cls_logits'][dec_id])
                dn_mask_dicts[dec_id].append(preds_dict[0]['dn_mask_dict'])
        dn_pred_bboxes = [dn_pred_bboxes[idx] for idx in range(num_decoder)]
        dn_pred_logits = [dn_pred_logits[idx] for idx in range(num_decoder)]
        dn_mask_dicts = [dn_mask_dicts[idx] for idx in range(num_decoder)]
        dn_loss_cls, dn_loss_bbox = multi_apply(
            self.dn_loss_single, dn_pred_bboxes, dn_pred_logits, dn_mask_dicts
        )

        loss_dict['dn_loss_cls'] = dn_loss_cls[-1]
        loss_dict['dn_loss_bbox'] = dn_loss_bbox[-1]
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(dn_loss_cls[:-1],
                                           dn_loss_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.dn_loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.dn_loss_bbox'] = loss_bbox_i
            num_dec_layer += 1

        sparse_hm_voxel = preds_dict[0]['sparse_heatmap']
        spatial_shape, batch_index, voxel_indices, spatial_indices, num_voxels = self._get_voxel_infos(sparse_hm_voxel)
        voxel_hp_target = multi_apply(
            self.sparse_hp_target_single,
            gt_bboxes_3d,
            gt_labels_3d,
            num_voxels,
            spatial_indices,
        )
        # voxel_hp_target = self.sparse_hp_target_single(sparse_hm_voxel, gt_bboxes_3d,gt_labels_3d)
        # TODO: Fix bugs for hp target (uncorrect when batchsize != 1)
        hp_target = [ t.permute(1,0) for t in voxel_hp_target[0]]
        hp_target = torch.cat(hp_target,dim=0)
        pred_hm = sparse_hm_voxel.features.clone()
        loss_heatmap = self.loss_heatmap(clip_sigmoid(pred_hm), hp_target, avg_factor=max(hp_target.eq(1).float().sum().item(), 1))
        # heatmap_target = torch.cat(hp_target, dim=0)
        # loss_heatmap = self.loss_heatmap(clip_sigmoid(preds_dict[0]['dense_heatmap']), heatmap_target, avg_factor=max(heatmap_target.eq(1).float().sum().item(), 1))
        loss_dict['loss_heatmap'] = loss_heatmap
        return loss_dict


    def sparse_hp_target_single(self,gt_bboxes_3d, gt_labels_3d, num_voxels, spatial_indices):
        num_max_objs = 500
        gaussian_overlap = 0.1
        min_radius = 2
        device = gt_labels_3d.device
        gt_bboxes_3d = torch.cat([gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]], dim=1).to(device)
        grid_size = torch.tensor(self.train_cfg['grid_size'])
        pc_range = torch.tensor(self.train_cfg['point_cloud_range'])
        voxel_size = torch.tensor(self.train_cfg['voxel_size'])
        feature_map_size = grid_size[:2] // self.train_cfg['out_size_factor']  # [x_len, y_len]
        # heatmap = gt_bboxes_3d.new_zeros((self.num_classes[0], feature_map_size[1], feature_map_size[0]))
        

        inds = gt_bboxes_3d.new_zeros(num_max_objs).long()
        mask = gt_bboxes_3d.new_zeros(num_max_objs).long()
        heatmap = gt_bboxes_3d.new_zeros(sum(self.num_classes), num_voxels)
        x, y, z = gt_bboxes_3d[:, 0], gt_bboxes_3d[:, 1], gt_bboxes_3d[:, 2]

        coord_x = (x - self.pc_range[0]) / voxel_size[0] / self.downsample_scale
        coord_y = (y - self.pc_range[1]) / voxel_size[1] / self.downsample_scale

        spatial_shape = [self.test_cfg['grid_size'][0] / self.downsample_scale, self.test_cfg['grid_size'][1] / self.downsample_scale]
        coord_x = torch.clamp(coord_x, min=0, max=spatial_shape[1] - 0.5)  # bugfixed: 1e-6 does not work for center.int()
        coord_y = torch.clamp(coord_y, min=0, max=spatial_shape[0] - 0.5)  #

        center = torch.cat((coord_y[:, None], coord_x[:, None]), dim=-1)
        center_int = center.int()
        center_int_float = center_int.float()

        dx, dy, dz = gt_bboxes_3d[:, 3], gt_bboxes_3d[:, 4], gt_bboxes_3d[:, 5]
        dx = dx / voxel_size[0] / self.downsample_scale
        dy = dy / voxel_size[1] / self.downsample_scale

        radius = self.gaussian_radius(dx, dy, min_overlap=gaussian_overlap)
        radius = torch.clamp_min(radius.int(), min=min_radius)

        for k in range(min(num_max_objs, gt_bboxes_3d.shape[0])):
            if dx[k] <= 0 or dy[k] <= 0:
                continue

            if not (0 <= center_int[k][0] <= spatial_shape[1] and 0 <= center_int[k][1] <= spatial_shape[0]):
                continue

            cur_class_id = (gt_labels_3d[k]).long()
            distance = self.distance(spatial_indices, center[k])
            inds[k] = distance.argmin()
            mask[k] = 1

            # gt_center
            self.draw_gaussian_to_heatmap_voxels(heatmap[cur_class_id], distance, radius[k].item() * 1)

            # nearnest
            self.draw_gaussian_to_heatmap_voxels(heatmap[cur_class_id], self.distance(spatial_indices, spatial_indices[inds[k]]), radius[k].item() * 1)

        return [heatmap]
    
    def draw_gaussian_to_heatmap_voxels(self, heatmap, distances, radius, k=1):
    
        diameter = 2 * radius + 1
        sigma = diameter / 6
        masked_gaussian = torch.exp(- distances / (2 * sigma * sigma))

        torch.max(heatmap, masked_gaussian, out=heatmap)

        return heatmap

    def distance(self, voxel_indices, center):
        distances = ((voxel_indices - center.unsqueeze(0))**2).sum(-1)
        return distances


    def _get_voxel_infos(self, x):
        spatial_shape = x.spatial_shape
        voxel_indices = x.indices
        spatial_indices = []
        num_voxels = []
        batch_size = x.batch_size
        batch_index = voxel_indices[:, 0]

        for bs_idx in range(batch_size):
            batch_inds = batch_index==bs_idx
            spatial_indices.append(voxel_indices[batch_inds][:, [1, 2]]) # y, x
            num_voxels.append(batch_inds.sum())

        return spatial_shape, batch_index, voxel_indices, spatial_indices, num_voxels


    def gaussian_radius(self, height, width, min_overlap=0.5):
        """
        Args:
            height: (N)
            width: (N)
            min_overlap:
        Returns:
        """
        a1 = 1
        b1 = (height + width)
        c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
        sq1 = (b1 ** 2 - 4 * a1 * c1).sqrt()
        r1 = (b1 + sq1) / 2

        a2 = 4
        b2 = 2 * (height + width)
        c2 = (1 - min_overlap) * width * height
        sq2 = (b2 ** 2 - 4 * a2 * c2).sqrt()
        r2 = (b2 + sq2) / 2

        a3 = 4 * min_overlap
        b3 = -2 * min_overlap * (height + width)
        c3 = (min_overlap - 1) * width * height
        sq3 = (b3 ** 2 - 4 * a3 * c3).sqrt()
        r3 = (b3 + sq3) / 2
        
        ret = torch.min(torch.min(r1, r2), r3)
        return ret

    def query_embed(self, ref_points, img_metas):
        ref_points = inverse_sigmoid(ref_points.clone()).sigmoid()
        bev_embeds = self._bev_query_embed(ref_points, img_metas)
        return bev_embeds 


    def _get_targets_single(self, gt_bboxes_3d, gt_labels_3d, pred_bboxes, pred_logits):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            
            gt_bboxes_3d (Tensor):  LiDARInstance3DBoxes(num_gts, 9)
            gt_labels_3d (Tensor): Ground truth class indices (num_gts, )
            pred_bboxes (list[Tensor]): num_tasks x (num_query, 10)
            pred_logits (list[Tensor]): num_tasks x (num_query, task_classes)
        Returns:
            tuple[Tensor]: a tuple containing the following.
                - labels_tasks (list[Tensor]): num_tasks x (num_query, ).
                - label_weights_tasks (list[Tensor]): num_tasks x (num_query, ).
                - bbox_targets_tasks (list[Tensor]): num_tasks x (num_query, 9).
                - bbox_weights_tasks (list[Tensor]): num_tasks x (num_query, 10).
                - pos_inds (list[Tensor]): num_tasks x Sampled positive indices.
                - neg_inds (Tensor): num_tasks x Sampled negative indices.
        """
        device = gt_labels_3d.device
        gt_bboxes_3d = torch.cat(
            (gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]), dim=1
        ).to(device)
        
        task_masks = []
        flag = 0
        for class_name in self.class_names:
            task_masks.append([
                torch.where(gt_labels_3d == class_name.index(i) + flag)
                for i in class_name
            ])
            flag += len(class_name)
        
        task_boxes = []
        task_classes = []
        flag2 = 0
        for idx, mask in enumerate(task_masks):
            task_box = []
            task_class = []
            for m in mask:
                task_box.append(gt_bboxes_3d[m])
                task_class.append(gt_labels_3d[m] - flag2)
            task_boxes.append(torch.cat(task_box, dim=0).to(device))
            task_classes.append(torch.cat(task_class).long().to(device))
            flag2 += len(mask)
        
        def task_assign(bbox_pred, logits_pred, gt_bboxes, gt_labels, num_classes):
            num_bboxes = bbox_pred.shape[0]
            assign_results = self.assigner.assign(bbox_pred, logits_pred, gt_bboxes, gt_labels)
            sampling_result = self.sampler.sample(assign_results, bbox_pred, gt_bboxes)
            pos_inds, neg_inds = sampling_result.pos_inds, sampling_result.neg_inds
            # label targets
            labels = gt_bboxes.new_full((num_bboxes, ),
                                    num_classes,
                                    dtype=torch.long)
            labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
            label_weights = gt_bboxes.new_ones(num_bboxes)
            # bbox_targets
            code_size = gt_bboxes.shape[1]
            bbox_targets = torch.zeros_like(bbox_pred)[..., :code_size]
            bbox_weights = torch.zeros_like(bbox_pred)
            bbox_weights[pos_inds] = 1.0
            
            if len(sampling_result.pos_gt_bboxes) > 0:
                bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
            return labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds

        labels_tasks, labels_weights_tasks, bbox_targets_tasks, bbox_weights_tasks, pos_inds_tasks, neg_inds_tasks\
             = multi_apply(task_assign, pred_bboxes, pred_logits, task_boxes, task_classes, self.num_classes)
        
        return labels_tasks, labels_weights_tasks, bbox_targets_tasks, bbox_weights_tasks, pos_inds_tasks, neg_inds_tasks
            
    def get_targets(self, gt_bboxes_3d, gt_labels_3d, preds_bboxes, preds_logits):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            gt_bboxes_3d (list[LiDARInstance3DBoxes]): batch_size * (num_gts, 9)
            gt_labels_3d (list[Tensor]): Ground truth class indices. batch_size * (num_gts, )
            pred_bboxes (list[list[Tensor]]): batch_size x num_task x [num_query, 10].
            pred_logits (list[list[Tensor]]): batch_size x num_task x [num_query, task_classes]
        Returns:
            tuple: a tuple containing the following targets.
                - task_labels_list (list(list[Tensor])): num_tasks x batch_size x (num_query, ).
                - task_labels_weight_list (list[Tensor]): num_tasks x batch_size x (num_query, )
                - task_bbox_targets_list (list[Tensor]): num_tasks x batch_size x (num_query, 9)
                - task_bbox_weights_list (list[Tensor]): num_tasks x batch_size x (num_query, 10)
                - num_total_pos_tasks (list[int]): num_tasks x Number of positive samples
                - num_total_neg_tasks (list[int]): num_tasks x Number of negative samples.
        """
        (labels_list, labels_weight_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
            self._get_targets_single, gt_bboxes_3d, gt_labels_3d, preds_bboxes, preds_logits
        )
        task_num = len(labels_list[0])
        num_total_pos_tasks, num_total_neg_tasks = [], []
        task_labels_list, task_labels_weight_list, task_bbox_targets_list, \
            task_bbox_weights_list = [], [], [], []

        for task_id in range(task_num):
            num_total_pos_task = sum((inds[task_id].numel() for inds in pos_inds_list))
            num_total_neg_task = sum((inds[task_id].numel() for inds in neg_inds_list))
            num_total_pos_tasks.append(num_total_pos_task)
            num_total_neg_tasks.append(num_total_neg_task)
            task_labels_list.append([labels_list[batch_idx][task_id] for batch_idx in range(len(gt_bboxes_3d))])
            task_labels_weight_list.append([labels_weight_list[batch_idx][task_id] for batch_idx in range(len(gt_bboxes_3d))])
            task_bbox_targets_list.append([bbox_targets_list[batch_idx][task_id] for batch_idx in range(len(gt_bboxes_3d))])
            task_bbox_weights_list.append([bbox_weights_list[batch_idx][task_id] for batch_idx in range(len(gt_bboxes_3d))])
        
        return (task_labels_list, task_labels_weight_list, task_bbox_targets_list,
                task_bbox_weights_list, num_total_pos_tasks, num_total_neg_tasks)
        
    def _loss_single_task(self,
                          pred_bboxes,
                          pred_logits,
                          labels_list,
                          labels_weights_list,
                          bbox_targets_list,
                          bbox_weights_list,
                          num_total_pos,
                          num_total_neg):
        """"Compute loss for single task.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            pred_bboxes (Tensor): (batch_size, num_query, 10)
            pred_logits (Tensor): (batch_size, num_query, task_classes)
            labels_list (list[Tensor]): batch_size x (num_query, )
            labels_weights_list (list[Tensor]): batch_size x (num_query, )
            bbox_targets_list(list[Tensor]): batch_size x (num_query, 9)
            bbox_weights_list(list[Tensor]): batch_size x (num_query, 10)
            num_total_pos: int
            num_total_neg: int
        Returns:
            loss_cls
            loss_bbox 
        """
        labels = torch.cat(labels_list, dim=0)
        labels_weights = torch.cat(labels_weights_list, dim=0)
        bbox_targets = torch.cat(bbox_targets_list, dim=0)
        bbox_weights = torch.cat(bbox_weights_list, dim=0)
        
        pred_bboxes_flatten = pred_bboxes.flatten(0, 1)
        pred_logits_flatten = pred_logits.flatten(0, 1)
        
        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * 0.1
        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            pred_logits_flatten, labels, labels_weights, avg_factor=cls_avg_factor
        )

        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * bbox_weights.new_tensor(self.train_cfg.code_weights)[None, :]

        loss_bbox = self.loss_bbox(
            pred_bboxes_flatten[isnotnan, :10],
            normalized_bbox_targets[isnotnan, :10],
            bbox_weights[isnotnan, :10],
            avg_factor=num_total_pos
        )

        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox) 
        return loss_cls, loss_bbox

    def loss_single(self,
                    pred_bboxes,
                    pred_logits,
                    gt_bboxes_3d,
                    gt_labels_3d):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            pred_bboxes (list[Tensor]): num_tasks x [bs, num_query, 10].
            pred_logits (list(Tensor]): num_tasks x [bs, num_query, task_classes]
            gt_bboxes_3d (list[LiDARInstance3DBoxes]): batch_size * (num_gts, 9)
            gt_labels_list (list[Tensor]): Ground truth class indices. batch_size * (num_gts, )
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        batch_size = pred_bboxes[0].shape[0]
        pred_bboxes_list, pred_logits_list = [], []
        for idx in range(batch_size):
            pred_bboxes_list.append([task_pred_bbox[idx] for task_pred_bbox in pred_bboxes])
            pred_logits_list.append([task_pred_logits[idx] for task_pred_logits in pred_logits])
        cls_reg_targets = self.get_targets(
            gt_bboxes_3d, gt_labels_3d, pred_bboxes_list, pred_logits_list
        )
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        loss_cls_tasks, loss_bbox_tasks = multi_apply(
            self._loss_single_task, 
            pred_bboxes,
            pred_logits,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_pos,
            num_total_neg
        )


        return sum(loss_cls_tasks), sum(loss_bbox_tasks)
    
    def _dn_loss_single_task(self,
                             pred_bboxes,
                             pred_logits,
                             mask_dict):
        known_labels, known_bboxs = mask_dict['known_lbs_bboxes']
        map_known_indice = mask_dict['map_known_indice'].long()
        known_indice = mask_dict['known_indice'].long()
        batch_idx = mask_dict['batch_idx'].long()
        bid = batch_idx[known_indice]
        known_labels_raw = mask_dict['known_labels_raw']
        
        pred_logits = pred_logits[(bid, map_known_indice)]
        pred_bboxes = pred_bboxes[(bid, map_known_indice)]
        num_tgt = known_indice.numel()

        # filter task bbox
        task_mask = known_labels_raw != pred_logits.shape[-1]
        task_mask_sum = task_mask.sum()
        
        if task_mask_sum > 0:
            # pred_logits = pred_logits[task_mask]
            # known_labels = known_labels[task_mask]
            pred_bboxes = pred_bboxes[task_mask]
            known_bboxs = known_bboxs[task_mask]

        # classification loss
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_tgt * 3.14159 / 6 * self.split * self.split  * self.split
        # if self.sync_cls_avg_factor:
        #     cls_avg_factor = reduce_mean(
        #         pred_logits.new_tensor([cls_avg_factor]))
        
        label_weights = torch.ones_like(known_labels)
        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            pred_logits, known_labels.long(), label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_tgt = loss_cls.new_tensor([num_tgt])
        num_tgt = torch.clamp(reduce_mean(num_tgt), min=1).item()

        # regression L1 loss
        normalized_bbox_targets = normalize_bbox(known_bboxs, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = torch.ones_like(pred_bboxes)
        bbox_weights = bbox_weights * bbox_weights.new_tensor(self.train_cfg.code_weights)[None, :]
        # bbox_weights[:, 6:8] = 0
        loss_bbox = self.loss_bbox(
                pred_bboxes[isnotnan, :10], normalized_bbox_targets[isnotnan, :10], bbox_weights[isnotnan, :10], avg_factor=num_tgt)
 
        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)

        if task_mask_sum == 0:
            # loss_cls = loss_cls * 0.0
            loss_bbox = loss_bbox * 0.0

        return self.dn_weight * loss_cls, self.dn_weight * loss_bbox

    def dn_loss_single(self,
                       pred_bboxes,
                       pred_logits,
                       dn_mask_dict):
        loss_cls_tasks, loss_bbox_tasks = multi_apply(
            self._dn_loss_single_task, pred_bboxes, pred_logits, dn_mask_dict
        )
        return sum(loss_cls_tasks), sum(loss_bbox_tasks)

    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, img=None, rescale=False):
        preds_dicts = self.bbox_coder.decode(preds_dicts)
        num_samples = len(preds_dicts)
        
        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            bboxes = img_metas[i]['box_type_3d'](bboxes, bboxes.size(-1))
            scores = preds['scores']
            labels = preds['labels']
            ret_list.append([bboxes, scores, labels])
        return ret_list