import copy
import torch
import torch.nn as nn
from ultralytics.utils.tal import dist2bbox, make_anchors
import math
import torch.nn.functional as F
def autopad(k, p=None, d=1):  
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k] 
    return p
class Conv(nn.Module):
    default_act = nn.SiLU()  
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    def forward_fuse(self, x):
        return self.act(self.conv(x))
class DFL(nn.Module):
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1
    def forward(self, x):
        b, c, a = x.shape  
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
class MASEM(nn.Module):
    def __init__(self, in_channels, scale=1, group_num=8, gate_treshold=0.5, compress_c=8):
        super().__init__()
        self.c = in_channels
        self.mid = int(self.c * scale)
        self.gate_treshold = gate_treshold
        self.compress_c = compress_c
        self.conv0 = nn.Sequential(
            nn.Conv2d(self.c, self.c, 3, padding=1, groups=self.c), 
            nn.Conv2d(self.c, self.c, 1) 
        )
        self.conv_spatial = nn.Sequential(
            nn.Conv2d(self.c, self.c, 3, padding=1, groups=self.c),  
            nn.Conv2d(self.c, self.c, 1) 
        )
        # gate
        self.conv1 = nn.Conv2d(self.c, self.c // 2, 1)
        self.conv2 = nn.Conv2d(self.c, self.c // 2, 1)
        self.reduce_conv = nn.Conv2d(self.c, self.c//2, 1)
        self.gn = nn.GroupNorm(num_channels=self.c//2, num_groups=group_num)
        self.sigmoid_gate = nn.Sigmoid()
        # channel
        self.weight_conv1 = nn.Conv2d(self.c//2, self.compress_c, kernel_size=1)
        self.weight_conv2 = nn.Conv2d(self.c//2, self.compress_c, kernel_size=1)
        self.weights_conv = nn.Conv2d(self.compress_c * 2, 2, kernel_size=1)
        # spatial
        self.conv_squeeze = nn.Conv2d(2, 2, 3, padding=1)
        self.conv = nn.Conv2d(self.c, self.c, 1)
    def forward(self, x):
        residual = x
        attn1 = self.conv0(x)  
        attn2 = self.conv_spatial(attn1)  
        # gate
        attn1_conv = self.conv1(attn1)
        attn2_conv = self.conv2(attn2)
        attn_concat = torch.cat([attn1_conv, attn2_conv], dim=1)
        attn_reduced = self.reduce_conv(attn_concat)
        gn_attn = self.gn(attn_reduced)
        w_gamma = self.gn.weight / sum(self.gn.weight)
        w_gamma = w_gamma.view(1, -1, 1, 1)
        reweigts = self.sigmoid_gate(gn_attn * w_gamma)
        w1 = torch.where(reweigts > self.gate_treshold, torch.ones_like(reweigts), reweigts)
        attn1_gated = attn1_conv * w1
        attn2_gated = attn2_conv * w1
        # channel
        weight1 = self.weight_conv1(attn1_gated)
        weight2 = self.weight_conv2(attn2_gated)
        weights_concat = torch.cat([weight1, weight2], dim=1)
        weights = self.weights_conv(weights_concat)
        weights = torch.softmax(weights, dim=1)
        w1_fuse = weights[:, 0:1, :, :]
        w2_fuse = weights[:, 1:2, :, :]
        fused_attn1 = attn1_gated * w1_fuse
        fused_attn2 = attn2_gated * w2_fuse
        fused_attn = torch.cat([fused_attn1, fused_attn2], dim=1)
        # spatial
        avg_spatial = torch.mean(fused_attn, dim=1, keepdim=True)
        max_spatial, _ = torch.max(fused_attn, dim=1, keepdim=True)
        spatial_agg = torch.cat([avg_spatial, max_spatial], dim=1)
        sig_spatial = self.conv_squeeze(spatial_agg).sigmoid()
        spatial_attn = fused_attn * sig_spatial[:, 0:1, :, :] + fused_attn * sig_spatial[:, 1:2, :, :]
        final_attn = self.conv(spatial_attn)
        return residual + x * final_attn
class FASFF(nn.Module):
    def __init__(self, level, ch, multiplier=1, rfb=False, vis=False):
        super(FASFF, self).__init__()
        self.level = level
        self.dim = [int(ch[3] * multiplier), int(ch[2] * multiplier), int(ch[1] * multiplier),
                    int(ch[0] * multiplier)]
        self.inter_dim = self.dim[self.level]
        if level == 0:
            self.stride_level_1 = Conv(int(ch[2] * multiplier), self.inter_dim, 3, 2)
            self.stride_level_2 = Conv(int(ch[1] * multiplier), self.inter_dim, 3, 2)
            self.expand = Conv(self.inter_dim, int(
                ch[3] * multiplier), 3, 1)
        elif level == 1:
            self.compress_level_0 = Conv(
                int(ch[3] * multiplier), self.inter_dim, 1, 1)
            self.stride_level_2 = Conv(
                int(ch[1] * multiplier), self.inter_dim, 3, 2)
            self.expand = Conv(self.inter_dim, int(ch[2] * multiplier), 3, 1)
        elif level == 2:
            self.compress_level_0 = Conv(
                int(ch[2] * multiplier), self.inter_dim, 1, 1)
            self.stride_level_2 = Conv(
                int(ch[0] * multiplier), self.inter_dim, 3, 2)
            self.expand = Conv(self.inter_dim, int(ch[1] * multiplier), 3, 1)
        elif level == 3:
            self.compress_level_0 = Conv(
                int(ch[2] * multiplier), self.inter_dim, 1, 1)
            self.compress_level_1 = Conv(
                int(ch[1] * multiplier), self.inter_dim, 1, 1)
            self.expand = Conv(self.inter_dim, int(
                ch[0] * multiplier), 3, 1)
        compress_c = 8 if rfb else 16
        self.weight_level_0 = Conv(
            self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = Conv(
            self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = Conv(
            self.inter_dim, compress_c, 1, 1)
        self.weight_levels = Conv(
            compress_c * 3, 3, 1, 1)
        self.vis = vis
    def forward(self, x):  
        x_level_add = x[2]
        x_level_0 = x[3]  
        x_level_1 = x[1]  
        x_level_2 = x[0]  
        if self.level == 0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_add)
            level_2_downsampled_inter = F.max_pool2d(
                x_level_1, 3, stride=2, padding=1)
            level_2_resized = self.stride_level_2(level_2_downsampled_inter)
        elif self.level == 1:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(
                level_0_compressed, scale_factor=2, mode='nearest')
            level_1_resized = x_level_add
            level_2_resized = self.stride_level_2(x_level_1)
        elif self.level == 2:
            level_0_compressed = self.compress_level_0(x_level_add)
            level_0_resized = F.interpolate(
                level_0_compressed, scale_factor=2, mode='nearest')
            level_1_resized = x_level_1
            level_2_resized = self.stride_level_2(x_level_2)
        elif self.level == 3:
            level_0_compressed = self.compress_level_0(x_level_add)
            level_0_resized = F.interpolate(
                level_0_compressed, scale_factor=4, mode='nearest')
            x_level_1_compressed = self.compress_level_1(x_level_1)
            level_1_resized = F.interpolate(
                x_level_1_compressed, scale_factor=2, mode='nearest')
            level_2_resized = x_level_2
        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)
        levels_weight_v = torch.cat(
            (level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)
        fused_out_reduced = level_0_resized * levels_weight[:, 0:1, :, :] + \
                            level_1_resized * levels_weight[:, 1:2, :, :] + \
                            level_2_resized * levels_weight[:, 2:, :, :]
        out = self.expand(fused_out_reduced)
        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return out
class DWConv(Conv):
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)
class MASEM_Head(nn.Module):
    dynamic = False  
    export = False 
    end2end = False 
    max_det = 300 
    shape = None
    anchors = torch.empty(0) 
    strides = torch.empty(0) 
    def __init__(self, nc=80, ch=(), multiplier=1, rfb=False, 
                 sru_scale=1, sru_group_num=8, sru_gate_treshold=0.5):
        super().__init__()
        self.nc = nc  
        self.nl = len(ch)  
        assert self.nl == 4, "Input 'ch' must have 4 elements (4-level feature maps) for FASFF"
        self.reg_max = 16  
        self.no = nc + self.reg_max * 4  
        self.stride = torch.zeros(self.nl)  
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
        self.l0_fusion = FASFF(level=0, ch=ch, multiplier=multiplier, rfb=rfb)
        self.l1_fusion = FASFF(level=1, ch=ch, multiplier=multiplier, rfb=rfb)
        self.l2_fusion = FASFF(level=2, ch=ch, multiplier=multiplier, rfb=rfb)
        self.l3_fusion = FASFF(level=3, ch=ch, multiplier=multiplier, rfb=rfb)
        self.sru_list = nn.ModuleList([
            MASEM(
                in_channels=ch[i],
                scale=sru_scale,
                group_num=sru_group_num,
                gate_treshold=sru_gate_treshold
            ) for i in range(self.nl)  
        ])
        if self.end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)
    def forward(self, x):
        x_sru = [self.sru_list[i](xi) for i, xi in enumerate(x)]
        x1 = self.l0_fusion(x_sru)
        x2 = self.l1_fusion(x_sru)
        x3 = self.l2_fusion(x_sru)
        x4 = self.l3_fusion(x_sru)
        x = [x4, x3, x2, x1]  
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  
            return x
        y = self._inference(x)
        return y if self.export else (y, x)
    def forward_end2end(self, x):
        x_sru = [self.sru_list[i](xi) for i, xi in enumerate(x)]
        x_detach = [xi.detach() for xi in x_sru] 
        one2one = [
            torch.cat((self.one2one_cv2[i](x_detach[i]), self.one2one_cv3[i](x_detach[i])), 1) 
            for i in range(self.nl)
        ]
        for i in range(self.nl):
            x_sru[i] = torch.cat((self.cv2[i](x_sru[i]), self.cv3[i](x_sru[i])), 1)
        if self.training:  
            return {"one2many": x_sru, "one2one": one2one}
        y = self._inference(one2one)
        y = self.postprocess(y.permute(0, 2, 1), self.max_det, self.nc)
        return y if self.export else (y, {"one2many": x_sru, "one2one": one2one})
    def _inference(self, x):
        shape = x[0].shape  
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape
        if self.export and self.format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}: 
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        if self.export and self.format in {"tflite", "edgetpu"}:
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides
        return torch.cat((dbox, cls.sigmoid()), 1)
    def bias_init(self):
        m = self  
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  
            a[-1].bias.data[:] = 1.0  
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2) 
        if self.end2end:
            for a, b, s in zip(m.one2one_cv2, m.one2one_cv3, m.stride): 
                a[-1].bias.data[:] = 1.0  
                b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2) 
    def decode_bboxes(self, bboxes, anchors):
        return dist2bbox(bboxes, anchors, xywh=not self.end2end, dim=1)
    @staticmethod
    def postprocess(preds: torch.Tensor, max_det: int, nc: int = 80):
        batch_size, anchors, _ = preds.shape  
        boxes, scores = preds.split([4, nc], dim=-1)
        index = scores.amax(dim=-1).topk(min(max_det, anchors))[1].unsqueeze(-1)
        boxes = boxes.gather(dim=1, index=index.repeat(1, 1, 4))
        scores = scores.gather(dim=1, index=index.repeat(1, 1, nc))
        scores, index = scores.flatten(1).topk(min(max_det, anchors))
        i = torch.arange(batch_size)[..., None]  
        return torch.cat([boxes[i, index // nc], scores[..., None], (index % nc)[..., None].float()], dim=-1)
