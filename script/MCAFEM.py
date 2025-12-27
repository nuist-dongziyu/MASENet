import torch
import torch.nn as nn
import math
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
class GroupConv(Conv):
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True): 
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)
class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e) 
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
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
class MCAFEM(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, p=1, kernel_size=3, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv_first = Conv(c1, 2 * self.c, 1, 1)
        self.cv_final = Conv((4 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.cv_block_1 = Conv(2 * self.c, self.c, 1, 1)
        dim_hid = int(p * 2 * self.c)
        self.cv_block_2 = nn.Sequential(Conv(2 * self.c, dim_hid, 1, 1), GroupConv(dim_hid, dim_hid, kernel_size, 1),
                                      Conv(dim_hid, self.c, 1, 1))
        self.attention = MASEM(in_channels=self.c)
        self.residual_adjust = nn.Sequential(
                                    nn.Conv2d(c1, c1, 3, padding=1, groups=c1, bias=False),  
                                    nn.BatchNorm2d(c1),  
                                    nn.SiLU(),           
                                    nn.Conv2d(c1, c2, 1, bias=False),  
                                    nn.BatchNorm2d(c2),  
                                    nn.SiLU()           
                                )               
    def forward(self, x):
        residual = self.residual_adjust(x)
        y = self.cv_first(x)
        y0 = self.cv_block_1(y)
        y1 = self.cv_block_2(y)
        y2, y3 = y.chunk(2, 1)
        y3 = self.attention(y3)
        y = list((y0, y1, y2, y3))
        y.extend(m(y[-1]) for m in self.m)
        main_output = self.cv_final(torch.cat(y, 1))
        final_output = main_output + residual
        return final_output
