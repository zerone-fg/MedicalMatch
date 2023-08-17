from __future__ import division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class MLP(nn.Module):

    def __init__(self, dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(dim, dim*4)
        self.fc2 = nn.Linear(dim*4, dim)
        self.act = nn.functional.gelu
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class MultiScaleAtten(nn.Module):
    def __init__(self, dim):
        super(MultiScaleAtten, self).__init__()
        self.qkv_linear = nn.Linear(dim, dim * 3)
        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Linear(dim, dim)
        self.num_head = 8
        self.scale = (dim // self.num_head)**0.5

    def forward(self, x):
        B, num_blocks, _, _, C = x.shape  # (B, num_blocks, num_blocks, N, C)
        qkv = self.qkv_linear(x).reshape(B, num_blocks, num_blocks, -1, 3, self.num_head, C // self.num_head).permute(4, 0, 1, 2, 5, 3, 6).contiguous() # (3, B, num_block, num_block, head, N, C)
        q, k, v = qkv[0], qkv[1], qkv[2]
        atten = q @ k.transpose(-1, -2).contiguous()
        atten = self.softmax(atten)
        atten_value = (atten @ v).transpose(-2, -3).contiguous().reshape(B, num_blocks, num_blocks, -1, C)
        atten_value = self.proj(atten_value)  # (B, num_block, num_block, N, C)
        return atten_value


class InterTransBlock(nn.Module):
    def __init__(self, dim):
        super(InterTransBlock, self).__init__()
        self.SlayerNorm_1 = nn.LayerNorm(dim, eps=1e-6)
        self.SlayerNorm_2 = nn.LayerNorm(dim, eps=1e-6)
        self.Attention = MultiScaleAtten(dim)
        self.FFN = MLP(dim)

    def forward(self, x):
        h = x  # (B, N, H)
        x = self.SlayerNorm_1(x)

        x = self.Attention(x)  # padding 到right_size
        x = h + x

        h = x
        x = self.SlayerNorm_2(x)

        x = self.FFN(x)
        x = h + x

        return x


class SpatialAwareTrans(nn.Module):
    def __init__(self, dim=256, num=1):  # (224*64, 112*128, 56*256, 28*256, 14*512) dim = 256
        super(SpatialAwareTrans, self).__init__()
        self.ini_win_size = 2
        self.channels = [32, 64, 128]
        self.dim = dim
        self.depth = 3
        self.fc_module = nn.ModuleList()
        self.fc_rever_module = nn.ModuleList()
        self.num = num
        for i in range(self.depth):
            self.fc_module.append(nn.Linear(self.channels[i], self.dim))

        for i in range(self.depth):
            self.fc_rever_module.append(nn.Linear(self.dim, self.channels[i]))

        self.group_attention = []
        for i in range(self.num):
            self.group_attention.append(InterTransBlock(dim))
        self.group_attention = nn.Sequential(*self.group_attention)
        self.split_list = [8 * 8, 4 * 4, 2 * 2]

        ### 窗口大小划分分别为 28:2, 56:4, 112:8

    def forward(self, x):
        # project channel dimension to 256
        x = [self.fc_module[i](item.permute(0, 2, 3, 1)) for i, item in enumerate(x)]  # [(B, H, W, C)]
        # Patch Matching
        for j, item in enumerate(x):
            B, H, W, C = item.shape
            win_size = self.ini_win_size ** (self.depth - j)
            item = item.reshape(B, H // win_size, win_size, W // win_size, win_size, C).permute(0, 1, 3, 2, 4, 5).contiguous()
            item = item.reshape(B, H // win_size, W // win_size, win_size * win_size, C).contiguous()
            x[j] = item
        x = tuple(x)
        x = torch.cat(x, dim=-2)  # (B, H // win, W // win, N, C)
        # Scale fusion
        for i in range(self.num):
            x = self.group_attention[i](x)  # (B, H // win_size, W // win_size, win_size*win_size, C)

        x = torch.split(x, self.split_list, dim=-2)
        x = list(x)
        # patch reversion
        for j, item in enumerate(x):
            B, num_blocks, _, N, C = item.shape
            win_size = self.ini_win_size ** (self.depth - j)
            item = item.reshape(B, num_blocks, num_blocks, win_size, win_size, C).permute(0, 1, 3, 2, 4, 5).contiguous().reshape(B, num_blocks*win_size, num_blocks*win_size, C)
            item = self.fc_rever_module[j](item).permute(0, 3, 1, 2).contiguous()
            x[j] = item
        return x


def gen_indices(i, k, s):
    assert i >= k, 'Sample size has to be bigger than the patch size'
    for j in range(0, i - k + 1, s):
        yield j
    if j + k < i:
        yield i - k


def get_grid(data_shape, grid_num, overlap=False):
    grids = []
    i_y, i_x = data_shape
    k_y, k_x = i_y // grid_num, i_x // grid_num
    if overlap == False :
        s_y, s_x = k_y, k_x
    else:
        s_y, s_x= k_y // 2, k_x // 2
    y_steps = gen_indices(i_y, k_y, s_y)
    for y in y_steps:
        x_steps = gen_indices(i_x, k_x, s_x)
        for x in x_steps:
            grid_idx = (
                    slice(y, y + k_y),
                    slice(x, x + k_x)
            )
            grids.append(grid_idx)
    return grids


class ConvBlock(nn.Module):
    """two convolution layers with batch norm and leaky relu"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(ConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv_conv(x)


class DownBlock(nn.Module):
    """Downsampling followed by ConvBlock"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, dropout_p)

        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """Upssampling followed by ConvBlock"""

    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p,
                 bilinear=True):
        super(UpBlock, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels1, in_channels2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        self.dropout = self.params['dropout']
        assert (len(self.ft_chns) == 5)
        self.in_conv = ConvBlock(
            self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1 = DownBlock(
            self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2 = DownBlock(
            self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3 = DownBlock(
            self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4 = DownBlock(
            self.ft_chns[3], self.ft_chns[4], self.dropout[4])

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return [x0, x1, x2, x3, x4]


class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
                                  kernel_size=3, padding=1)

    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        output = self.out_conv(x)
        return output


class UNet(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params)
        self.decoder = Decoder(params)
        self.grid_num = 1
        self.overlap = True
        self.fuse = nn.Sequential(
            nn.Conv2d(224, 224, 3, 1, padding=1),
            nn.BatchNorm2d(224),
            nn.ReLU())
        self.reduction = nn.Conv2d(224, 128, 1)

        self.conv1 = nn.Conv2d(128, class_num, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(128, class_num, kernel_size=1, stride=1, padding=0, bias=True)

        self.inter_trans = SpatialAwareTrans(dim=128)

    def forward(self, x, mask_x=None, need_fp=False, cluster=False, cls_aug=False):
        feature = self.encoder(x)
        if need_fp:
            outs = self.decoder([torch.cat((feat, nn.Dropout2d(0.5)(feat))) for feat in feature])
            return outs.chunk(2)

        output = self.decoder(feature)

        if cluster:
            feat_1 = feature[1].clone()  ### (32, 112, 112)
            feat_3 = feature[3].clone()  ### (128, 28, 28)
            feat_2 = feature[2].clone()  #### (64, 56, 56)

            feat_inter = self.inter_trans([feat_1, feat_2, feat_3])

            feat_1_1, feat_2_1, feat_3_1 = feat_inter
            feat_1_1 = F.interpolate(feat_1_1, (56, 56), mode='bilinear', align_corners=True)
            feat_3_1 = F.interpolate(feat_3_1, (56, 56), mode='bilinear', align_corners=True)

            feat_fuse = self.fuse(torch.cat((feat_1_1, feat_2_1, feat_3_1), dim=1))
            feat_fuse = self.reduction(feat_fuse)

            if cls_aug:
                feat_x, feat_s = feat_fuse.chunk(2)
                batch_size, C, H, W = feat_x.shape
                grids = get_grid(feat_x.shape[-2:], grid_num=self.grid_num, overlap=self.overlap)
                mask_x = F.interpolate(mask_x.unsqueeze(1).float(), (56, 56), mode='bilinear', align_corners=True).squeeze(1)

                prototype_grids = []  ## (b, cls, grids, channels)
                for b in range(batch_size):
                    prototype_grids_b = self.getFeatures(feat_x[b], mask_x[b], grids)  ### (cls, grids, channels)
                    prototype_grids.append(prototype_grids_b)

                prototype_grids = torch.stack(prototype_grids, dim=0)  ### (b, cls, grids, channels)
                prototype_grids = prototype_grids.permute(1, 2, 0, 3).contiguous()  ### (cls, grids, b, channels)
                prototype_grids = prototype_grids.detach()
                dist = [self.calDist(feat_s, proto, grids, (56, 56)) for proto in prototype_grids]  ####[cls, b, h, w)

                pred = torch.stack(dist, dim=0)
                pred = pred.permute(1, 0, 2, 3).contiguous()
                # out_cls = F.softmax(pred, dim=1)
                out_cls = pred

                output_x, output_s = output.chunk(2)
                # output_s = F.interpolate(output_s.detach(), (56, 56), mode='bilinear', align_corners=True)
                # out_final = out_cls * output_s
                out_final_cls = F.interpolate(out_cls, (224, 224), mode='bilinear', align_corners=True)
                return output_x, out_final_cls
            else:
                b, channels, h_f, w_f = feat_fuse.shape
                f1 = rearrange(self.conv1(feat_fuse), 'n c h w -> n c (h w)')
                f2 = rearrange(self.conv2(feat_fuse), 'n c h w -> n c (h w)')
                corr_map = torch.matmul(f1.transpose(1, 2), f2) / torch.sqrt(torch.tensor(f1.shape[1]).float())
                corr_map = F.softmax(corr_map, dim=-1)

                out_temp = F.interpolate(output.detach(), (h_f, w_f), mode='bilinear', align_corners=True)
                out_temp = rearrange(out_temp, 'n c h w -> n c (h w)')
                out = rearrange(torch.matmul(out_temp, corr_map), 'n c (h w) -> n c h w', h=h_f, w=w_f)
                corr_out = F.interpolate(out, size=(224, 224), mode="bilinear", align_corners=True)

                return corr_out

        return output

    def calDist(self, fts, prototype, grids, img_size):
        """
        Calculate the distance between features and prototypes

        Args:
            fts: input features
                expect shape: N x C x H x W
            prototype: prototype of one semantic class
                expect shape: 1 x C
        """
        H, W = img_size
        dist = torch.zeros((fts.shape[0], H, W)).float().cuda()  ### (12, 56, 56)
        dist = dist-1
        for (i, grid) in enumerate(grids):
            local_dist = []
            for b in range(8):
                local_dist.append(F.cosine_similarity(fts[:, :, grid[0], grid[1]], prototype[i][b][None, ..., None, None].cuda(), dim=1))
            local_dist = torch.stack(local_dist, dim=0) #### (12, 12, 14, 14) 在dim=1上相加
          
            value, _ = torch.max(local_dist, dim=0)
            new_local_dist = value
            dist[:, grid[0], grid[1]] = (dist[:, grid[0], grid[1]] > new_local_dist) * dist[:, grid[0], grid[1]]+(dist[:, grid[0], grid[1]] <= new_local_dist)*new_local_dist

        return dist

    def getFeatures(self, fts, mask, grids):
        """
        Extract foreground and background features via masked average pooling

        Args:
            fts: input features, expect shape: B x C x H' x W'
            mask: binary mask, expect shape: B x H x W
        """
        channel, h_f, w_f = fts.shape
        mask_1 = mask.reshape(-1)
        fts_1 = fts.reshape(channel, -1)

        global_proto = []
        for cls in range(2):
            if torch.sum(mask == cls) > 0:
                global_proto.append(torch.mean(fts_1[:, mask_1 == cls], dim=-1))
            else:
                global_proto.append(torch.zeros(channel))

        ##### local prototype 计算####
        local_proto_grids = []
        for cls in range(2):
            local_proto = []
            for grid in grids:
                mask_local = mask[grid[0], grid[1]]  ### (h, w)
                feat_local = fts[:, grid[0], grid[1]]  ### (c, h, w)

                masked_fts = feat_local[:, mask_local == cls]
                if masked_fts.shape[1] == 0:
                    local_proto.append(global_proto[cls])
                else:
                    local_proto.append(torch.mean(masked_fts, dim=-1))
            local_proto = torch.stack(local_proto, dim=0).cuda()
            local_proto_grids.append(local_proto)

        local_proto_grids = torch.stack(local_proto_grids, dim=0)

        return local_proto_grids


