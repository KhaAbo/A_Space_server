import torch.nn as nn

class FPContext(nn.Module):
    def __init__(self, context_range=[5], channel_list=[256, 512, 1024, 2048], use_dilated_conv=False, use_aspp=False):
        super(FPContext, self).__init__()
        self.num_context_module = len(context_range)
        self.context_range = context_range
        self.context_modules = nn.ModuleList([])
        self.down_convs = nn.ModuleList([])
        self.use_dilated_conv = use_dilated_conv
        self.use_aspp = use_aspp
            
        for i in range(len(channel_list)):
            self.down_convs.append(nn.Conv2d(channel_list[i], 256, 1, 1, 0))

        for i in range(len(channel_list)):
            self.context_modules.append(nn.ModuleList([]))
            for j in range(len(self.context_range)):
                self.context_modules[i].append(nn.Conv2d(256, 256, 3, 1, 1))

    def forward(self, feature_list):
        fp_context_fts = []
        for i in range(len(feature_list)):
            down_conv_ft = self.down_convs[i](feature_list[i])
            fp_context_list = []
            for layer in self.context_modules[i]:
                fp_context_list.append(layer(down_conv_ft))
            fp_context_fts.append(fp_context_list)

        return fp_context_fts


class WiderFaceBaseNetFPContext(nn.Module):
    def __init__(self, backbone, fpn, pred_net, pred_net_1, fp_context=None, out_bb_ft=False):
        super(WiderFaceBaseNetFPContext, self).__init__()
        self.backbone = backbone
        self.fpn = fpn
        self.pred_net = pred_net
        self.pred_net_1 = pred_net_1
        self.out_bb_ft = out_bb_ft

        self.fp_context = fp_context

    def forward(self, x):
        feature_list = self.backbone(x)
        if self.fp_context is not None:
            fp_context_fts = self.fp_context(feature_list)

        pyramid_feature_list = self.fpn(feature_list)

        conf, loc, mask_fp_context_fts = self.pred_net(pyramid_feature_list)
        conf_1 = self.pred_net_1(pyramid_feature_list, fp_context_fts, mask_fp_context_fts) 
        if self.fp_context is not None:
            return conf, loc, conf_1
        else:
            return conf, loc


class WiderFaceBaseNet(nn.Module):
    def __init__(self, backbone, fpn, pred_net, out_bb_ft=False):
        super(WiderFaceBaseNet, self).__init__()
        self.backbone = backbone
        self.fpn = fpn
        self.pred_net = pred_net
        self.out_bb_ft = out_bb_ft

    def forward(self, x):
        feature_list = self.backbone(x)
        fpn_list= self.fpn(feature_list)
        if len(fpn_list) == 2:
            pyramid_feature_list = fpn_list[0]
            dsfd_ft_list = fpn_list[1]
        else:
            pyramid_feature_list = fpn_list
        if len(fpn_list) == 2:
            conf, loc = self.pred_net(pyramid_feature_list, dsfd_ft_list)
            return conf, loc 
        conf, loc = self.pred_net(pyramid_feature_list)
        if self.out_bb_ft:
            return conf, loc, feature_list
        else:
            return conf, loc
