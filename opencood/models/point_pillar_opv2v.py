import torch
import torch.nn as nn

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.fuse_utils import regroup
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.sub_modules.self_attn import AttFusion

from opencood.models.sub_modules.deformable_fusion import KPN_S
from opencood.loss.mmd_loss import MMDLoss

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



class PointPillarOPV2V(nn.Module):
    def __init__(self, args):
        super(PointPillarOPV2V, self).__init__()

        self.max_cav = args['max_cav']
        # PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)
        # used to downsample the feature map for efficient computation
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
        self.compression = False

        if args['compression'] > 0:
            self.compression = True
            self.naive_compressor = NaiveCompressor(256, args['compression'])

        self.fusion_net = AttFusion(256)

        self.cls_head = nn.Conv2d(128 * 2, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(128 * 2, 7 * args['anchor_number'],
                                  kernel_size=1)

        if args['backbone_fix']:
            self.backbone_fix()
        
        ###jinlong:
        self.args= args
        if args['learnable_weight']:
            self.reweight = KPN_S(in_channel=args['reweight_dim'])

            print('the reweight network is loaded !!!!')

        if args['mmd_loss']:

            self.mmd = MMDLoss(n_kernels=args['reweight_dim'])

            print('the mmd loss is loaded !!!!')

    def backbone_fix(self):
        """
        Fix the parameters of backbone during finetune on timedelay。
        """
        for p in self.pillar_vfe.parameters():
            p.requires_grad = False

        for p in self.scatter.parameters():
            p.requires_grad = False

        for p in self.backbone.parameters():
            p.requires_grad = False

        if self.compression:
            for p in self.naive_compressor.parameters():
                p.requires_grad = False
        if self.shrink_flag:
            for p in self.shrink_conv.parameters():
                p.requires_grad = False

        for p in self.cls_head.parameters():
            p.requires_grad = False
        for p in self.reg_head.parameters():
            p.requires_grad = False

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())

        return split_x
    
    def forward(self, data_dict,model_B=None):
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']
        spatial_correction_matrix = data_dict['spatial_correction_matrix']

        # B, max_cav, 3(dt dv infra), 1, 1
        prior_encoding =\
            data_dict['prior_encoding'].unsqueeze(-1).unsqueeze(-1)

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}
        # n, 4 -> n, c
        batch_dict = self.pillar_vfe(batch_dict)
        # n, c -> N, C, H, W
        batch_dict = self.scatter(batch_dict)
        batch_dict_after_backbone = self.backbone(batch_dict)
        if model_B != None:
            batch_dict_after_backbone_B = model_B(data_dict)['batch_dict_after_backbone']
            split_A = self.regroup(batch_dict_after_backbone['spatial_features_2d'], record_len)
            split_B = self.regroup(batch_dict_after_backbone_B['spatial_features_2d'], record_len)
            fusion_features = []
            bs = len(split_A)
            for i in range(bs):
                data_A = split_A[i][:1,]
                if split_A[i].shape[0] > 1:
                    data_B = split_B[i][1:,]
                    data = torch.cat((data_A,data_B),dim=0)
                else:
                    data = data_A
                fusion_features.append(data)
            
            fusion_features = torch.cat(fusion_features, dim=0)
            spatial_features_2d = fusion_features
        else:
            spatial_features_2d = batch_dict_after_backbone['spatial_features_2d']
        # downsample feature to reduce memory
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)
        # compressor
        if self.compression:
            spatial_features_2d = self.naive_compressor(spatial_features_2d)

        if model_B != None:
            ###jinlong
            if self.args['learnable_weight']:

                out = []

                if self.args['mmd_loss']:
                    origin_fea = spatial_features_2d.clone().detach()
                    split_x_gt = self.regroup(origin_fea, record_len)
                    mmd_loss = []

                split_x = self.regroup(spatial_features_2d, record_len)
                for i in range(len(split_x)):

                    xx = split_x[i]
                    yy = split_x_gt[i]
                    xx_1 = xx.clone()
                    if xx[1:,].size()[0] > 0:
                        # pdb.set_trace()
                        b = xx[1:,].clone()#.detach()

                        b = self.reweight(b, b)

                        

                        xx_1[1:,] = xx[1:,]+b.clone()

                    
                        if self.args['mmd_loss']:

                            mmd_loss.append(self.mmd(yy[1:,], xx_1[1:,]))

                        b=b.cpu()
                        out.append(xx_1)
                    else:
                        out.append(xx)
                    
                    # plot_a = xx[:1,].detach().cpu().numpy()[0]
                    # plot_a = np.mean(plot_a,axis=0)
                    # plot_b = xx[1:,].detach().cpu().numpy()[0]
                    # plot_b = np.mean(plot_b,axis=0)

                    # min_val = np.min(plot_a)
                    # max_val = np.max(plot_a)

                    # normalized_a = (plot_a - min_val) / (max_val - min_val)

                    # plt.imshow(normalized_a, cmap='RdGy', interpolation='nearest')
                    # plt.savefig('ego.png')
                    # plt.clf()


                    # min_val = np.min(plot_b)
                    # max_val = np.max(plot_b)

                    # normalized_b = (plot_b - min_val) / (max_val - min_val)

                    # plt.imshow(normalized_b, cmap='RdGy', interpolation='nearest')
                    # plt.savefig('cav_before.png')
                    # plt.clf()



                    # plot_c = xx_1[1:,].detach().cpu().numpy()[0]
                    # plot_c = np.mean(plot_c,axis=0)
    

                    # min_val = np.min(plot_c)
                    # max_val = np.max(plot_c)

                    # normalized_c = (plot_c - min_val) / (max_val - min_val)

                    # plt.imshow(normalized_c, cmap='RdGy', interpolation='nearest')
                    # plt.savefig('cav_after.png')
                    # plt.clf()


                spatial_features_2d = torch.cat(out, dim=0)



        fused_feature = self.fusion_net(spatial_features_2d, record_len)

        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)

        # output_dict = {'psm': psm,
        #                'rm': rm}
        
        output_dict = {'psm': psm,
                       'rm': rm,
                       'before_feature': spatial_features_2d,
                       'batch_dict_after_backbone':batch_dict_after_backbone}

        if model_B != None:
            if self.args['mmd_loss']:
                output_dict.update({'mmd': sum(mmd_loss)
                                            })
            
        return output_dict
        # if self.training:
        #     feature = []
        #     spatial_features_2d_list = []
        #     for data_per in data_dict:#######[source_data_dict, target_data_dict]####
        #         voxel_features = data_per['processed_lidar']['voxel_features']
        #         voxel_coords = data_per['processed_lidar']['voxel_coords']
        #         voxel_num_points = data_per['processed_lidar']['voxel_num_points']
        #         record_len = data_per['record_len']

        #         batch_dict = {'voxel_features': voxel_features,
        #                     'voxel_coords': voxel_coords,
        #                     'voxel_num_points': voxel_num_points,
        #                     'record_len': record_len}
        #         # n, 4 -> n, c
        #         batch_dict = self.pillar_vfe(batch_dict)
        #         # n, c -> N, C, H, W
        #         batch_dict = self.scatter(batch_dict)
        #         batch_dict = self.backbone(batch_dict)

        #         spatial_features_2d = batch_dict['spatial_features_2d']

        #         # downsample feature to reduce memory
        #         if self.shrink_flag:
        #             spatial_features_2d = self.shrink_conv(spatial_features_2d)
        #         # compressor
        #         if self.compression:
        #             spatial_features_2d = self.naive_compressor(spatial_features_2d)

        #         fused_feature = self.fusion_net(spatial_features_2d, record_len)


        #         feature.append(fused_feature)####[source_feature, target_feature]####
        #         spatial_features_2d_list.append(spatial_features_2d)

        #     output_dict = {'psm': self.cls_head(feature[0]),#source_feature
        #                 'rm': self.reg_head(feature[0]),#source_feature
        #                 'source_feature': feature[0],#source_feature
        #                 'target_feature': feature[1],#target_feature
        #                 'source_multifea': spatial_features_2d_list[0],#source_feature
        #                 'target_multifea': spatial_features_2d_list[1],#target_feature
        #                 'target_psm': self.cls_head(feature[1]),#target_feature
        #                 'target_rm': self.reg_head(feature[1])#target_feature
        #                 }
        # else:
        #     voxel_features = data_dict['processed_lidar']['voxel_features']
        #     voxel_coords = data_dict['processed_lidar']['voxel_coords']
        #     voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        #     record_len = data_dict['record_len']

        #     batch_dict = {'voxel_features': voxel_features,
        #                 'voxel_coords': voxel_coords,
        #                 'voxel_num_points': voxel_num_points,
        #                 'record_len': record_len}
        #     # n, 4 -> n, c
        #     batch_dict = self.pillar_vfe(batch_dict)
        #     # n, c -> N, C, H, W
        #     batch_dict = self.scatter(batch_dict)
        #     batch_dict = self.backbone(batch_dict)

        #     spatial_features_2d = batch_dict['spatial_features_2d']

        #     # downsample feature to reduce memory
        #     if self.shrink_flag:
        #         spatial_features_2d = self.shrink_conv(spatial_features_2d)
        #     # compressor
        #     if self.compression:
        #         spatial_features_2d = self.naive_compressor(spatial_features_2d)

        #     fused_feature = self.fusion_net(spatial_features_2d, record_len)

        #     # pdb.set_trace()


        #     psm = self.cls_head(fused_feature)
        #     rm = self.reg_head(fused_feature)

        #     output_dict = {'psm': psm,
        #                 'rm': rm}

        # return output_dict