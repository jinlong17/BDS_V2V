#!/bin/sh
###
 # @Descripttion: 
 # @version: 
 # @Author: Jinlong Li CSU PhD
 # @Date: 2022-07-03 18:01:32
 # @LastEditors: Jinlong Li CSU PhD
 # @LastEditTime: 2023-03-31 11:57:01
### 

# source /home/jinlong/anaconda3/etc/profile.d/conda.sh


# conda activate v2xvit



# model="/home/jinlong/4.3D_detection/Detection_CVPR/v2vreal/opencood/logs/point_pillar_intermediate_fusion_2022_10_31_19_48_11"



#################for DA 
# model="/home/jinlong/4.3D_detection/Detection_CVPR/v2vreal/opencood/logs/1.V2Vnoise_point_pillar_intermediate_fusion_2022_11_05_00_20_58"
# hypes_yaml="/home/jinlong/4.3D_detection/Detection_CVPR/v2vreal/opencood/hypes_yaml/point_pillar_intermediate_fusion_da.yaml"

# model="/home/jinlong/4.3D_detection/Detection_CVPR/v2vreal/opencood/logs/2.nonoise_corpbevtlidar_2022_11_06_18_53_27"
# model="/home/jinlong/Desktop/model_da/without_da/CoBEVT"
# model_source="/home/jinlong/4.3D_detection/Detection_CVPR/v2vreal/opencood/logs/2.nonoise_corpbevtlidar_2022_11_06_18_53_27/net_epoch60.pth"
# model_target="/home/jinlong/4.3D_detection/Detection_CVPR/v2vreal/opencood/logs/2.nonoise_corpbevtlidar_2022_11_06_18_53_27/net_epoch60.pth"

# model="/home/jinlong/jinlong_NAS/1.model_trained_saved/DA_V2V_sim2real_early_2023/0.noise_corpbevtlidar_det_V0_2023_03_29_13"

# hypes_yaml="./opencood/hypes_yaml/point_pillar_fax.yaml"

# model="/home/jinlong/4.3D_detection/Detection_CVPR/v2vreal/opencood/logs/point_pillar_fcooper_2022_11_06_18_55_23"

# hypes_yaml="/home/jinlong/4.3D_detection/Detection_CVPR/v2vreal/opencood/hypes_yaml/point_pillar_fcooper.yaml"

# hypes_yaml="/home/jinlong/4.3D_detection/Detection_CVPR/v2vreal/opencood/hypes_yaml/point_pillar_fcooper.yaml"

# hypes_yaml="/home/jinlong/4.3D_detection/Detection_CVPR/v2vreal/opencood/hypes_yaml/point_pillar_transformer.yaml"

# hypes_yaml="/home/jinlong/4.3D_detection/Detection_CVPR/v2vreal/opencood/hypes_yaml/point_pillar_intermediate_fusion.yaml"


# hypes_yaml="/home/jinlong/4.3D_detection/Detection_CVPR/v2vreal/opencood/hypes_yaml/point_pillar_transformer.yaml"

# model="/home/jinlong/4.3D_detection/Detection_CVPR/v2vreal/opencood/logs/point_pillar_v2vnet_2022_11_08_01_55_01"
# model="/home/jinlong/4.3D_detection/Detection_CVPR/v2vreal/opencood/logs/point_pillar_v2vnet_2022_11_08_01_55_01"
# model="/home/jinlong/4.3D_detection/Detection_CVPR/v2vreal/opencood/logs/point_pillar_v2vnet_2022_11_08_01_55_01"
# hypes_yaml="/home/jinlong/4.3D_detection/Detection_CVPR/v2vreal/opencood/hypes_yaml/point_pillar_v2vnet.yaml"



# model="/home/jinlong/4.3D_detection/Detection_CVPR/v2vreal/opencood/logs/2.V2Vno_noise_point_pillar_mcwin_transformer_nocompression_half_hetero_rte_split_att_2022_11_08_01_28_46"
# hypes_yaml="/home/jinlong/4.3D_detection/Detection_CVPR/v2vreal/opencood/hypes_yaml/point_pillar_transformer.yaml"



# run python script
######## using for DA point pillar training
# CUDA_VISIBLE_DEVICES=5  python3 ./opencood/tools/train_DA.py  --hypes_yaml $hypes_yaml --model $model   #--model_source $model_source --model_target $model_target #--model_dir $path  #--model $model  


# hypes_yaml="/home/jinlong/4.3D_detection/Detection_CVPR/v2vreal/opencood/hypes_yaml/point_pillar_fax_stn.yaml"



# hypes_yaml="/home/jinlongli/1.Detection_Set/DA_V2V/opencood/hypes_yaml/point_pillar_fax_deformable.yaml"

#hypes_yaml='/home/jinlongli/1.Detection_Set/4.CrossDa/opencood/hypes_yaml/point_pillar_opv2v.yaml'
#hypes_yaml='/home/jinlongli/1.Detection_Set/4.CrossDa/opencood/hypes_yaml/point_pillar_intermediate_V2VAM.yaml'
hypes_yaml='/home/jinlongli/1.Detection_Set/4.CrossDa/opencood/hypes_yaml/point_pillar_transformer.yaml'
# hypes_yaml='/home/jinlongli/1.Detection_Set/4.CrossDa/opencood/hypes_yaml/point_pillar_fax.yaml'



# hypes_yaml="/home/jinlongli/1.Detection_Set/DA_V2V/opencood/hypes_yaml/point_pillar_v2vnet.yaml"


# hypes_yaml="/home/jinlongli/1.Detection_Set/DA_V2V/opencood/hypes_yaml/point_pillar_fax.yaml"

# hypes_yaml="/home/jinlongli/1.Detection_Set/DA_V2V/opencood/hypes_yaml/point_pillar_late_fusion.yaml"
# hypes_yaml="/home/jinlongli/1.Detection_Set/DA_V2V/opencood/hypes_yaml/point_pillar_early_fusion.yaml"
# hypes_yaml="/home/jinlongli/1.Detection_Set/DA_V2V/opencood/hypes_yaml/point_pillar_opv2v.yaml"

# hypes_yaml="/home/jinlongli/1.Detection_Set/DA_V2V/opencood/hypes_yaml/point_pillar_transformer.yaml"

# hypes_yaml="/home/jinlongli/1.Detection_Set/DA_V2V/opencood/hypes_yaml/point_pillar_intermediate_fusion.yaml"


# model_B="/home/jinlongli/2.model_saved/2.crossdata_DA2023/attfuse_openv2v_2023_07_10_06_32_21/net_epoch56.pth"
# model="/home/jinlongli/2.model_saved/2.crossdata_DA2023/attfuse_v2xset_perfect_2023_09_05_11_48_55/net_epoch56.pth"


# model_B='/home/jinlongli/2.model_saved/2.crossdata_DA2023/V2VAM_openv2v_2023_07_10_06_49_34/net_epoch56.pth'
# model='/home/jinlongli/2.model_saved/2.crossdata_DA2023/V2VAM_v2xset_perfect_2023_09_05_11_49_37/net_epoch120.pth'

model='/home/jinlongli/2.model_saved/2.crossdata_DA2023/v2xvit_openv2v_2023_07_10_06_33_11/net_epoch56.pth'
model_B='/home/jinlongli/2.model_saved/2.crossdata_DA2023/v2xvit_v2xset_perfect_2023_09_05_11_51_01/net_epoch56.pth'

# model_B='/home/jinlongli/2.model_saved/2.crossdata_DA2023/CoBEVT_openv2v_2023_07_10_07_49_29/net_epoch56.pth'
# model='/home/jinlongli/2.model_saved/2.crossdata_DA2023/CoBEVT_v2xset_perfect_2023_09_05_11_50_16/net_epoch60.pth'

######## using for oringinal point pillar training
CUDA_VISIBLE_DEVICES=6 python3 ./opencood/tools/train.py  --hypes_yaml $hypes_yaml  --model $model --model_B_dir $model_B #--model_dir $path  #--model 



# cd ../




# exit the virtual environment
# conda deactivate