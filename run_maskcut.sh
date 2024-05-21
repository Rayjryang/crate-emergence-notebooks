export CUDA_VISIBLE_DEVICES=1

checkpoint_dir=/private/home/yaodongyu/projects/mae_crate/vis_cutler/cutler_crate_base/ckpts/

dataset_dir=/datasets01/COCO/022719




###### L/8 , vis layer 20, input size 448

checkpoint_path=L8_in21k_res_mlp_fixed_decouple_x4_mixup_open_warm10_4096_lr5e5_wd01_dp01_91e_no_randaug_no_labelsm_L8_v3_256_checkpoint.pth  #  crate alpha  L/14 on 21k


echo 'eval '$checkpoint_path

out_dir=vis/mask_cut/json/crate_alpha_L8_21k_layer_20_448_t0.3_return_x
# dim = 576 small
# dim = 192 tiny
# 19167 classes

python maskcut/maskcut_crate.py   --pretrain_path $checkpoint_dir$checkpoint_path \
--tau 0.3 --N 3 \
--fixed_size 448 \
--dataset-path $dataset_dir \
--out-dir $out_dir \
--patch-size 8 \
--arch crate-alpha \
--vit-arch large \
--vis-depth 20 \
--feat-dim 1024 \
--num-classes 19167


###### B/8 , vis layer 10, input size 448

checkpoint_path=B8_in21k_res_mlp_fixed_decouple_x4_no_mixup_open_warm10_4096_lr5e5_wd01_91e_no_randaug_no_label_sm_v3_256_spot_checkpoint.pth  #  crate alpha  L/14 on 21k


echo 'eval '$checkpoint_path

out_dir=vis/mask_cut/json/crate_alpha_B8_21k_layer_10_448_t0.3_return_x
# dim = 576 small
# dim = 192 tiny
# 19167 classes

python maskcut/maskcut_crate.py   --pretrain_path $checkpoint_dir$checkpoint_path \
--tau 0.3 --N 3 \
--fixed_size 448 \
--dataset-path $dataset_dir \
--out-dir $out_dir \
--patch-size 8 \
--arch crate-alpha \
--vit-arch large \
--vis-depth 10 \
--feat-dim 768 \
--num-classes 19167
