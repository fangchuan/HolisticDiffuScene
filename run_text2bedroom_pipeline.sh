export CUDA_VISIBLE_DEVICES=1
NUM_SAMPLES=1
OUTPUT_FOLDER=/mnt/nas_3dv/hdd1/fangchuan/rebuttal_experiments/

eval "$(conda shell.bash hook)"
conda activate structured3d

# run layout sampling
cd /mnt/nas_3dv/hdd1/fangchuan/HolisticDiffuScene/
MODEL_FLAGS="--layout_channels 32 --layout_size 23 --num_channels 128 --num_res_blocks 3 --b_learn_sigma True  --b_class_cond False --b_text_cond True --use_input_encoding False"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine  --timestep_respacing ddim200 --use_ddim True"
python scripts/st3d_room_layout_sample.py --data_dir /mnt/nas_3dv/hdd1/datasets/Structured3d/preprocessed/text2pano/test/bedroom/ \
 --model_path log/ST3D_bedroom_textcondition_openai-2023-09-08-15-04-50-375770/ema_0.9999_180000.pt \
 $MODEL_FLAGS \
 $DIFFUSION_FLAGS \
 --num_samples $NUM_SAMPLES  --log_dir $OUTPUT_FOLDER  --room_type 'bedroom'

# # run panorama sampling
# conda activate control-v11
# PANO_INPUT_FOLDER=$OUTPUT_FOLDER
# CKPT_PATH="/mnt/nas_3dv/hdd1/fangchuan/Layout_Controlnet/ckpts/control_v11p_sd15_seg_bedroom_fullres_32000.ckpt"
# cd /mnt/nas_3dv/hdd1/fangchuan/Layout_Controlnet/scripts
# python st3d_panorama_sample.py --input_folder $PANO_INPUT_FOLDER --ckpt_filepath $CKPT_PATH

# # run super-resolution
# python scripts/sr_val_ddpm_text_T_vqganfin_oldcanvas.py --config configs/stableSRNew/v2-finetune_text_T_512.yaml \
#                                                         --ckpt ../stablesr_000117.ckpt \
#                                                         --vqgan_ckpt ../vqgan_cfw_00011.ckpt \
#                                                         --init-img input_img_1 \
#                                                         --outdir output_img_1 \
#                                                         --ddpm_steps 200 --dec_w 0.5 --colorfix_type adain
# run panoramic reconstrcution
# python st3d_panorama_recons.py --input_folder $PANO_INPUT_FOLDER


