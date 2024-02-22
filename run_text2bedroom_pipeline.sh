export CUDA_VISIBLE_DEVICES=1
NUM_SAMPLES=1
OUTPUT_FOLDER=/mnt/nas_3dv/hdd1/datasets/fangchuan/codes/HolisticDiffuScene/sample_results/rebuttal_experiments/

eval "$(conda shell.bash hook)"
conda activate structured3d

# run layout sampling
cd /mnt/nas_3dv/hdd1/datasets/fangchuan/codes/HolisticDiffuScene/
MODEL_FLAGS="--layout_channels 32 --layout_size 23 --num_channels 128 --num_res_blocks 3 --b_learn_sigma True  --b_class_cond False --b_text_cond True --use_input_encoding False"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine --timestep_respacing 250"
python scripts/st3d_room_layout_sample.py --data_dir /mnt/nas_3dv/hdd1/datasets/datasets/Structured3D/text2pano/test/bedroom/ \
 --model_path log/ST3D_bedroom_textcondition_openai-2023-09-08-15-04-50-375770/ema_0.9999_180000.pt \
 $MODEL_FLAGS \
 $DIFFUSION_FLAGS \
 --num_samples $NUM_SAMPLES  --log_dir $OUTPUT_FOLDER  --room_type 'bedroom'

# # run panorama sampling
conda activate control-v11
PANO_INPUT_FOLDER=$OUTPUT_FOLDER/bedroom
CKPT_PATH="/mnt/nas_3dv/hdd1/datasets/fangchuan/codes/Layout_Controlnet/ckpts/control_v11p_sd15_seg_bedroom_fullres_32000.ckpt"
cd /mnt/nas_3dv/hdd1/datasets/fangchuan/codes/Layout_Controlnet/scripts
python st3d_panorama_sample.py --input_folder $PANO_INPUT_FOLDER --ckpt_filepath $CKPT_PATH

# # run super-resolution
# conda activate stablesr
# cd /mnt/nas_3dv/hdd1/datasets/huxiaotao/StableSR/
# SR_INPUT_FOLDER=$OUTPUT_FOLDER/bedroom
# SR_OUTPUT_FOLDER=$OUTPUT_FOLDER/bedroom
# python scripts/sr_val_ddpm_text_T_vqganfin_oldcanvas_text2room_exp.py --config configs/stableSRNew/v2-finetune_text_T_512.yaml \
#                                                         --ckpt ../stablesr_000117.ckpt \
#                                                         --vqgan_ckpt ../vqgan_cfw_00011.ckpt \
#                                                         --init-img $SR_INPUT_FOLDER \
#                                                         --outdir $SR_OUTPUT_FOLDER \
#                                                         --ddpm_steps 200 --dec_w 0.5 --colorfix_type adain
# # run panoramic reconstrcution
# conda activate control-v11
# cd /mnt/nas_3dv/hdd1/datasets/fangchuan/codes/Layout_Controlnet/scripts
# POISSON_EXE=/mnt/nas_3dv/hdd1/datasets/fangchuan/codes/lib/PoissonRecon/Bin/Linux/PoissonRecon
# MESH_TEX_EXE=/mnt/nas_3dv/hdd1/datasets/fangchuan/codes/PanoTexturing/build/apps/pano_texrecon/panorecons
# python st3d_panorama_recons.py --input_folder $PANO_INPUT_FOLDER --poisson_exe_path ${POISSON_EXE} --mesh_tex_exe_path ${MESH_TEX_EXE}


