export CUDA_VISIBLE_DEVICES=0
NUM_SAMPLES=30
OUTPUT_FOLDER=/mnt/nas_3dv/hdd1/fangchuan/HolisticDiffuScene/sample_results/2023-09-13-16-06-26-991550/

eval "$(conda shell.bash hook)"
conda activate structured3d

# run layout sampling
# cd /mnt/nas_3dv/hdd1/fangchuan/HolisticDiffuScene/
# MODEL_FLAGS="--layout_channels 34 --layout_size 48 --num_channels 128 --num_res_blocks 3 --b_learn_sigma True  --b_class_cond False --b_text_cond True --use_input_encoding False"
# DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine  --timestep_respacing 250"
# python scripts/st3d_room_layout_sample.py --data_dir /mnt/nas_3dv/hdd1/datasets/Structured3d/preprocessed/text2pano/test/livingroom/ \
#  --model_path log/ST3D_livingroom_textcondition_openai-2023-08-14-18-13-22-588311/ema_0.9999_200000.pt \
#  $MODEL_FLAGS \
#  $DIFFUSION_FLAGS \
#  --num_samples $NUM_SAMPLES  --log_dir $OUTPUT_FOLDER  --room_type 'livingroom'


# run panorama sampling
PANO_INPUT_FOLDER=$OUTPUT_FOLDER/livingroom
CKPT_PATH="/mnt/nas_3dv/hdd1/fangchuan/Layout_Controlnet/ckpts/control_v11p_sd15_seg_livingroom_fullres_40000.ckpt"
conda activate control-v11
cd /mnt/nas_3dv/hdd1/fangchuan/Layout_Controlnet/scripts
# python st3d_panorama_sample.py --input_folder $PANO_INPUT_FOLDER --ckpt_filepath $CKPT_PATH
# run super-resolution

# run panoramic reconstrcution
python st3d_panorama_recons.py --input_folder $PANO_INPUT_FOLDER


