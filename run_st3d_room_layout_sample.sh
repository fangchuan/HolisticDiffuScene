# bedroom parametrization
# MODEL_FLAGS="--layout_channels 32 --layout_size 23 --num_channels 128 --num_res_blocks 3 --b_learn_sigma True  --b_class_cond False --b_text_cond True --use_input_encoding False"

# living room parametrization
MODEL_FLAGS="--layout_channels 34 --layout_size 48 --num_channels 128 --num_res_blocks 3 --b_learn_sigma True  --b_class_cond False --b_text_cond True --use_input_encoding False"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine  --timestep_respacing 250"


python scripts/st3d_room_layout_sample.py --data_dir /mnt/nas_3dv/hdd1/datasets/datasets/Structured3D/text2pano/test/livingroom/ \
 --model_path log/ST3D_livingroom_textcondition_openai-2023-08-14-18-13-22-588311/ema_0.9999_200000.pt \
 $MODEL_FLAGS \
 $DIFFUSION_FLAGS \
 --room_type 'livingroom'

# bedroom --model_path log/openai-2023-09-08-15-04-50-375770/ema_0.9999_180000.pt \
# livingroom --model_path log/ST3D_livingroom_textcondition_openai-2023-08-14-18-13-22-588311/ema_0.9999_200000.pt \
