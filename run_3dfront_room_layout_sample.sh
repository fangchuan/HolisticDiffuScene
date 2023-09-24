# bedroom parametrization
# MODEL_FLAGS="--layout_channels 33 --layout_size 23 --num_channels 128 --num_res_blocks 3 --b_learn_sigma True --b_class_cond True --b_text_cond False --use_input_encoding False"

# living room parametrization
MODEL_FLAGS="--layout_channels 36 --layout_size 41 --num_channels 128 --num_res_blocks 3 --b_learn_sigma True --b_class_cond False --b_text_cond True --use_input_encoding False"

DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine  --timestep_respacing 250"


python scripts/threed_front_room_layout_sample.py --data_dir /mnt/nas_3dv/hdd1/datasets/3D_FRONT_FUTURE/full/test/livingroom/ \
    --model_path log/openai-2023-09-23-20-13-33-964075/ema_0.9999_100000.pt \
     $MODEL_FLAGS \
     $DIFFUSION_FLAGS \
     --num_samples 1000 \
      --room_type 'livingroom' \
      --path_to_pickled_3d_futute_models /mnt/nas_3dv/hdd1/datasets/3D_FRONT_FUTURE/threed_future_model_livingroom.pkl