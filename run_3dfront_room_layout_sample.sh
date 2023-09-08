MODEL_FLAGS="--layout_channels 33 --layout_size 23 --num_channels 128 --num_res_blocks 3 --b_learn_sigma True --b_class_cond True --b_text_cond False --use_input_encoding False"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine  --timestep_respacing 250"


python scripts/threed_front_room_layout_sample.py --data_dir /mnt/nas_3dv/hdd1/datasets/3D_FRONT_FUTURE/bedroom/ --model_path log/TDFRONT_bedroom_uncondition_openai-2023-07-26-16-17-12-692249/ema_0.9999_200000.pt $MODEL_FLAGS $DIFFUSION_FLAGS