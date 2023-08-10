MODEL_FLAGS="--layout_channels 34 --layout_size 48 --num_channels 128 --num_res_blocks 2 --b_learn_sigma True  --b_class_cond False --b_text_cond True --use_input_encoding False"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine  --timestep_respacing 250"


python scripts/st3d_room_layout_sample.py --data_dir /mnt/nas_3dv/hdd1/datasets/Structured3d/preprocessed/debug_livingroom/train/livingroom/ --model_path log/openai-2023-08-09-10-22-59-590824/ema_0.9999_100000.pt $MODEL_FLAGS $DIFFUSION_FLAGS