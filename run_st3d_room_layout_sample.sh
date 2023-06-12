MODEL_FLAGS="--layout_size 1024 --num_channels 128 --num_res_blocks 3 --b_learn_sigma True --b_class_cond True"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine --b_use_kl True  --timestep_respacing 250"


python scripts/st3d_room_layout_sample.py --model_path log/ema_0.9999_090000.pt $MODEL_FLAGS $DIFFUSION_FLAGS