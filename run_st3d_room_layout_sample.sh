MODEL_FLAGS="--layout_size 1024 --num_channels 128 --num_res_blocks 3 --b_learn_sigma True --b_class_cond True"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine  --timestep_respacing 250"


python scripts/st3d_room_layout_sample.py --model_path log/openai-2023-06-30-20-13-26-941391/ema_0.9999_200000.pt $MODEL_FLAGS $DIFFUSION_FLAGS