MODEL_FLAGS="--layout_size 1024 --num_channels 128 --num_res_blocks 3 --b_learn_sigma True --b_class_cond True"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine --b_use_kl True"
TRAIN_FLAGS="--lr 1e-4 --batch_size 64 --schedule_sampler loss-second-moment"

python scripts/st3d_room_layout_train.py --data_dir /data/dataset/Structured3D/preprocessed/st3d_train_full_raw_light/ $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS

# mpiexec -n $NUM_GPUS python scripts/image_train.py --data_dir path/to/images $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS