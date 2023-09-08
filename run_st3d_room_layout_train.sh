# bedroom parametrization
MODEL_FLAGS="--layout_channels 32 --layout_size 23 --num_channels 128 --num_res_blocks 3 --b_learn_sigma True  --b_class_cond False --b_text_cond True --use_input_encoding False"

# living room parametrization
# MODEL_FLAGS="--layout_channels 34 --layout_size 48 --num_channels 128 --num_res_blocks 3 --b_learn_sigma True --b_class_cond False --b_text_cond True --use_input_encoding False"


DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine"
TRAIN_FLAGS="--lr 1e-4 --batch_size 64 --schedule_sampler loss-second-moment --use_3d_iou False "
NUM_GPUS=2

mpiexec -n $NUM_GPUS python scripts/st3d_room_layout_train.py --data_dir /mnt/nas_3dv/hdd1/datasets/Structured3d/preprocessed/text2pano/train/bedroom/ $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS

