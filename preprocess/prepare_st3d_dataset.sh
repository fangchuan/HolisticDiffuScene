python prepare_st3d_dataset.py \
 --dataset_path /data/dataset/Structured3D/Structured3D/ \
 --room_type st3d_livingroom \
 --annotated_labels_path /data/dataset/Structured3D/preprocessed/annotations/livingroom/latest_labels/ \
 --out_train_path /mnt/nas_3dv/hdd1/datasets/Structured3d/preprocessed/20240219_text2pano/train \
 --out_test_path /mnt/nas_3dv/hdd1/datasets/Structured3d/preprocessed/20240219_text2pano/test
python prepare_st3d_dataset.py \
 --dataset_path /data/dataset/Structured3D/Structured3D/ \
 --room_type st3d_bedroom \
 --annotated_labels_path /data/dataset/Structured3D/preprocessed/annotations/bedroom/latest_labels/ \
 --out_train_path /mnt/nas_3dv/hdd1/datasets/Structured3d/preprocessed/20240219_text2pano/train \
 --out_test_path /mnt/nas_3dv/hdd1/datasets/Structured3d/preprocessed/20240219_text2pano/test
python prepare_st3d_dataset.py \
 --dataset_path /data/dataset/Structured3D/Structured3D/ \
 --room_type st3d_kitchen \
 --annotated_labels_path /data/dataset/Structured3D/preprocessed/annotations/kitchen/latest_labels/ \
 --out_train_path /mnt/nas_3dv/hdd1/datasets/Structured3d/preprocessed/20240219_text2pano/train \
 --out_test_path /mnt/nas_3dv/hdd1/datasets/Structured3d/preprocessed/20240219_text2pano/test

# zip the dataset
cd /mnt/nas_3dv/hdd1/datasets/Structured3d/preprocessed/
zip -r 20240219_text2pano.zip 20240219_text2pano/

# ossutil upload
ossutil cp /mnt/nas_3dv/hdd1/datasets/Structured3d/preprocessed/20240219_text2pano.zip oss://lightillusions-text2room/20240219_text2pano.zip