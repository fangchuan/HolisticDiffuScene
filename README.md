# CtrlRoom

## Environment Setup and Batch Generation

### download dataset:
url: oss://lightillusions-text2room/CtrlRoom/20240219_text2pano.zip
unzip 20240219_text2pano.zip to /your/dataset/path

### download our codes:
* LayoutGenerationNetwork: https://github.com/fangchuan/HolisticDiffuScene/tree/develop
* PanoramaGenerationNetwork: https://github.com/fangchuan/Layout_ControlNet/tree/develop

### download pretrained weight/checkpoint: 
* layout generation ckpts: [gdrive](https://drive.google.com/drive/folders/1GzCFsp6aeowe-4iGp882D7pP3HZt3XXx)
* panorama generation ckpts: [gdrive](https://drive.google.com/drive/folders/1qr2T8UCU6SskajTJz9n1m9c4B6z8l9wR)

please find your favorite layoutdiffusion model(ST3D_xxxx) and panoramadiffusion model(control_v11p_sd15_seg_xxxx) and download to where you store your codes. 'ckpts.zip' stands for checkpoints for panformer(panorama depth estimator), please download and put them in to 'LayoutControlNet/panoformer/ckpts/mp3d_weights/'

### pull docker image: 

```docker run -it -d  --gpus all --name ctrlroom -v /your/codes/path/:/codes/ -v /your/dataset/path/:/dataset registry.cn-beijing.aliyuncs.com/ctrlroom/ctrl-room:v2``` 
* please replace '/your/codes/path/' and '/your/dataset/path/' with your own path;
### run docker image 

```docker run -it -d  --gpus all --name ctrlroom -v /your/codes/path/:/codes/ -v /your/dataset/path/:/dataset registry.cn-beijing.aliyuncs.com/ctrlroom/ctrl-room:v2``` 

### run batch generation script

#### text-to-bedroom generation:
please revise the 'RAW_DATASET_PATH', 'LAYOUT_MODEL_PATH', 'APP_MODEL_PATH' and 'train_stats_file' according to aforementioned downloaded data also revise carefully other paths consistent to your machine you can generate NUM_SAMPLES samples by run the bash script
```bash run_text2bedroom_pipeline.sh```

#### text-to-kitchen generation:
please revise the 'RAW_DATASET_PATH', 'LAYOUT_MODEL_PATH', 'APP_MODEL_PATH' and 'train_stats_file' according to aforementioned downloaded data also revise carefully other paths consistent to your machine you can generate NUM_SAMPLES samples by run the bash script
```bash run_text2kitchen_pipeline.sh```

#### text-to-bathroom generation:
please revise the 'RAW_DATASET_PATH', 'LAYOUT_MODEL_PATH', 'APP_MODEL_PATH' and 'train_stats_file' according to aforementioned downloaded data  also revise carefully other paths consistent to your machine you can generate NUM_SAMPLES samples by run the bash script
```bash run_text2livingroom_pipeline.sh```

## FAQ: possible problems:
* AttributeError: module 'PIL.Image' has no attribute 'LINEAR'
``` 
conda activate control-v11
pip install Pillow==9.5 
```
