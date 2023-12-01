
# Fully Sparse Transformer 3D Detector for LiDAR Point Cloud 

[Paper](https://ieeexplore.ieee.org/document/10302363), [nuScenes LeaderBoard](https://www.nuscenes.org/object-detection?externalData=all&mapData=all&modalities=Lidar)

<!-- ## Introduction -->

All statistics are measured on a single Tesla A100 GPU using the best model of official repositories. Some sparse module in the model are supported. </em>
</div><br/>

FSTR is a fully sparse LiDAR-based detector that achieves better accuracy-efficient trade-off compare with other popular LiDAR-based detectors. A lightweight DETR-like framework with signle decoder layer is designed for lidar-only detection, which obtains **73.6%** NDS (**FSTR-XLarge with TTA**) on nuScenes benchmark and **31.5%** CDS (**FSTR-Large**) on Argoverse2 validation dataset.

## Currently Supported Features
- [x] Support nuScenes dataset
- [ ] Support Argoverse2 dataset
## Preparation

* Environments  
Python == 3.8 \
CUDA == 11.1 \
pytorch == 1.9.0 \
mmcv-full == 1.6.0 \
mmdet == 2.24.0 \
mmsegmentation == 0.29.1 \
mmdet3d == 1.0.0rc5 \
[flash-attn](https://github.com/HazyResearch/flash-attention) == 0.2.2 \
[Spconv-plus](https://github.com/dvlab-research/spconv-plus) == 2.1.21

* Data   
Follow the [mmdet3d](https://github.com/open-mmlab/mmdetection3d/blob/master/docs/en/data_preparation.md) to process the nuScenes dataset.

## Train & inference
```bash
# train
bash tools/dist_train.sh /path_to_your_config 8
# inference
bash tools/dist_test.sh /path_to_your_config /path_to_your_pth 8 --eval bbox
```
## Main Results
Results on nuScenes **val set**. The default batch size is 2 on each GPU. The FPS are all evaluated with a single Tesla A100 GPU. (15e + 5e means the last 5 epochs should be trained without [GTsample](https://github.com/Poley97/FSTR/blob/master/projects/configs/lidar/fstr_voxel0075_cbgs_20e.py.py#L33-L69))

| Config            | mAP      | NDS     | Schedule|Inference FPS|
|:--------:|:----------:|:---------:|:--------:|:--------:|
| [FSTR](./projects/configs/lidar/fstr_voxel0075_cbgs_20e.py) | 64.2% | 69.1%  | 15e+5e | 15.4 |
| [FSTR-Large](./projects/configs/lidar/fstr_large_voxel0075_cbgs_20e.py) | 65.5% | 70.3%    | 15e+5e | 9.5 |  


Results on nuScenes **test set**. To reproduce our result, replace `ann_file=data_root + '/nuscenes_infos_train.pkl'` in [training config](./projects/configs/lidar/fstr_large_voxel0075_cbgs_20e.py) with `ann_file=[data_root + '/nuscenes_infos_train.pkl', data_root + '/nuscenes_infos_val.pkl']`:

| Config            |Modality| mAP      | NDS     | Schedule|Inference FPS|
|:--------:|:----------:|:---------:|:--------:|:--------:|:--------:|
| [FSTR](./projects/configs/lidar/fstr_voxel0075_cbgs_20e.py) | 66.2% | 70.4%  | 15e+5e | 15.4 |
| [FSTR](./projects/configs/lidar/fstr_voxel0075_cbgs_20e.py) +TTA | 67.6% | 71.5%  | 15e+5e | - |
| [FSTR-Large](./projects/configs/lidar/fstr_large_voxel0075_cbgs_20e.py) + TTA | 69.5% | 73.0%  | 15e+5e | - |
| [FSTR-XLarge](./projects/configs/lidar/fstr_xlarge_voxel0050_cbgs_20e.py) + TTA | 70.2% | 73.5%  | 15e+5e | - |

Note that [FSTR-XLarge](./projects/configs/lidar/fstr_xlarge_voxel0050_cbgs_20e.py) are trained on a 8 Tesla A100 GPUs.
## Citation
If you find our FSTR helpful in your research, please consider citing: 
```bibtex   
@article{zhang2023fully,
  title={Fully Sparse Transformer 3D Detector for LiDAR Point Cloud},
  author={Zhang, Diankun and Zheng, Zhijie and Niu, Haoyu and Wang, Xueqing and Liu, Xiaojun},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2023},
  publisher={IEEE}
}
```

## Contact
If you have any questions, feel free to open an issue or contact us at zhangdiankun19@mails.ucas.edu.cn, or tanfeiyang@megvii.com.

## Acknowledgement
Parts of our Code refer to the the recent work [CMT](https://github.com/junjie18/CMT).
