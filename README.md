
# Fully Sparse Transformer 3D Detector for LiDAR Point Cloud 

[Paper](https://ieeexplore.ieee.org/document/10302363), [LeaderBoard](https://www.nuscenes.org/object-detection?externalData=all&mapData=all&modalities=Lidar)

<!-- ## Introduction -->

All statistics are measured on a single Tesla A100 GPU using the best model of official repositories. Some sparse module in the model are supported. </em>
</div><br/>

FSTR is a fully sparse LiDAR-based detector that achieves better accuracy-efficient trade-off compare with other popular LiDAR-based detectors. A lightweight DETR-like framework with signle decoder layer is designed for lidar-only detection, which obtains **73.6%**(**FSTR-XL with TTA**) NDS on nuScenes benchmark and **31.5%** CDS on Argoverse2 validation dataset,

## Preparation

* Environments  
Python == 3.8 \
CUDA == 11.1 \
pytorch == 1.9.0 \
mmcv-full == 1.6.0 \
mmdet == 2.24.0 \
mmsegmentation == 0.29.1 \
mmdet3d == 1.0.0rc5 \
[flash-attn](https://github.com/HazyResearch/flash-attention) == 0.2.2
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
If you have any questions, feel free to open an issue or contact us at zhangdiankun19@mails.ucas.edu.com, or tanfeiyang@megvii.com.
