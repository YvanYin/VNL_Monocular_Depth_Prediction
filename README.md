#### Enforcing geometric constraints of virtual normal for depth prediction.

This repository contains the source code of our paper:
[Yin Wei, Yifan Liu, Chunhua Shen, Youliang Yan, Enforcing geometric constraints of virtual normal for depth prediction](https://arxiv.org/abs/1907.12209) (accepted for publication in ICCV' 2019).

## Some Results

![NYU_Depth](./examples/nyu_gif.gif)
![kitti_Depth](./examples/kitti_gif.gif)
![SurfaceNormal](./examples/surface_normal.jpg)


## Framework
![SurfaceNormal](./examples/framework.jpg)

## Hightlights
- **State-of-the-art performance:** The AbsRel errs are 10.8% and 7.2% on NYU and KITTI. More details please refer to our full paper. The absrel err of the released model on NYU can reach 10.5%. 

****
## Installation
- Please refer to [Installation](./Installation.md).

## Datasets
- NYUDV2
   The details of datasets can be found [here](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html). The Eigen split of labeled images can be downloaded [here](https://cloudstor.aarnet.edu.au/plus/s/G2ckXCJX3pvrzRU). Please extract it to ./datasets. Our SOTA model is trained on the around 20K unlabled images.
    
- KITTI
  The details of KITTI benchmark for monocular depth prediction is [here](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction). We use both the official split and Eigen split. You can find the filenames [here](https://github.com/mrharicot/monodepth/tree/master/utils/filenames).
  
  
## Model Zoo
- ResNext101_32x4d backbone, trained on NYU dataset, download [here](https://cloudstor.aarnet.edu.au/plus/s/7kdsKYchLdTi53p)
- ResNext101_32x4d backbone, trained on KITTI dataset (Eigen split), download [here](https://cloudstor.aarnet.edu.au/plus/s/eviO16z68cKbip5)
- ResNext101_32x4d backbone, trained on KITTI dataset (Official split), download [here](https://cloudstor.aarnet.edu.au/plus/s/pqIxORtFrVOFoea)



  
## Inference

```bash
# Run the inferece on NYUDV2 dataset
 python  ./tools/test_nyu_metric.py \
		--dataroot    ./datasets/NYUDV2 \
		--dataset     nyudv2 \
		--cfg_file     lib/configs/resnext101_32x4d_nyudv2_class \
		--load_ckpt   ./nyu_rawdata.pth 
```
If you want to test the kitti dataset, please see [here](./datasets/KITTI/README.md)


### Citation
```
@article{wei2019enforcing,
  title={Enforcing geometric constraints of virtual normal for depth prediction},
  author={Wei, Yin and Liu, Yifan and Shen, Chunhua and Yan, Youliang},
  journal={arXiv preprint arXiv:1907.12209},
  year={2019}
}
```
### Contact
Wei Yin: wei.yin@adelaide.edu.au

