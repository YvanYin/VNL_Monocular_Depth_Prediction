### Test on KITTI

- Step 1
  Download data from [here](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction).

- Step 2
  Create a json file named test_annotations.json. Put it under ./dataset/KITTI/annotations.
  The annotations are like follows.
  ```
  [
   {'depth_path': 'test/<image_name>.png', 'rgb_path': 'test/<image_name>.png' }
   ...
  ]
```
