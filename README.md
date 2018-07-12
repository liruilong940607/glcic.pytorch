# Image-Completion-Pytorch

This is a implement of the paper 'Globally and Locally Consistent Image Completion'[[Project Page]](http://hi.cs.waseda.ac.jp/~iizuka/projects/completion/en/)

## Usage

For ATR Dataset:
```
CUDA_VISIBLE_DEVICES=0 python train --config=config.yaml
```

For LIP Dataset:
```
CUDA_VISIBLE_DEVICES=1 python train --config=config_lip.yaml
```

