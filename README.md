# IMCFN
This is the source code for the paper "Image Manipulation Fusion Network for Multimodal Fake News Detection."
![IMCFN Framework](https://github.com/wenbin-zheng/IMCFN/blob/main/IMCFN.jpg)

## Download data
If you want to download the `Weibo` dataset, you can access the following link: [https://github.com/yaqingwang/EANN-KDD18](https://github.com/yaqingwang/EANN-KDD18/)

If you want to download the `GossipCop` dataset, you can access the following link: [https://github.com/shiivangii/SpotFakePlus]( https://github.com/shiivangii/SpotFakePlus/)

Then, you should put them into `./Data`

## Data pre-processing

Use `data_preprocess_weibo.py` to pre-process the `Weibo` dataset.

Use `data_preprocess_gossipcop.py` to pre-process the `GossipCop` dataset.

If you want to change dataset for training, you should revise
```python
import utils.data_preprocess_weibo as data_preprocess
```
```python
--dataset default='weibo'
```
## Setup

### Dependencies

1. [Python = 3.10](https://github.com/dmlc/dgl/)
2. [torch = 2.3.1](https://pytorch.org/get-started/locally/)
4. [transformers = 4.6.0](https://huggingface.co/docs/transformers/installation)


### Run the code

run ```main.py ```

## Reference
Thanks for their great work
* [MINER-UVS](https://github.com/wangbing1416/MINER-UVS)
