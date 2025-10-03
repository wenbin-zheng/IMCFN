# IMCFN
This is the source code for the paper "Image Manipulation Fusion Network for Multimodal Fake News Detection."
![IMCFN Framework](https://github.com/wenbin-zheng/IMCFN/blob/main/IMCFN.jpg)

## Download data
 `Weibo` dataset is available at [https://github.com/yaqingwang/EANN-KDD18](https://github.com/yaqingwang/EANN-KDD18/)

 `GossipCop` dataset can be access the following link: [https://github.com/shiivangii/SpotFakePlus]( https://github.com/shiivangii/SpotFakePlus/)

The downloaded datasets need to be moved into the `./Data` folder.

## Data pre-processing

Use `dataprocess_weibo.py` to pre-process the `Weibo` dataset.

Use `datprocess_gossipcop.py` to pre-process the `GossipCop` dataset.

If you want to change dataset for training, you should revise
```python
import utils.dataprocess_weibo as processed_data
```
```python
--dataset default='weibo'
```
## Setup

### Dependencies

1. [Python = 3.10](https://github.com/dmlc/dgl/)
2. [torch = 2.3.1](https://pytorch.org/get-started/locally/)
3. [transformers = 4.6.0](https://huggingface.co/docs/transformers/installation)


### Run the code

run ```train.py ```

## Reference
Thanks for their great work
* [MINER-UVS](https://github.com/wangbing1416/MINER-UVS)
* [WWW 2021](https://github.com/RMSnow/WWW2021)
