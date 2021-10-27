# Dynamic-Vision-Transformer (NeurIPS 2021)

This repo contains the official **MindSpore** code for the Dynamic Vision Transformer (DVT).

- [Not All Images are Worth 16x16 Words: Dynamic Transformers for Efficient Image Recognition](https://arxiv.org/pdf/2105.15075.pdf)

## Introduction

<p align="center">
    <img src="https://github.com/blackfeather-wang/Dynamic-Vision-Transformer/blob/main/figures/examples.png" width= "400">
</p>

We develop a Dynamic Vision Transformer (DVT) to automatically configure a proper number of tokens for each individual image, leading to a significant improvement in computational efficiency,  both theoretically and empirically.
<p align="center">
    <img src="https://github.com/blackfeather-wang/Dynamic-Vision-Transformer/blob/main/figures/overview.png" width= "810">
</p>

## Training
You have to execute script from "src" directory. It will create directory "../results/{DATETIME}__{EXPERIMENT_NAME}" and place results there.

```
bash scripts/train_ascend.sh {0-7} EXPERIMENT_NAME --config=CONFIG_PATH --device {Ascend (default)|GPU} [TRAIN.PY_ARGUMENTS]

# training for feature reuse and releation reuse
bash scripts/train_ascend.sh 0-7 deit_dvt_12_49_196_w_f_w_r_adamw_originhead_dataaug_mixup --config=configs/local/vit_dvt/deit_dvt_12_49_196_w_f_w_r_adamw_originhead_dataaug_mixup.yml.j2

# training for feature reuse and w/o releation reuse
bash scripts/train_ascend.sh 0-7 deit_dvt_12_49_196_w_f_n_r_adamw_originhead_dataaug_mixup --config=configs/local/vit_dvt/deit_dvt_12_49_196_w_f_n_r_adamw_originhead_dataaug_mixup.yml.j2

# training for w/o feature reuse and releation reuse
bash scripts/train_ascend.sh 0-7 deit_dvt_12_49_196_n_f_w_r_adamw_originhead_dataaug_mixup --config=configs/local/vit_dvt/deit_dvt_12_49_196_n_f_w_r_adamw_originhead_dataaug_mixup.yml.j2

# training for w/o feature reuse and w/o releation reuse
bash scripts/train_ascend.sh 0-7 deit_dvt_12_49_196_n_f_n_r_adamw_originhead_dataaug_mixup --config=configs/local/vit_dvt/deit_dvt_12_49_196_n_f_n_r_adamw_originhead_dataaug_mixup.yml.j2

# inference for feature reuse and releation reuse
bash scripts/inference_ascend.sh 0 deit_dvt_12_49_196_w_f_w_r_adamw_originhead_dataaug_mixup_inference --config=configs/local/vit_dvt/deit_dvt_12_49_196_w_f_w_r_adamw_originhead_dataaug_mixup_inference.yml.j2

```

## Results

- Models Overview

|model|flops|acc|
|-|-|-|
|deit-s/16|4.608|78.67|
|deit-s/32|1.145|72.116|
|vit-b/16|17.58|79.1|
|vit-b/32|4.41|73.972|

- Top-1 accuracy on ImageNet v.s. GFLOPs

![]()

<p align="center">
    <img src="deit_dvt_vs_vit_inference.png" width= "500">
</p>

- Visualization
<p align="center">
    <img src="https://github.com/blackfeather-wang/Dynamic-Vision-Transformer/raw/main/figures/result_visual.png" width= "700">
</p>


## Requirements

* Mindspore 1.5 (https://www.mindspore.cn/install/en)
* jinja2 (https://anaconda.org/anaconda/jinja2)
* tqdm (for GPU only)
* mpi4py (for GPU only)


## Citation

If you find this work valuable or use our code in your own research, please consider citing us with the following bibtex:

```
@inproceedings{wang2021not,
        title = {Not All Images are Worth 16x16 Words: Dynamic Transformers for Efficient Image Recognition},
       author = {Wang, Yulin and Huang, Rui and Song, Shiji and Huang, Zeyi and Huang, Gao},
    booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
         year = {2021}
}
```

## Contact
This is a MindSpore implementation version. If you have any question, please feel free to contact Yulin Wang: wang-yl19@mails.tsinghua.edu.cn and [Guanfu Chen](https://github.com/guanfuchen): guanfuchen@zju.edu.cn.
