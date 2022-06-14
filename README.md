## Introduction
style gan2 是style gan的改良版，主要功能可以透過少量的訓練資料來合成出特定的風格。
Style gan是由mapping network和Synthesis network組成，前者控制生成圖像的style，後者用於生成圖片。
mapping network要對latent space作解藕，去找出特徵間的關係，找到的隱藏特徵即為latent code，而latent code組成的空間就稱為latent space
style gan 最大的特似是可以做mixing style，把input的圖片找出彼此之間的feature，會以表格方式去套用兩者的feature。

## Literature Review
GAN: 同時訓練discriminator與generator，generator要努力產出假的圖片不被discriminator發現，discriminator則是要努力去判斷圖片的真假。
Style gan1: 能達到的task與Introduction介紹的類似，但是在人像的細節處(如頭髮)會有些水珠、鬼影等現象。

## Dataset
StyleGAN trained with Flickr-Faces-HQ dataset at 1024×1024.
StyleGAN trained with CelebA-HQ dataset at 1024×1024.
StyleGAN trained with LSUN Bedroom dataset at 256×256.

## Baseline
StyleGAN

## Main Approach
在這次任務中，使用pretrained model + 8張網路找的人向圖片再去尋找latent code，latent variable z經過mapping network變為w，z是均勻分布的隨機向量，要變為w是因為在feature中不是均勻分布。
style mixing則是透過把不同的latent code z1和z2輸入mapping network去得到w1和w2代表兩種不同的style，再透過synthesis network去取得交叉點，交叉之前用w1，交叉之後使用w2。如此生成的圖片就去友兩者source的特徵。

## Metric
在這個部分論文有使用FID, Path length, Precision, Recall在FFHQ資料庫、LSUN Car資料庫做評比，但因為我僅使用並且重train style的部分，所以這個部分比較缺乏。

## Results & Analysis
可以從PPT中看到，在8*8的表格中都具有其各自的特色。
另外，有嘗試進行執行動漫腳色與真人的合成，效果並不是很理想，或許可以加入future work的部分。

## Requirements

* TensorFlow 1.14 or 1.15 with GPU support. 
* CUDA 10.0 toolkit and cuDNN 7.5.
## 




Pre-trained networks are stored as `*.pkl` files on the [StyleGAN2 Google Drive folder](https://drive.google.com/open?id=1QHc-yF5C3DChRwSdZKcx1w6K8JvSxQi7). Below, you can either reference them directly using the syntax `gdrive:networks/<filename>.pkl`, or download them manually and reference by filename.

**Generating images**:

```.bash
# Generate uncurated ffhq images (matches paper Figure 12)
python run_generator.py generate-images --network=gdrive:networks/stylegan2-ffhq-config-f.pkl \
  --seeds=6600-6625 --truncation-psi=0.5

# Generate curated ffhq images (matches paper Figure 11)
python run_generator.py generate-images --network=gdrive:networks/stylegan2-ffhq-config-f.pkl \
  --seeds=66,230,389,1518 --truncation-psi=1.0

# Generate uncurated car images
python run_generator.py generate-images --network=gdrive:networks/stylegan2-car-config-f.pkl \
  --seeds=6000-6025 --truncation-psi=0.5

# Example of style mixing (matches the corresponding video clip)
python run_generator.py style-mixing-example --network=gdrive:networks/stylegan2-ffhq-config-f.pkl \
  --row-seeds=85,100,75,458,1500 --col-seeds=55,821,1789,293 --truncation-psi=1.0
```

The results are placed in `results/<RUNNING_ID>/*.png`. You can change the location with `--result-dir`. For example, `--result-dir=~/my-stylegan2-results`.

**Projecting images to latent space**:

```.bash
# Project generated images
python run_projector.py project-generated-images --network=gdrive:networks/stylegan2-car-config-f.pkl \
  --seeds=0,1,5

# Project real images
python run_projector.py project-real-images --network=gdrive:networks/stylegan2-car-config-f.pkl \
  --dataset=car --data-dir=~/datasets
```

You can import the networks in your own Python code using `pickle.load()`. For this to work, you need to include the `dnnlib` source directory in `PYTHONPATH` and create a default TensorFlow session by calling `dnnlib.tflib.init_tf()`. See [run_generator.py](./run_generator.py) and [pretrained_networks.py](./pretrained_networks.py) for examples.

## Training networks

To reproduce the training runs for config F in Tables 1 and 3, run:

```.bash
python run_training.py --num-gpus=8 --data-dir=~/datasets --config=config-f \
  --dataset=ffhq --mirror-augment=true
python run_training.py --num-gpus=8 --data-dir=~/datasets --config=config-f \
  --dataset=car --total-kimg=57000
python run_training.py --num-gpus=8 --data-dir=~/datasets --config=config-f \
  --dataset=cat --total-kimg=88000
python run_training.py --num-gpus=8 --data-dir=~/datasets --config=config-f \
  --dataset=church --total-kimg 88000 --gamma=100
python run_training.py --num-gpus=8 --data-dir=~/datasets --config=config-f \
  --dataset=horse --total-kimg 100000 --gamma=100
```

For other configurations, see `python run_training.py --help`.

We have verified that the results match the paper when training with 1, 2, 4, or 8 GPUs. Note that training FFHQ at 1024&times;1024 resolution requires GPU(s) with at least 16 GB of memory. The following table lists typical training times using NVIDIA DGX-1 with 8 Tesla V100 GPUs:

| Configuration | Resolution      | Total kimg | 1 GPU   | 2 GPUs  | 4 GPUs  | 8 GPUs | GPU mem |
| :------------ | :-------------: | :--------: | :-----: | :-----: | :-----: | :----: | :-----: |
| `config-f`    | 1024&times;1024 | 25000      | 69d 23h | 36d 4h  | 18d 14h | 9d 18h | 13.3 GB |
| `config-f`    | 1024&times;1024 | 10000      | 27d 23h | 14d 11h | 7d 10h  | 3d 22h | 13.3 GB |
| `config-e`    | 1024&times;1024 | 25000      | 35d 11h | 18d 15h | 9d 15h  | 5d 6h  | 8.6 GB  |
| `config-e`    | 1024&times;1024 | 10000      | 14d 4h  | 7d 11h  | 3d 20h  | 2d 3h  | 8.6 GB  |
| `config-f`    | 256&times;256   | 25000      | 32d 13h | 16d 23h | 8d 21h  | 4d 18h | 6.4 GB  |
| `config-f`    | 256&times;256   | 10000      | 13d 0h  | 6d 19h  | 3d 13h  | 1d 22h | 6.4 GB  |

Training curves for FFHQ config F (StyleGAN2) compared to original StyleGAN using 8 GPUs:

![Training curves](./docs/stylegan2-training-curves.png)

After training, the resulting networks can be used the same way as the official pre-trained networks:

```.bash
# Generate 1000 random images without truncation
python run_generator.py generate-images --seeds=0-999 --truncation-psi=1.0 \
  --network=results/00006-stylegan2-ffhq-8gpu-config-f/networks-final.pkl
```

## Evaluation metrics

To reproduce the numbers for config F in Tables 1 and 3, run:

```.bash
python run_metrics.py --data-dir=~/datasets --network=gdrive:networks/stylegan2-ffhq-config-f.pkl \
  --metrics=fid50k,ppl_wend --dataset=ffhq --mirror-augment=true
python run_metrics.py --data-dir=~/datasets --network=gdrive:networks/stylegan2-car-config-f.pkl \
  --metrics=fid50k,ppl2_wend --dataset=car
python run_metrics.py --data-dir=~/datasets --network=gdrive:networks/stylegan2-cat-config-f.pkl \
  --metrics=fid50k,ppl2_wend --dataset=cat
python run_metrics.py --data-dir=~/datasets --network=gdrive:networks/stylegan2-church-config-f.pkl \
  --metrics=fid50k,ppl2_wend --dataset=church
python run_metrics.py --data-dir=~/datasets --network=gdrive:networks/stylegan2-horse-config-f.pkl \
  --metrics=fid50k,ppl2_wend --dataset=horse
```

For other configurations, see the [StyleGAN2 Google Drive folder](https://drive.google.com/open?id=1QHc-yF5C3DChRwSdZKcx1w6K8JvSxQi7).

Note that the metrics are evaluated using a different random seed each time, so the results will vary between runs. In the paper, we reported the average result of running each metric 10 times. The following table lists the available metrics along with their expected runtimes and random variation:

| Metric      | FFHQ config F  | 1 GPU  | 2 GPUs  | 4 GPUs | Description |
| :---------- | :------------: | :----: | :-----: | :----: | :---------- |
| `fid50k`    | 2.84 &pm; 0.03 | 22 min | 14 min  | 10 min | [Fr&eacute;chet Inception Distance](https://arxiv.org/abs/1706.08500)
| `is50k`     | 5.13 &pm; 0.02 | 23 min | 14 min  | 8 min  | [Inception Score](https://arxiv.org/abs/1606.03498)
| `ppl_zfull` | 348.0 &pm; 3.8 | 41 min | 22 min  | 14 min | [Perceptual Path Length](https://arxiv.org/abs/1812.04948) in Z, full paths
| `ppl_wfull` | 126.9 &pm; 0.2 | 42 min | 22 min  | 13 min | [Perceptual Path Length](https://arxiv.org/abs/1812.04948) in W, full paths
| `ppl_zend`  | 348.6 &pm; 3.0 | 41 min | 22 min  | 14 min | [Perceptual Path Length](https://arxiv.org/abs/1812.04948) in Z, path endpoints
| `ppl_wend`  | 129.4 &pm; 0.8 | 40 min | 23 min  | 13 min | [Perceptual Path Length](https://arxiv.org/abs/1812.04948) in W, path endpoints
| `ppl2_wend` | 145.0 &pm; 0.5 | 41 min | 23 min  | 14 min | [Perceptual Path Length](https://arxiv.org/abs/1812.04948) without center crop
| `ls`        | 154.2 / 4.27   | 10 hrs | 6 hrs   | 4 hrs  | [Linear Separability](https://arxiv.org/abs/1812.04948)
| `pr50k3`    | 0.689 / 0.492  | 26 min | 17 min  | 12 min | [Precision and Recall](https://arxiv.org/abs/1904.06991)

Note that some of the metrics cache dataset-specific data on the disk, and they will take somewhat longer when run for the first time.

## License

Copyright &copy; 2019, NVIDIA Corporation. All rights reserved.

This work is made available under the Nvidia Source Code License-NC. To view a copy of this license, visit https://nvlabs.github.io/stylegan2/license.html

## Citation

```
@article{Karras2019stylegan2,
  title   = {Analyzing and Improving the Image Quality of {StyleGAN}},
  author  = {Tero Karras and Samuli Laine and Miika Aittala and Janne Hellsten and Jaakko Lehtinen and Timo Aila},
  journal = {CoRR},
  volume  = {abs/1912.04958},
  year    = {2019},
}
```

## Acknowledgements

We thank Ming-Yu Liu for an early review, Timo Viitanen for his help with code release, and Tero Kuosmanen for compute infrastructure.
