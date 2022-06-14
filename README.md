## Introduction
style gan2 是style gan的改良版，主要功能可以透過少量的訓練資料來合成出特定的風格。
Style gan是由mapping network和Synthesis network組成，前者控制生成圖像的style，後者用於生成圖片。
mapping network要對latent space作解藕，去找出特徵間的關係，找到的隱藏特徵即為latent code，而latent code組成的空間就稱為latent space
style gan 最大的特似是可以做mixing style，把input的圖片找出彼此之間的feature，會以表格方式去套用兩者的feature。

## Literature Review
GAN: 同時訓練discriminator與generator，generator要努力產出假的圖片不被discriminator發現，discriminator則是要努力去判斷圖片的真假。
Style gan1: 能達到的task與Introduction介紹的類似，但是在人像的細節處(如頭髮)會有些水珠、鬼影等現象。
![image](https://raw.githubusercontent.com/fallshoes00/stylegan2_workshop-master/main/%E6%B0%B4%E7%8F%A0.PNG)
## Dataset
StyleGAN trained with Flickr-Faces-HQ dataset at 1024×1024.
StyleGAN trained with CelebA-HQ dataset at 1024×1024.
StyleGAN trained with LSUN Bedroom dataset at 256×256.

## Baseline
StyleGAN1:
![image](https://github.com/NVlabs/stylegan/raw/master/stylegan-teaser.png)

## Main Approach
在這次任務中，使用pretrained model + 8張網路找的人向圖片再去尋找latent code，latent variable z經過mapping network變為w，z是均勻分布的隨機向量，要變為w是因為在feature中不是均勻分布。
style mixing則是透過把不同的latent code z1和z2輸入mapping network去得到w1和w2代表兩種不同的style，再透過synthesis network去取得交叉點，交叉之前用w1，交叉之後使用w2。如此生成的圖片就去友兩者source的特徵。

## Metric
在這個部分論文有使用FID, Path length, Precision, Recall在FFHQ資料庫、LSUN Car資料庫做評比，但因為我僅使用並且重train style的部分，所以這個部分比較缺乏。

## Results & Analysis
可以從PPT中看到，在8*8的表格中都具有其各自的特色。
另外，有嘗試進行執行動漫腳色與真人的合成，效果並不是很理想，或許可以加入future work的部分。

## Error Analysis
優點:搞出來的東西很有趣
缺點:
專案老舊，使用tensorflow1.14，僅支援CUDA10，而最新的RTX30系列顯卡為11.0以上，故使用colab來操作
要訓練一段時間，而且在colab的平台上很容易斷線就要整個重新來過。
人臉看起來也有許多不自然的部分(如右圖)。
![image](https://github.com/fallshoes00/stylegan2_workshop-master/blob/main/%E6%80%AA%E6%80%AA%E7%9A%84.PNG?raw=true)

這是其他真人圖片實驗的結果:
![image](https://github.com/fallshoes00/stylegan2_workshop/blob/main/%E4%BA%BA%E5%83%8F88.png?raw=true)

而在使用非真實人臉(動漫)時，結果也很不理想。
![image](https://github.com/fallshoes00/stylegan2_workshop/blob/main/%E5%8B%95%E7%95%AB88.png?raw=true)


## Code and requirements
Code: [stylegan2encoder.ipynb]
* Python3.6
* TensorFlow 1.14 or 1.15 with GPU support. 
* CUDA 10.0 toolkit and cuDNN 7.5.
* All parameters are set by default.
## Contribution of each member
這個組只剩下我自己，其他人都期中就沒有回應了，後來助教的comment說題目太大可能太難，在實際動手後就發現真的目標過大，無法完成，因此最後是挑了個之前看到的很有趣的Disnep變臉來玩玩看，沒想到做出來結果超級怪的...

## Citations

@article{DBLP:journals/corr/abs-1812-04948,
  author    = {Tero Karras and
               Samuli Laine and
               Timo Aila},
  title     = {A Style-Based Generator Architecture for Generative Adversarial Networks},
  journal   = {CoRR},
  volume    = {abs/1812.04948},
  year      = {2018},
  url       = {http://arxiv.org/abs/1812.04948},
  eprinttype = {arXiv},
  eprint    = {1812.04948},
  timestamp = {Tue, 01 Jan 2019 15:01:25 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-1812-04948.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
@article{Karras2019stylegan2,
  title   = {Analyzing and Improving the Image Quality of {StyleGAN}},
  author  = {Tero Karras and Samuli Laine and Miika Aittala and Janne Hellsten and Jaakko Lehtinen and Timo Aila},
  journal = {CoRR},
  volume  = {abs/1912.04958},
  year    = {2019},
}
```
