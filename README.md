## Note:
Code Tutorial + Implementation Tutorial

<a href="https://www.youtube.com/watch?v=wcqLFDXaDO8">
   <img alt="Qries" src="https://user-images.githubusercontent.com/61938694/154516539-90e2d4d0-4383-41f4-ad32-4c6d67bd2442.jpg"
   width="300">
</a>

<a href="https://www.youtube.com/watch?v=_Br5WRwUz_U">
   <img alt="Qries" src="https://user-images.githubusercontent.com/61938694/154628085-eede604f-442d-4bdb-a1ed-5ad3264e5aa0.jpg"
   width="300">
</a>

# VQGAN
Vector Quantized Generative Adversarial Networks (VQGAN) is a generative model for image modeling. It was introduced in [Taming Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2012.09841). The concept is build upon two stages. The first stage learns in an autoencoder-like fashion by encoding images into a low-dimensional latent space, then applying vector quantization by making use of a codebook. Afterwards, the quantized latent vectors are projected back to the original image space by using a decoder. Encoder and Decoder are fully convolutional. The second stage is learning a transformer for the latent space. Over the course of training it learns which codebook vectors go along together and which not. This can then be used in an autoregressive fashion to generate before unseen images from the data distribution.

## Results for First Stage (Reconstruction):


### 1. Epoch:

<img src="https://user-images.githubusercontent.com/61938694/154057590-3f457a92-42dd-4912-bb1e-9278a6ae99cc.jpg" width="500">


### 50. Epoch:

<img src="https://user-images.githubusercontent.com/61938694/154057511-266fa6ce-5c45-4660-b669-1dca0841823f.jpg" width="500">



## Results for Second Stage (Generating new Images):
Original Left | Reconstruction Middle Left | Completion Middle Right | New Image Right
### 1. Epoch:

<img src="https://user-images.githubusercontent.com/61938694/154058167-9627c71c-d180-449a-ba18-19a85843cee2.jpg" width="500">

### 100. Epoch:

<img src="https://user-images.githubusercontent.com/61938694/154058563-700292b6-8fbb-4ba1-b4d7-5955030e4489.jpg" width="500">

Note: Let the model train for even longer to get better results.

<hr>

## Train VQGAN on your own data:
### Training First Stage
1. (optional) Configure Hyperparameters in ```training_vqgan.py```
2. Set path to dataset in ```training_vqgan.py```
3. ```python training_vqgan.py```

### Training Second Stage
1. (optional) Configure Hyperparameters in ```training_transformer.py```
2. Set path to dataset in ```training_transformer.py```
3. ```python training_transformer.py```


## Citation
```bibtex
@misc{esser2021taming,
      title={Taming Transformers for High-Resolution Image Synthesis}, 
      author={Patrick Esser and Robin Rombach and Björn Ommer},
      year={2021},
      eprint={2012.09841},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
