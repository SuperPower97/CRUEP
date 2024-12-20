# CRUEP Echo Chambers of Relevance: Predicting User Engagement via Compressed Retrieval Knowledge Augmentation

## Environmental Settings

Our experiments are conducted on Ubuntu 22.04, a single NVIDIA 4090D GPU, 128GB RAM, and Intel i7-13700KF. SKAPP is implemented by `Python 3.8`, `Cuda 12.4`.

Create a virtual environment and install GPU-support packages via [Anaconda](https://www.anaconda.com/):
```shell
# create virtual environment
conda create --name CRUEP python=3.8 cudatoolkit=12.4

# activate virtual environment
conda activate CRUEP

# install other dependencies
pip install -r requirements.txt
```

## Usage

Here we take the sample of ICIP dataset as an example to demonstrate the usage.

### Preprocess

Step 1: build the dataset:
Use src/preprocess/preprocess.py to preprocess the dataset. The pre-trained model to be used can be downloaded from the following link:

Image2text: https://huggingface.co/Salesforce/blip-image-captioning-large

Image2vec: https://huggingface.co/google/vit-base-patch16-224-in21k

Text2vec: https://huggingface.co/SeanLee97/angle-bert-base-uncased-nli-en-v1

Change the `image_path` and `dataset_path` to your own path and change the `model_path` to the path of the pre-trained model.

```shell
python src/preprocess/icip_process/1_build_dataset.py
```

Step 2: process the dataset:
```shell
python src/preprocess/icip_process/2_preprocess.py
```

### Pre-training
Step 1: Self-boosting Compression:

Note that we use path `ICIP/origin` as the dataset path 

```shell
python src/IB_pretrain/IB_pretrain.py
```

### Retrieve
Step 1: Curated Retriever:
```shell
python src/curated_retriever/ICIP/compress_retrieval.py
```

### Train
Step 1: Heterogeneous UGC Graph Contriving for training

```shell
python src/train_hetero.py \
  --seed=2024 \
  --device=cuda:0 \
  --metric=MSE \
  --save=RESULT \
  --epochs=1000 \
  --batch_size=256 \
  --early_stop_turns=5 \
  --loss=MSE \
  --optim=Adam \
  --lr=1e-4 \
  --decay_rate=1.0 \
  --dataset=ICIP \
  --dataset_path=datasets \
  --retrieval_num=500 \
  --feature_dim=1600 \
  --model=CRUEP \
  --ib_fusion=True
```

### Test

Step 1: Heterogeneous UGC Graph Contriving for testing

Replace the path `model_path` with the model parameter path obtained by training the code `python src/train_hetero.py` above.

```shell
python src/test_hetero.py \
  --seed=2024 \
  --device=cuda:0 \
  --metric='MSE,SRC,MAE' \
  --save=RESULT \
  --batch_size=256 \
  --feature_dim=1600 \
  --dataset=ICIP \
  --dataset_path=datasets \
  --model=CRUEP \
  --retrieval_num=500 \
  --model_path="" \
  --ib_fusion=Trus
```




