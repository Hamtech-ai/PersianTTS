#!/bin/bash
#step 0
conda env create -f environment.yml
conda activate tf_gpu

#step 1
wget [Dataset]

#step 2
git clone --depth 1 --branch v0.9 https://github.com/TensorSpeech/TensorFlowTTS

#step 3
rsync -a ./TensorFlowTTS/ ./mods/

#step 4
unzip LJSpeech-1.1.zip
mv LJSpeech-1.1 ./TensorFlowTTS/ljspeech

cd TensorFlowTTS
pip install .

tensorflow-tts-preprocess --rootdir ./ljspeech --outdir ./dump --config preprocess/ljspeech_preprocess.yaml --dataset ljspeech
tensorflow-tts-normalize --rootdir ./dump --outdir ./dump --config preprocess/ljspeech_preprocess.yaml --dataset ljspeech

#step 5

CUDA_VISIBLE_DEVICES=0 python examples/tacotron2/train_tacotron2.py \
  --train-dir ./dump/train/ \
  --dev-dir ./dump/valid/ \
  --outdir ./out/train.tacotron2.v1/ \
  --config ./examples/tacotron2/conf/tacotron2.v1.yaml \
  --use-norm 1 \
  --mixed_precision 0 \
  --resume ""

# step 6

CUDA_VISIBLE_DEVICES=0 python examples/tacotron2/extract_duration.py \
  --rootdir ./dump/train/ \
  --outdir ./dump/train/durations/ \
  --checkpoint ./out/train.tacotron2.v1/checkpoints/model-65000.h5 \
  --use-norm 1 \
  --config ./examples/tacotron2/conf/tacotron2.v1.yaml \
  --batch-size 32 \
  --win-front 3 \
  --win-back 3 

CUDA_VISIBLE_DEVICES=0 python examples/tacotron2/extract_duration.py \
  --rootdir ./dump/valid/ \
  --outdir ./dump/valid/durations/ \
  --checkpoint ./out/train.tacotron2.v1/checkpoints/model-65000.h5 \
  --use-norm 1 \
  --config ./examples/tacotron2/conf/tacotron2.v1.yaml \
  --batch-size 32 \
  --win-front 3 \
  --win-back 3 

# step 7

CUDA_VISIBLE_DEVICES=0 python examples/fastspeech2/train_fastspeech2.py \
  --train-dir ./dump/train/ \
  --dev-dir ./dump/valid/ \
  --outdir ./out/train.fastspeech2.v1/ \
  --config ./examples/fastspeech2/conf/fastspeech2.v1.yaml \
  --use-norm 1 \
  --f0-stat ./dump/stats_f0.npy \
  --energy-stat ./dump/stats_energy.npy \
  --mixed_precision 1 \
  --resume ""

# step 8

CUDA_VISIBLE_DEVICES=0 python examples/fastspeech2/decode_fastspeech2.py \
  --rootdir ./dump/valid \
  --outdir ./predictions/fastspeech2.v1/ \
  --config ./examples/fastspeech2/conf/fastspeech2.v1.yaml \
  --checkpoint ./out/train.fastspeech2.v1/checkpoints/model-150000.h5 \
  --batch-size 8

# step 9


CUDA_VISIBLE_DEVICES=0 python examples/multiband_melgan/decode_mb_melgan.py \
  --rootdir ./predictions/fastspeech2.v1/ \
  --outdir ./prediction/multiband_melgan.v1/ \
  --checkpoint ./libritts_24k.h5 \
  --config ./multiband_melgan.v1_24k.yaml \
  --batch-size 32 \
  --use-norm 1


