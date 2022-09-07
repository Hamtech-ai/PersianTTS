# Training
this is a guide to train a persian TTS using TensorflowTTS.
in this file i will explain the steps we have to take for training a persian tts. i provide you with a bash script file which does all these steps automatically but its recommended to do these steps manually and just read the script to get the general idea.

step 0:
it's recommended to create a new conda environment and activate it before training. to prevent any dependency mismatch, there is a environment.yml in this zip file with which you can create a conda env with all the requierments installed. 

step 1:
you need to have a persian preprocessed dataset.

step 2:
clone TensorflowTTS tag v0.9
https://github.com/TensorSpeech/TensorFlowTTS
this is the repo

step 3:
you need to replace the modifed files for persian trainig.
then you need to install TensorflowTTS.
$ pip install .

step 4:
extract and preproccess the data
this is an example command for the preprocess. you need to run both preprocess and normalize commands if your dataset is in ./ljspeech dir.
tensorflow-tts-preprocess --rootdir ./ljspeech --outdir ./dump --config preprocess/ljspeech_preprocess.yaml --dataset ljspeech
tensorflow-tts-normalize --rootdir ./dump --outdir ./dump --config preprocess/ljspeech_preprocess.yaml --dataset ljspeech


step 5:
start the train for tacotron2. the train countinues for 200k steps but around 50k-70k steps could do the work.

CUDA_VISIBLE_DEVICES=0 python examples/tacotron2/train_tacotron2.py \
  --train-dir ./dump/train/ \
  --dev-dir ./dump/valid/ \
  --outdir ./out/train.tacotron2.v1/ \
  --config ./examples/tacotron2/conf/tacotron2.v1.yaml \
  --use-norm 1 \
  --mixed_precision 0 \
  --resume ""

step 6:
use the trained tacotron2 to extract the durations for fastspeech2. you can do that using this command. you can replace checkpoint with your last checkpoint.

CUDA_VISIBLE_DEVICES=0 python examples/tacotron2/extract_duration.py \
  --rootdir ./dump/train/ \
  --outdir ./dump/train/durations/ \
  --checkpoint ./out/train.tacotron2.v1/checkpoints/model-65000.h5 \
  --use-norm 1 \
  --config ./examples/tacotron2/conf/tacotron2.v1.yaml \
  --batch-size 32 \
  --win-front 3 \
  --win-back 3 

do that for valid files too:

CUDA_VISIBLE_DEVICES=0 python examples/tacotron2/extract_duration.py \
  --rootdir ./dump/valid/ \
  --outdir ./dump/valid/durations/ \
  --checkpoint ./out/train.tacotron2.v1/checkpoints/model-65000.h5 \
  --use-norm 1 \
  --config ./examples/tacotron2/conf/tacotron2.v1.yaml \
  --batch-size 32 \
  --win-front 3 \
  --win-back 3 

step 7:
train fastspeech2:

you can do that with this command:
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

step 8:
Decode mel-spectrogram from folder ids
CUDA_VISIBLE_DEVICES=0 python examples/fastspeech2/decode_fastspeech2.py \
  --rootdir ./dump/valid \
  --outdir ./predictions/fastspeech2.v1/ \
  --config ./examples/fastspeech2/conf/fastspeech2.v1.yaml \
  --checkpoint ./out/train.fastspeech2.v1/checkpoints/model-150000.h5 \
  --batch-size 8
step 9:
Decode audio using this command:
CUDA_VISIBLE_DEVICES=0 python examples/multiband_melgan/decode_mb_melgan.py \
  --rootdir ./predictions/fastspeech2.v1/ \
  --outdir ./prediction/multiband_melgan.v1/ \
  --checkpoint ./examples/multiband_melgan/exp/train.multiband_melgan.v1/checkpoints/generator-940000.h5 \
  --config ./examples/multiband_melgan/conf/multiband_melgan.v1.yaml \
  --batch-size 32 \
  --use-norm 1