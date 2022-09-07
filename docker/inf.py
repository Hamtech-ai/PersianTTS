from phonemizer import phonemize as pho 
import yaml
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow_tts.inference import AutoConfig
from tensorflow_tts.inference import TFAutoModel
from tensorflow_tts.inference import AutoProcessor
import soundfile as sf

from fastapi import FastAPI
from pydantic import BaseModel, ValidationError, validator
from fastapi.responses import FileResponse



processor = AutoProcessor.from_pretrained(
    pretrained_path="res/ljspeech_mapper.json"
)

config = AutoConfig.from_pretrained("res/fastspeech2.v1.yaml")
fastspeech2 = TFAutoModel.from_pretrained(
    config=config, 
    pretrained_path=None, # "../examples/fastspeech2/checkpoints/model-150000.h5",
    is_build=True, # don't build model if you want to save it to pb. (TF related bug)
    name="fastspeech2"
)

fastspeech2.load_weights("res/model-300000.h5")

# initialize melgan model
melgan_config = AutoConfig.from_pretrained('res/multiband_melgan.v1_24k.yaml')
melgan = TFAutoModel.from_pretrained(
    config=melgan_config,
    pretrained_path="res/libritts_24k.h5"
)

# Opened model in global scope


app = FastAPI(title="Persian Text-To-Speech")

    
class Data(BaseModel):
    text: str
    text_type: str

    @validator('text')
    def text_max_len(cls, v):
        assert (len(v) < 200, 'text is too long')
        return v

    @validator('text_type')
    def text_type_check(cls, v):
        assert (v == 'txt' or v == 'pho', 'text_type should be either "txt" or "pho"')
        return v


@app.post("/predict")
def predict(data: Data):

    if data.text_type == 'txt':
        input_text= pho (data.text,
             language='fa',
             backend='espeak',
             preserve_punctuation=True
             )
    else:
        input_text = data.text

    input_ids = processor.text_to_sequence(input_text)

    mel_before, mel_after, duration_outputs, _, _ = fastspeech2.inference(
        input_ids=tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
        speaker_ids=tf.convert_to_tensor([0], dtype=tf.int32),
        speed_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
        f0_ratios =tf.convert_to_tensor([1.0], dtype=tf.float32),
        energy_ratios =tf.convert_to_tensor([1.0], dtype=tf.float32)
    )

    audio_before = melgan.inference(mel_before)[0, :, 0]
    audio_after = melgan.inference(mel_after)[0, :, 0]

    sf.write('./audio_after.wav', audio_after, 22050, "PCM_16")
    #################log##################
    return FileResponse('./audio_after.wav')

