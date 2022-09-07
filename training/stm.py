from pydub import AudioSegment
from os import listdir
from os.path import isfile, join
mypath = './wavs/'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
for wav in onlyfiles:
    sound = AudioSegment.from_wav(mypath+wav)
    sound = sound.set_channels(1)
    sound.export("./wavs/"+wav, format="wav")
