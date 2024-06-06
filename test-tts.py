import ChatTTS
from IPython.display import Audio
import torch, torchaudio
import shortuuid
import time,datetime,json,os,sys


def getCurTimeStampStr():
    timestamp = int(datetime.datetime.now().timestamp())
    string_timestamp = str(timestamp)
    return string_timestamp
  

chat = ChatTTS.Chat()
chat.load_models(compile=False) # Set to True for better performance

texts = ["这个函数会返回一个四元组",]

rand_spk = chat.sample_random_speaker()
params_infer_code = {
    'prompt':'[speed_5]',
    'temperature':.3,
    'spk_emb': rand_spk,
}
uuid=getCurTimeStampStr()
modelDir="speakerModel/"
os.makedirs(modelDir, exist_ok=True)
torch.save(rand_spk, f'{modelDir}{uuid}.pth')

wavs = chat.infer(texts, use_decoder=True, params_infer_code=params_infer_code)

ans_id = shortuuid.uuid()
wavDir="./outWav/"
torchaudio.save(f"{wavDir}{uuid}.wav", torch.from_numpy(wavs[0]), 24000)