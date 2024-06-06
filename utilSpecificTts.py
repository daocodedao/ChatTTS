import torch
import numpy as np
import ChatTTS
import datetime
import torch, torchaudio
import shortuuid
from utils.logger_settings import api_logger

gChat = None
seeds = {
    "旁白2222": {"seed": 2222},
    "中年女性7869": {"seed": 7869},
    "年轻女性6615": {"seed": 6615},
    "中年男性4099": {"seed": 4099},
    "年轻男性6653": {"seed": 6653},
}

def getCurTimeStampStr():
    timestamp = int(datetime.datetime.now().timestamp())
    string_timestamp = str(timestamp)
    return string_timestamp

def deterministic(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  

def generate_audio(text, 
                   outPath,
                   temperature = 0.3, 
                   top_P = 0.7, 
                   top_K = 20, 
                   audio_seed_input = 2222, 
                   text_seed_input = 42, 
                   refine_text_flag = True):

    global gChat
    if not gChat:
        gChat = ChatTTS.Chat()
        gChat.load_models()
    if not outPath:
        outPath = f"./out/{getCurTimeStampStr()}.wav"
    if not text or len(text) == 0:
        api_logger.error("generate_audio text is none")
        exit(1)

    # torch.manual_seed(audio_seed_input)
    deterministic(audio_seed_input)
    rand_spk = gChat.sample_random_speaker()
    params_infer_code = {
        'spk_emb': rand_spk, 
        'temperature': temperature,
        'top_P': top_P,
        'top_K': top_K,
        }
    params_refine_text = {'prompt': '[oral_2][laugh_0][break_6]'}
    
    torch.manual_seed(text_seed_input)

    if refine_text_flag:
        text = gChat.infer(text, 
                          skip_refine_text=False,
                          refine_text_only=True,
                          params_refine_text=params_refine_text,
                          params_infer_code=params_infer_code
                          )
    
    wav = gChat.infer(text, 
                     skip_refine_text=True, 
                     params_refine_text=params_refine_text, 
                     params_infer_code=params_infer_code
                     )
    
    # audio_data = np.array(wav[0]).flatten()
    # sample_rate = 24000
    # text_data = text[0] if isinstance(text, list) else text

    torchaudio.save(outPath, torch.from_numpy(wav[0]), 24000)
    # return [(sample_rate, audio_data), text_data]


if __name__ == "__main__":
    ans_id = getCurTimeStampStr()
    wavDir="./out/"
    outPath = f"{wavDir}{ans_id}.wav"
    text = "四川美食确实以辣闻名，但也有不辣的选择。比如甜水面、赖汤圆、蛋烘糕、叶儿粑等，这些小吃口味温和，甜而不腻，也很受欢迎。"

    # chat = ChatTTS.Chat()
    # chat.load_models() 
    manSeed=seeds["旁白2222"]["seed"]

    generate_audio(text, outPath, audio_seed_input=manSeed)