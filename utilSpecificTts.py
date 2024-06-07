import os
import platform
import numpy as np
import ChatTTS
import datetime
import torch, torchaudio
import shortuuid
from utils.logger_settings import api_logger

import cn2an
import re
# from utilDigit import convert_arabic_to_chinese_in_string

def getProxy():
    if platform.system() == "Linux":
        return "192.168.0.77:18808"
    else:
        return "127.0.0.1:10809"

os.environ['HTTP_PROXY'] = getProxy()
os.environ['HTTPS_PROXY'] = getProxy()


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
# 参数说明：

# 情感控制
# speed : 控制音频速度，范围为 0-9，数字越大，速度越快
# temperate : 控制音频情感波动性，范围为 0-1，数字越大，波动性越大
# top_P ：控制音频的情感相关性，范围为 0.1-0.9，数字越大，相关性越高
# top_K ：控制音频的情感相似性，范围为 1-20，数字越小，相似性越高
# 文本控制
# Refine text : 控制是否对文本进行口语化处理，取消勾选则后面三个选项无效
# oral : 控制文本口语化程度，范围为 0-9，数字越大，添加的“就是”、“那么”之类的连接词越多
# laugh : 控制文本是否添加笑声，范围为 0-9，数字越大，笑声越多
# break : 控制文本是否添加停顿，范围为 0-9，数字越大，停顿越多
# 种子控制
# Audio Seed : 配置音色种子值，不同种子对应不同音色，不同种子间差异性较大
# Text Seed : 配置情感种子值，不同种子对应不同情感，不同种子间差异性较小

    global gChat
    if not gChat:
        gChat = ChatTTS.Chat()
        gChat.load_models()
    # if not outPath:
    #     outPath = f"./out/{getCurTimeStampStr()}.wav"
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
    params_refine_text = {'prompt': '[break_0]'}
    
    torch.manual_seed(text_seed_input)

    if refine_text_flag:
        text = gChat.infer(text, 
                          skip_refine_text=False,
                          refine_text_only=True,
                          params_refine_text=params_refine_text,
                          params_infer_code=params_infer_code
                          )
        api_logger.info(text)
        
    wavs = gChat.infer(text, 
                     skip_refine_text=True, 
                     params_refine_text=params_refine_text, 
                     params_infer_code=params_infer_code
                     )
    
    # audio_data = np.array(wav[0]).flatten()
    # sample_rate = 24000
    # text_data = text[0] if isinstance(text, list) else text
    # api_logger.info(text_data)
    # if outPath:
    #     torchaudio.save(outPath, torch.from_numpy(wav[0]), 24000)
    # # return wav[0]
    # return [(sample_rate, audio_data), text_data]

    audoArray = [torch.from_numpy(i) for i in wavs]
    combined_audio = torch.cat(audoArray, dim=1)
    api_logger.info(f"保存音频文件到  {outPath}")
    torchaudio.save(outPath, combined_audio, 24000)



# python utilSpecificTts.py --text "2004年就在 OpenAI 发布可以生成令人瞠目的视频的 Sora 和谷歌披露支持多达 150 万个Token上下文的 Gemini 1.5的几天后，Stability AI 最近展示了 Stable Diffusion 3 的预览版。" --out-path "./out/202303051504.wav" --audio-role 2222
if __name__ == "__main__":
    ans_id = getCurTimeStampStr()
    wavDir="./out/"
    outPath = f"{wavDir}{ans_id}.wav"
    text = "2004年就在 OpenAI 发布可以生成令人瞠目的视频的 Sora 和谷歌披露支持多达 150 万个Token上下文的 Gemini 1.5的几天后，Stability AI 最近展示了 Stable Diffusion 3 的预览版。"

    # text=convert_arabic_to_chinese_in_string(text)
    text = cn2an.transform(text, "an2cn")
    # chat = ChatTTS.Chat()
    # chat.load_models() 
    api_logger.info("转换")
    api_logger.info(text)
    manSeed=seeds["旁白2222"]["seed"]

    generate_audio(text, outPath, audio_seed_input=manSeed)