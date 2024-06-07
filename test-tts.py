import ChatTTS
from IPython.display import Audio
import torch, torchaudio
import shortuuid
import time,datetime,json,os,sys
from utilSpecificTts import getCurTimeStampStr, generate_audio
import cn2an
from utils.logger_settings import api_logger

ans_id = getCurTimeStampStr()
wavDir="./out/"
outPath = f"{wavDir}{ans_id}.mp3"
text = "2004年就在 OpenAI 发布可以生成令人瞠目的视频的 Sora \n谷歌披露支持多达 150 万个Token上下文的 Gemini 1.5\n几天后，Stability AI 最近展示了 Stable Diffusion 3 的预览版。"

# text=convert_arabic_to_chinese_in_string(text)
srcText = cn2an.transform(text, "an2cn")
# chat = ChatTTS.Chat()
# chat.load_models() 
api_logger.info("转换")
api_logger.info(text)
manSeed=2222

srcText = srcText.strip("\n")
srcText = srcText.replace("\n\n", "\n")
texts = srcText.split("\n")

generate_audio(texts, outPath, audio_seed_input=manSeed)