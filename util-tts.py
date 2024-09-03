import torch, torchaudio
import shortuuid
import time,datetime,json,os,sys
from utilSpecificTts import getCurTimeStampStr, generate_audio
import cn2an
from utils.logger_settings import api_logger
import argparse
import platform
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--text-prompt", type=str)
parser.add_argument("--out-path", type=str)
parser.add_argument("--audio-role", type=int, default=2222)
parser.add_argument("--process-id", type=str)

args = parser.parse_args()

srcText = args.text_prompt
if not srcText:
    api_logger.error("generate_audio text is none")
    exit(1)

outPath = args.out_path
if not outPath:
    api_logger.error("generate_audio outPath is none")
    exit(1)

audioRole = args.audio_role
if not audioRole:
    audioRole = 2222

api_logger.info("输入文字：" )
api_logger.info(srcText)
if platform.system() == "Darwin":
    srcText = cn2an.transform(srcText, "an2cn")
    api_logger.info("转换后文字：" )
    api_logger.info(srcText)

srcText = srcText.strip("\n")
srcText = srcText.replace("\\n", "\n")
srcText = srcText.replace("\n\n", "\n")
texts = srcText.split("\n")

api_logger.info(f"文字数组长度 {len(texts)}")
wavs_list = []
for text in texts:
    api_logger.info(f"准备TTS {text}")
    audios = generate_audio(text, outPath, audio_seed_input=audioRole)
    # audioArray = audioArray + [torch.from_numpy(i) for i in audios]
    wavs_list = wavs_list + [i for i in audios]

# combined_audio = torch.cat(audioArray, dim=1)
api_logger.info(f"保存音频文件到  {outPath}")
torchaudio.save(outPath, np.concatenate(wavs_list, axis=1), 24000)
# torchaudio.save(outPath, torch.from_numpy(wavs[0]), 24000)

    