import torch, torchaudio
import shortuuid
import time,datetime,json,os,sys
from utilSpecificTts import getCurTimeStampStr, generate_audio, merge_audio_files 
import cn2an
from utils.logger_settings import api_logger
import argparse
import platform
import numpy as np



parser = argparse.ArgumentParser()
parser.add_argument("--text-prompt", type=str)
parser.add_argument("--out-path", type=str)
parser.add_argument("--audio-role", type=int, default=2222)
# parser.add_argument("--process-id", type=str)

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

# if platform.system() == "Darwin":
#     # 阿拉伯数字 转换为 中文数字
#     srcText = cn2an.transform(srcText, "an2cn")
#     api_logger.info("转换后文字：" )
#     api_logger.info(srcText)

srcText = srcText.strip("\n")
srcText = srcText.replace("\\n", "\n")
srcText = srcText.replace("\n\n", "\n")
texts = srcText.split("\n")

os.makedirs(outPath, exist_ok=True)
outDir = os.path.dirname(outPath)
file_name_without_extension = os.path.splitext(os.path.basename(outPath))[0]


api_logger.info(f"文字数组长度 {len(texts)}")
wavsPathList = []
for idx,text in enumerate(texts) :
    api_logger.info(f"准备TTS {text}")
    outIdxPath = f"{outDir}/{file_name_without_extension}_{idx}.mp3"
    audios = generate_audio(text, outIdxPath, audio_seed_input=audioRole)
    # audioArray = audioArray + [torch.from_numpy(i) for i in audios]
    # wavsPathList = wavsPathList + [i for i in audios]
    if os.path.exists(outIdxPath):
        wavsPathList.append(outIdxPath)


api_logger.info(f"保存音频文件到  {outPath}")
merge_audio_files(wavsPathList, outPath)

api_logger.info(f"删除临时文件")
for tmpTath in wavsPathList:
    os.remove(tmpTath)

# combined_audio = torch.cat(audioArray, dim=1)

# torchaudio.save(outPath, np.concatenate(wavsPathList, axis=1), 24000)
# torchaudio.save(outPath, torch.from_numpy(wavs[0]), 24000)

    