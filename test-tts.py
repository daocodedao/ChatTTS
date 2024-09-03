import ChatTTS
from IPython.display import Audio
import torch, torchaudio
import shortuuid
import time,datetime,json,os,sys
from utilSpecificTts import getCurTimeStampStr, generate_audio
import cn2an
from utils.logger_settings import api_logger
import numpy as np
from tools.audio import pcm_arr_to_mp3_view


def save_mp3_file(wav, index):
    data = pcm_arr_to_mp3_view(wav)
    mp3_filename = f"output_audio_{index}.mp3"
    with open(mp3_filename, "wb") as f:
        f.write(data)
    api_logger.info(f"Audio saved to {mp3_filename}")


ans_id = getCurTimeStampStr()
wavDir="./out/"
outPath = f"{wavDir}{ans_id}.mp3"

os.makedirs(wavDir, exist_ok=True)

text = "在以色列军队对加沙拉法赫地区的持续军事行动中，存在一种普遍的看法，即即将取得胜利并彻底消灭哈马斯。\n然而，最近的发展表明，以色列可能不会迅速实现其目标，这可能会进一步加剧中东地区的紧张局势。"


srcText = cn2an.transform(text, "an2cn")
api_logger.info("转换")
api_logger.info(text)
audioRole=2222

from tools.audio import float_to_int16, has_ffmpeg_installed, load_audio

use_mp3 = has_ffmpeg_installed()
if not use_mp3:
    api_logger.warning("no ffmpeg installed, use wav file output")

srcText = srcText.strip("\n")
srcText = srcText.replace("\n\n", "\n")
texts = srcText.split("\n")

wavs_list = []
for i,text in enumerate(texts):
    api_logger.info(f"准备TTS {text}")
    outPathIdx = f"{wavDir}{ans_id}_{i}.mp3"
    audios = generate_audio(text, outPathIdx, audio_seed_input=audioRole)
    # wavs_list = wavs_list + [torch.from_numpy(i) for i in wavs]
    wavs_list.append(audios)

# api_logger.info(f"音频文件长度  {len(wavs_list)}")

# # combined_audio = torch.cat(wavs_list, dim=0)
# api_logger.info(f"保存音频文件到  {outPath}")
# # torchaudio.save(outPath, combined_audio, 24000)
# torchaudio.save(outPath, torch.from_numpy(np.concatenate(wavs_list, axis=1)), 24000)
# try:
#     torchaudio.save(outPath, torch.from_numpy(wavs[0]).unsqueeze(0), 24000)
# except:
#     torchaudio.save(outPath, torch.from_numpy(wavs[0]), 24000)