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
text = "在以色列军队对加沙拉法赫地区的持续军事行动中，存在一种普遍的看法，即即将取得胜利并彻底消灭哈马斯。\n然而，最近的发展表明，以色列可能不会迅速实现其目标，这可能会进一步加剧中东地区的紧张局势。\n尽管投入了大量资金，但拉法赫的进展明显缓慢。\n国际组织报告称，该地区遭受了超过7万吨的集中轰炸，超过了二战期间伦敦、汉堡和德累斯顿等城市 的历史水平。\n尽管遭受如此猛烈的攻击，以色列并未达到消灭哈马斯的目标，哈马斯继续抵抗。"

# text=convert_arabic_to_chinese_in_string(text)
srcText = cn2an.transform(text, "an2cn")
# chat = ChatTTS.Chat()
# chat.load_models() 
api_logger.info("转换")
api_logger.info(text)
audioRole=2222

srcText = srcText.strip("\n")
srcText = srcText.replace("\n\n", "\n")
texts = srcText.split("\n")

audioArray = []
for text in texts:
    api_logger.info(f"准备TTS {text}")
    audios = generate_audio(text, outPath, audio_seed_input=audioRole)
    audioArray = audioArray + [torch.from_numpy(i) for i in audios]


combined_audio = torch.cat(audioArray, dim=1)
api_logger.info(f"保存音频文件到  {outPath}")
torchaudio.save(outPath, combined_audio, 24000)