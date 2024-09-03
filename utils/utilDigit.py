import re

def arabic_to_chinese_numeral(match):
    num = match.group()
    if '.' in num:
        # 处理小数
        integer_part, decimal_part = num.split('.')
        chinese_integer = convert_integer(integer_part)
        chinese_decimal = ''.join(convert_digit(digit) for digit in decimal_part)
        return f"{chinese_integer}点{chinese_decimal}"
    else:
        # 处理整数
        return convert_integer(num)

def convert_digit(digit):
    """将单个阿拉伯数字字符转换为中文数字字符"""
    digits = {'0': '零', '1': '一', '2': '二', '3': '三', '4': '四', '5': '五', '6': '六', '7': '七', '8': '八', '9': '九'}
    return digits[digit]

def convert_integer(num):
    """将整数部分转换为中文数字，考虑特殊规则如‘一万’等"""
    chinese_numerals = {
        '0': '', '1': '一', '2': '二', '3': '三', '4': '四', '5': '五', '6': '六', '7': '七', '8': '八', '9': '九',
        '10': '十', '11': '十一', '12': '十二', '13': '十三', '14': '十四', '15': '十五', '16': '十六', '17': '十七', '18': '十八', '19': '十九',
        '20': '二十', '30': '三十', '40': '四十', '50': '五十', '60': '六十', '70': '七十', '80': '八十', '90': '九十',
    }
    units = ['', '十', '百', '千', '万', '十万', '百万', '千万', '亿']
    
    num_str = str(num)
    length = len(num_str)
    result = []
    for i, digit in enumerate(reversed(num_str)):
        if digit != '0':
            numeral = chinese_numerals.get(num_str[-(i+1)*2:i+1], '') if i > 0 and length - (i+1)*2 >= 2 else chinese_numerals[digit]
            unit = units[i] if i < len(units) else ''
            result.append(numeral + unit)
    
    # 特殊处理连续的零和末尾的零
    result = [part for part in reversed(result) if part != '零' or result.index(part) == 0]
    
    return ''.join(result)

def convert_arabic_to_chinese_in_string(s):
    output_text = re.sub(r'\d+(\.\d+)?', arabic_to_chinese_numeral, s)
    return output_text

# input_text = "就在 OpenAI 发布可以生成令人瞠目的视频的 Sora 和谷歌披露支持多达 150 万个Token上下文的 Gemini 1.5的几天后，Stability AI 最近展示了 Stable Diffusion 3 的预览版。"
# output_text = convert_arabic_to_chinese_in_string(input_text)
# print(output_text)