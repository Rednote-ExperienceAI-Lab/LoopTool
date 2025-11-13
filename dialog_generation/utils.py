# -*- coding: utf-8 -*-
import copy
import numpy as np
import collections
import os
import re
import json
import random
import unicodedata
from openai import OpenAI
import datetime as dt
from datetime import datetime, timedelta
from queue import Queue
import threading
from gpt_key import client


difficulty_config = {
    'medium': {
        'probability': 1.0,
        'tool_num_range': [20, 30],
        'turn_num_range': [20, 36],
        'tag_num': 20
    },
}

def get_all_json_files(dirpath, recursive=True):
    all_json_files = []
    if recursive:
        for subdir, dirs, files in os.walk(dirpath):
            for f in files:
                if f.endswith('.json') or f.endswith('.jsonl'):
                    relative_path = os.path.join(os.path.relpath(subdir, dirpath), f)
                    fp = os.path.abspath(os.path.join(dirpath, relative_path))
                    all_json_files.append(fp)
    else:
        all_json_files = [os.path.join(dirpath, f)
                          for f in os.listdir(dirpath)
                          if f.endswith('.json') or f.endswith('.jsonl')]
    return all_json_files


def load_data_file(file_name):
    with open(file_name, "r", encoding="utf-8") as f:
        if file_name.endswith(".json"):
            data = json.load(f)
        elif file_name.endswith(".jsonl"):
            data = [json.loads(line) for line in f]
        else:
            raise ValueError("The file loaded is not json or jsonl file!")
    return data


def dump_data_file(data, file_name):
    with open(file_name, "w", encoding="utf-8") as f:
        if file_name.endswith(".json"):
            json.dump(data, f, indent=4, ensure_ascii=False)
        elif file_name.endswith(".jsonl"):
            for d in data:
                f.write(json.dumps(d, ensure_ascii=False)+"\n")
        else:
            raise ValueError("The file loaded is not json or jsonl file!")


def generate_random_time(year=2023):
    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31)
    delta = end_date - start_date
    random_days = random.randint(0, delta.days)
    random_date = start_date + timedelta(days=random_days)
    hour = random.randint(0, 23)
    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    random_time = "{:02d}:{:02d}:{:02d}".format(hour, minute, second)
    return random_date.strftime("%Y-%m-%d") + " " + random_time


def generate_random_date(year=2023):
    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31)
    delta = end_date - start_date
    random_days = random.randint(0, delta.days)
    random_date = start_date + timedelta(days=random_days)
    day_of_week_index = random_date.weekday()
    chinese_days = ['星期一', '星期二', '星期三', '星期四', '星期五', '星期六', '星期日']
    day_of_week_chinese = chinese_days[day_of_week_index]
    return random_date.strftime("%Y{y}%m{m}%d{d}").format(y="年", m="月", d="日") + "，" + day_of_week_chinese


def generate_random_date_en(year=2023):
    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31)
    delta = end_date - start_date
    random_days = random.randint(0, delta.days)
    random_date = start_date + timedelta(days=random_days)
    day_of_week_index = random_date.weekday()
    chinese_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_of_week_chinese = chinese_days[day_of_week_index]
    return random_date.strftime("%Y{y}%m{m}%d{d}").format(y="-", m="-", d="") + ", " + day_of_week_chinese


def convert_toolace_tools_to_gpt_tools(tools):
    gpt_tools = []
    for ori_tool in tools:
        tool = copy.deepcopy(ori_tool)
        tool['parameters'] = tool.pop('parameters')
        if tool['name'] == 'python_interpreter':
            tool['parameters'] = {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "能直接执行的Python代码"
                    }
                }
            }
        if "results" in tool:
            tool.pop("results")
        gpt_tools.append({'type': 'function', 'function': tool})
    return gpt_tools


def convert_toolace_tool_usages_to_gpt_tool_calls(tool_usages, idx):  # idx: 当前turn序号，用于生成不重复的tool call id
    tool_calls = []
    for i, tool_usage in enumerate(tool_usages):
        tool_calls.append({
            'id': str(hash(str(tool_usage)))+str(idx).zfill(3)+str(i).zfill(3),
            'type': 'function',
            'function': {
                'name': tool_usage['name'],
                'parameters': json.dumps(tool_usage['parameters'], ensure_ascii=False)
            }
        })
    return tool_calls


def convert_gpt_tool_calls_to_toolace_tool_usages(gpt_tool_calls):
    tool_usages = []
    for tool_call in gpt_tool_calls:
        tool_usage = {
            "name": tool_call['function']['name'],
            "parameters": json.loads(tool_call['function']['parameters'])
        }
        tool_usages.append(tool_usage)
    return tool_usages


def convert_toolace_data_to_text(messages):
    texts = []
    for turn in messages:
        if turn["role"] == "user":
            texts.append("<user>" + turn["content"] + "</user>")
        elif turn["role"] == "assistant":
            text = turn["content"]
            if "tool_usage" in turn:
                for tool_usage in turn["tool_usage"]:
                    t_usage = tool_usage["name"] + "|" + json.dumps(tool_usage["parameters"], ensure_ascii=False)
                    text += "<tool_usage>" + t_usage + "</tool_usage>"
            texts.append("<assistant>" + text + "</assistant>")
        elif turn["role"] == "tool":
            text = ""
            for tool_response in turn["tool_response"]:
                if "results" not in tool_response:
                    tr = str(tool_response)
                elif isinstance(tool_response["results"], str):
                    tr = tool_response["results"]
                else:
                    tr = json.dumps(tool_response["results"], ensure_ascii=False)
                text += "<tool_response>" + tr + "</tool_response>"
            texts.append("<tool>" + text + "</tool>")
        else:
            assert False
    texts = "\n".join(texts)
    return texts


def convert_gpt_user_role_output_to_toolace(gpt_output):
    content = gpt_output['content'].strip()
    if len(re.findall(r"<continuation>(.*?)</continuation>", content, re.DOTALL)) == 0:
        return None
    if "<continuation>" in content and "<continuation>\n<user>" not in content:
        content = content.replace("<continuation>", "<continuation>\n<user>")
    if "</continuation>" in content and "</user>\n</continuation>" not in content:
        content = content.replace("</continuation>", "</user>\n</continuation>")
    content = content.split("<continuation>", 1)[1].split("</continuation>", 1)[0].strip()
    if not (content.startswith("<user>") and content.endswith("</user>")):
        return None
    content = content.split("<user>", 1)[1].split("</user>", 1)[0].strip()
    return {"role": "user", "content": content}


def convert_gpt_tool_role_output_to_toolace(gpt_output):
    content = gpt_output['content'].strip()
    if content.startswith("```json"):
        content = content.strip().lstrip("```json").rstrip("```").strip()
    try:
        tool_res = json.loads(content)
        assert isinstance(tool_res, list)
    except:
        return None
    return {"role": "tool", "content": "", "tool_response": tool_res}


# def convert_gpt_assistant_role_output_to_toolace(gpt_output):
#     if gpt_output['content'] is None:
#         content = ""
#     else:
#         content = gpt_output['content'].strip()
#     if 'tool_calls' in gpt_output and gpt_output['tool_calls']: # 这是使用tool_call 功能的输出时使用
#         tool_usages = convert_gpt_tool_calls_to_toolace_tool_usages(gpt_output['tool_calls'])
#         return {"role": "assistant", "content": content, "tool_usage": tool_usages}
#     elif '<tool_call>' in content or '</tool_call>' in content:
#         if not ('<tool_call>' in content and '</tool_call>' in content):
#             return None
#         tool_usages = []
#         for tu in re.findall(r"<tool_call>(.*?)</tool_call>", content, re.DOTALL):
#             try:
#                 name, parameters = tu.split("|", 1)
#                 tool_usages.append({"name": name, "parameters": json.loads(parameters)})
#             except Exception as e:
#                 print(e)
#                 return None
#         content = content.split('<tool_call>')[0].strip()
#         return {"role": "assistant", "content": content, "tool_usage": tool_usages}
#     else:  # 无工具调用的情况
#         return {"role": "assistant", "content": content}

def convert_gpt_assistant_role_output_to_toolace(gpt_output, used_model=None, max_retries=3):
    if gpt_output['content'] is None:
        content = ""
    else:
        content = gpt_output['content'].strip()
    
    if 'tool_calls' in gpt_output and gpt_output['tool_calls']: # 这是使用tool_call 功能的输出时使用
        tool_usages = convert_gpt_tool_calls_to_toolace_tool_usages(gpt_output['tool_calls'])
        return {"role": "assistant", "content": content, "tool_usage": tool_usages}
    elif '<tool_call>' in content or '</tool_call>' in content:
        if not ('<tool_call>' in content and '</tool_call>' in content):
            return None
        tool_usages = []
        for tu in re.findall(r"<tool_call>(.*?)</tool_call>", content, re.DOTALL):
            try:
                name, parameters = tu.split("|", 1)
                
                # 尝试解析JSON，如果失败则使用GPT修复
                retry_count = 0
                current_parameters = parameters
                
                while retry_count < max_retries:
                    try:
                        parsed_params = json.loads(current_parameters)
                        tool_usages.append({"name": name, "parameters": parsed_params})
                        break
                    except json.JSONDecodeError as json_e:
                        print(f"JSON parsing failed (attempt {retry_count + 1}): {json_e}")
                        
                        # 如果没有提供client或model，直接返回错误
                        if client is None or used_model is None:
                            print("No client/model provided for JSON repair")
                            return None
                        
                        # 构造修复JSON的消息
                        messages = [
                            {
                                "role": "system", 
                                "content": "You are a JSON repair assistant. Fix ONLY the JSON syntax errors (missing quotes, brackets, commas, etc.) to make it valid JSON. DO NOT modify the actual content, values, or meaning of the parameters. Keep all original data intact and only correct formatting issues. Return ONLY the corrected JSON string, no explanation or additional text."
                            },
                            {
                                "role": "user", 
                                "content": f"Fix this malformed JSON (only syntax errors, keep all content unchanged): {current_parameters}"
                            }
                        ]
                        
                        try:
                            response = client.chat.completions.create(
                                model=used_model, 
                                messages=messages,
                                temperature=0.1  # 使用较低温度确保更准确的修复
                            )
                            fixed_parameters = response.choices[0].message.content.strip()
                            
                            # 清理可能的markdown代码块标记
                            if fixed_parameters.startswith('```json'):
                                fixed_parameters = fixed_parameters[7:]
                            if fixed_parameters.startswith('```'):
                                fixed_parameters = fixed_parameters[3:]
                            if fixed_parameters.endswith('```'):
                                fixed_parameters = fixed_parameters[:-3]
                            
                            current_parameters = fixed_parameters.strip()
                            retry_count += 1
                            
                        except Exception as api_e:
                            print(f"API call failed: {api_e}")
                            return None
                
                # 如果超过最大重试次数仍无法解析，返回错误
                if retry_count >= max_retries:
                    print(f"Failed to fix JSON after {max_retries} retries")
                    return None
                    
            except Exception as e:
                print(f"Error processing tool call: {e}")
                return None
        
        content = content.split('<tool_call>')[0].strip()
        return {"role": "assistant", "content": content, "tool_usage": tool_usages}
    else:  # 无工具调用的情况
        return {"role": "assistant", "content": content}




def normalize_unicode_text(text):
    """
    将Unicode文本标准化为英文字符
    处理全角字符、特殊符号、不可见字符等
    """
    if not isinstance(text, str) or not text:
        return text
    
    # 1. Unicode标准化 - 将组合字符分解
    text = unicodedata.normalize('NFKD', text)
    
    # 2. 全角字符转半角字符的映射表
    fullwidth_to_halfwidth = {
        # 数字 0-9
        '０': '0', '１': '1', '２': '2', '３': '3', '４': '4',
        '５': '5', '６': '6', '７': '7', '８': '8', '９': '9',
        
        # 字母 A-Z
        'Ａ': 'A', 'Ｂ': 'B', 'Ｃ': 'C', 'Ｄ': 'D', 'Ｅ': 'E', 'Ｆ': 'F', 'Ｇ': 'G', 'Ｈ': 'H',
        'Ｉ': 'I', 'Ｊ': 'J', 'Ｋ': 'K', 'Ｌ': 'L', 'Ｍ': 'M', 'Ｎ': 'N', 'Ｏ': 'O', 'Ｐ': 'P',
        'Ｑ': 'Q', 'Ｒ': 'R', 'Ｓ': 'S', 'Ｔ': 'T', 'Ｕ': 'U', 'Ｖ': 'V', 'Ｗ': 'W', 'Ｘ': 'X',
        'Ｙ': 'Y', 'Ｚ': 'Z',
        
        # 字母 a-z
        'ａ': 'a', 'ｂ': 'b', 'ｃ': 'c', 'ｄ': 'd', 'ｅ': 'e', 'ｆ': 'f', 'ｇ': 'g', 'ｈ': 'h',
        'ｉ': 'i', 'ｊ': 'j', 'ｋ': 'k', 'ｌ': 'l', 'ｍ': 'm', 'ｎ': 'n', 'ｏ': 'o', 'ｐ': 'p',
        'ｑ': 'q', 'ｒ': 'r', 'ｓ': 's', 'ｔ': 't', 'ｕ': 'u', 'ｖ': 'v', 'ｗ': 'w', 'ｘ': 'x',
        'ｙ': 'y', 'ｚ': 'z'
    }
    
    # 3. 特殊标点符号映射表
    punctuation_mapping = {
        # 各种引号
        '"': '"', '"': '"', ''': "'", ''': "'", '‚': "'", '„': '"',
        '‹': '<', '›': '>', '«': '"', '»': '"',
        
        # 各种破折号
        '—': '-', '–': '-', '−': '-', '‒': '-', '―': '-',
        
        # 各种省略号
        '…': '...', '‥': '..',
        
        # 其他符号
        '•': '*', '‧': '.', '·': '.', '※': '*',
        '§': 'S', '¶': 'P', '†': '+', '‡': '+',
        '°': 'deg', '′': "'", '″': '"', '‴': "'''",
        '¿': '?', '¡': '!', '¯': '-', '˜': '~',
        
        # 货币符号
        '€': 'EUR', '£': 'GBP', '¥': 'JPY', '¢': 'c',
        
        # 数学符号
        '×': 'x', '÷': '/', '±': '+/-', '∞': 'inf',
        '≤': '<=', '≥': '>=', '≠': '!=', '≈': '~',
        
        # 箭头
        '→': '->', '←': '<-', '↑': '^', '↓': 'v',
        '⇒': '=>', '⇐': '<=', '⇔': '<=>',
        
        # 全角标点
        '！': '!', '？': '?', '；': ';', '：': ':',
        '，': ',', '。': '.', '（': '(', '）': ')',
        '［': '[', '］': ']', '｛': '{', '｝': '}',
        '／': '/', '＼': '\\', '｜': '|', '＜': '<', '＞': '>',
        '＋': '+', '－': '-', '＊': '*', '＝': '=',
        '％': '%', '＆': '&', '＃': '#', '＠': '@',
        '＄': '$', '＾': '^', '｀': '`', '｜': '|',
        '～': '~', '＿': '_', '　': ' ', '‘': '\'', '’': '\'',
        '”': '\"', '“': '\"'
         # 全角空格转半角空格
    }
    
    # 4. 应用全角转半角
    for fullwidth, halfwidth in fullwidth_to_halfwidth.items():
        text = text.replace(fullwidth, halfwidth)
    
    # 5. 应用标点符号映射
    for unicode_char, ascii_char in punctuation_mapping.items():
        text = text.replace(unicode_char, ascii_char)
    
    # 6. 处理特殊Unicode空格字符
    unicode_spaces = [
        '\u00A0',  # 不换行空格
        '\u1680',  # Ogham空格
        '\u2000',  # en quad
        '\u2001',  # em quad
        '\u2002',  # en space
        '\u2003',  # em space
        '\u2004',  # 三分之一em space
        '\u2005',  # 四分之一em space
        '\u2006',  # 六分之一em space
        '\u2007',  # 数字空格
        '\u2008',  # 标点空格
        '\u2009',  # 细空格
        '\u200A',  # 发丝空格
        '\u202F',  # 窄不换行空格
        '\u205F',  # 中等数学空格
        '\u3000',  # 中日韩表意文字空格
    ]
    
    for unicode_space in unicode_spaces:
        text = text.replace(unicode_space, ' ')
    
    # 7. 移除或替换控制字符（保留换行和制表符）
    control_chars = []
    for i in range(len(text)):
        char = text[i]
        if unicodedata.category(char).startswith('C') and char not in '\n\t\r':
            control_chars.append(char)
    
    for char in set(control_chars):
        text = text.replace(char, '')
    
    # # 8. 处理连续的空格和换行
    # text = re.sub(r' +', ' ', text)  # 多个空格替换为单个空格
    # text = re.sub(r'\n\s*\n', '\n\n', text)  # 规范化多个换行
    
    # 9. 去除首尾空格
    text = text.strip()
    
    return text

def clean_json_content(obj):
    """
    递归清理JSON对象中的所有字符串内容
    """
    if isinstance(obj, dict):
        return {key: clean_json_content(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [clean_json_content(item) for item in obj]
    elif isinstance(obj, str):
        return normalize_unicode_text(obj)
    else:
        return obj


