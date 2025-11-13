import json
import re
import unicodedata

# Qwen格式的系统指令模板
instruction = """You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
If none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out. If the result of tool calls has fulfilled the user's request, summary the answer.

**Important Notes**
1. When the tool call has fulfilled the user's request, please provide a concise summary in plain text without extra tool calls. If no tool is suitable, state that explicitly. If the user's input lacks required parameters, ask for clarification.
2. During each tool invocation, it is important to carefully examine the corresponding tool's description and constraints. Ensure that the required fields of the tool are strictly satisfied, that parameter types conform to the definitions. If function calls uses the default parameter value, it is not necessary to specify the value during the call.
3. If the user's request cannot be completed through one-time function call, or if the parameters of subsequent function calls depend on the results of previous calls, then decompose it into multi-step calls. You only need to return the result of the first step. The use of fictitious parameters or placeholder is strictly prohibited.
4. In multi-turn dialogs, if you encounter an error and the task remains unfinished, retry with more necessary tool calls until completion. Based on the tool feedback, reflect on if understanding or selection of tool is wrong, what tool calling step is missing, and how to achieve the task goal from now on."""

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
        '‹': '<', '›': '>', '«': '"', '»': '"', '’': "'",
        
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
        '～': '~', '＿': '_', '　': ' ', ''': '\'', ''': '\'',
        '"': '\"', '"': '\"'
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
    
    # 8. 移除表情符号和其他符号
    text = re.sub(r'[\U0001F600-\U0001F64F]', '', text)
    text = re.sub(r'[\U0001F300-\U0001F5FF]', '', text)
    text = re.sub(r'[\U0001F680-\U0001F6FF]', '', text)
    text = re.sub(r'[\U0001F1E0-\U0001F1FF]', '', text)
    text = re.sub(r'[\U00002702-\U000027B0]', '', text)
    text = re.sub(r'[\U000024C2-\U0001F251]', '', text)
    
    # 9. 去除首尾空格
    text = text.strip()
    
    return text

def filter_and_normalize_conversation(conversations):
    """
    过滤和标准化对话数据
    """
    filtered_conversations = []
    
    for entry in conversations:
        # 复制原始entry
        filtered_entry = entry.copy()
        
        # 标准化文本内容
        if 'value' in filtered_entry and isinstance(filtered_entry['value'], str):
            # 应用Unicode标准化
            filtered_entry['value'] = normalize_unicode_text(filtered_entry['value'])
            
            # 如果标准化后内容为空或过短，跳过这个entry
            if not filtered_entry['value']:
                print(f"跳过空或过短的对话条目: {entry.get('from', 'unknown')}")
                continue
        
        filtered_conversations.append(filtered_entry)
    
    return filtered_conversations

def xlam_json_str_to_qwen_tool_calls(tool_calls_str):
    """将xlam格式的工具调用字符串转换为Qwen的tool_call格式"""
    tool_calls = json.loads(tool_calls_str)
    if not isinstance(tool_calls, list):
        tool_calls = [tool_calls]

    qwen_format = ""
    for tool_call in tool_calls:
        if isinstance(tool_call, dict):
            name = tool_call.get("name", "")
            arguments = tool_call.get("arguments", {})
            tool_call_json = {
                "name": name,
                "arguments": arguments
            }
            qwen_format += f"<tool_call>\n{json.dumps(tool_call_json)}\n</tool_call>\n"

    return qwen_format.rstrip('\n')

def remove_chat_tokens(formatted_input):
    """移除聊天格式的开头和结尾标记"""
    # 定义要移除的开头和结尾标记
    start_token = "<|im_start|>user\n"
    end_token = "<|im_end|>\n"
    
    # 移除开头标记
    if formatted_input.startswith(start_token):
        formatted_input = formatted_input[len(start_token):]
    
    # 移除结尾标记
    if formatted_input.endswith(end_token):
        formatted_input = formatted_input[:-len(end_token)]
    
    return formatted_input

def build_qwen_system_instruction(tools, date):
    """构建符合Qwen格式的系统指令"""
    system_prompt = f"{instruction}\n\n"
    
    # if tools:
    system_prompt += f"The current time is {date}.\n"
    system_prompt += "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
    system_prompt += "You are provided with function signatures within <tools></tools> XML tags:\n<tools>"
    for tool in tools:
        system_prompt += f"\n{json.dumps(tool)}"
    system_prompt += '\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>.'

    return system_prompt

def build_qwen_conversation_history(conversations, until_index, max_turns=5):
    """构建符合Qwen格式的对话历史"""
    if until_index == 0:
        return ""

    # 找到用户消息的索引
    user_message_indices = []
    for i in range(until_index):
        if conversations[i]['from'] == 'user':
            user_message_indices.append(i)

    # 限制历史轮数
    if len(user_message_indices) > max_turns:
        start_idx = user_message_indices[-max_turns]
    else:
        start_idx = 0

    formatted_history = ""
    i = start_idx
    while i < until_index:
        entry = conversations[i]
        
        if entry['from'] == 'user':
            formatted_history += f"<|im_start|>user\n{entry['value'].strip()}<|im_end|>\n"
            
        elif entry['from'] == 'assistant':
            content = ""
            if entry['value'].startswith('['):
                # 工具调用 - 转换为Qwen格式
                content = xlam_json_str_to_qwen_tool_calls(entry['value'])
            else:
                # 普通响应
                content = entry['value'].strip()
            formatted_history += f"<|im_start|>assistant\n{content}<|im_end|>\n"
            
        elif entry['from'] == 'tool':
            if isinstance(entry['value'], str):
                tool_responses = json.loads(entry['value'])
                assert isinstance(tool_responses, list)
            else:
                tool_responses = entry['value'] # list
            # 合并tool响应
            formatted_tool_response = ""
            for tool_response in tool_responses:
                assert isinstance(tool_response, dict)
                if "name" in tool_response and 'results' in tool_response:
                    formatted_tool_response += f"<tool_response>\n{json.dumps(tool_response.get('results'))}\n</tool_response>\n"
                else:
                    formatted_tool_response += f"<tool_response>\n{list(tool_response.values())[0]}\n</tool_response>\n"
            
            formatted_history += f"<|im_start|>user\n{formatted_tool_response.rstrip()}<|im_end|>\n"
        
        i += 1
    
    return remove_chat_tokens(formatted_history)

def convert_sample_toolcall(multi_turn_sample, max_history_turns=5):
    """Convert a single multi-turn conversations into multiple GRPO training samples."""
    training_samples = []
    
    # 过滤和标准化工具数据和对话数据
    date = multi_turn_sample.get('date', 'Unknown')
    tools = multi_turn_sample['tools']
    conversations = filter_and_normalize_conversation(multi_turn_sample['conversations'])
    
    # 如果过滤后没有足够的对话内容，跳过这个样本
    if len(conversations) < 2:
        print("跳过对话内容不足的样本")
        return []
    
    # 构建系统指令
    system_instruction = build_qwen_system_instruction(tools, date)
    
    # Iterate through conversation to find assistant turns
    assistant_turns = []
    for idx, entry in enumerate(conversations):
        if satisfy_condition(conversations, idx):
            assistant_turns.append(idx)
    
    # Generate training samples for each assistant turn
    for turn_idx in assistant_turns:
        current_entry = conversations[turn_idx]

        # 构建对话历史
        conversation_history = build_qwen_conversation_history(
            conversations, 
            turn_idx,
            max_turns=max_history_turns
        )

        # 输出为Qwen格式的工具调用
        grpo_output = xlam_json_str_to_qwen_tool_calls(current_entry['value'])

        training_samples.append({
            "instruction": system_instruction,  # 系统指令已包含在input中
            "input": conversation_history,
            "output": grpo_output
        })
    
    return training_samples

def convert_sample_response(multi_turn_sample, max_history_turns=5):
    """Convert assistant response samples into GRPO training samples."""
    training_samples = []
    
    # 过滤和标准化工具数据和对话数据
    date = multi_turn_sample.get('date', 'Unknown')
    tools = multi_turn_sample['tools']
    conversations = filter_and_normalize_conversation(multi_turn_sample['conversations'])
    
    # 如果过滤后没有足够的对话内容，跳过这个样本
    if len(conversations) < 2:
        print("跳过对话内容不足的样本")
        return []
    
    # 构建系统指令
    system_instruction = build_qwen_system_instruction(tools, date)
    
    # Iterate through conversation to find assistant turns
    assistant_turns = []
    for idx, entry in enumerate(conversations):
        if entry['from'] == 'assistant' and not entry['value'].startswith('['):
            assistant_turns.append(idx)
    
    # Generate training samples for each assistant turn
    for turn_idx in assistant_turns:
        current_entry = conversations[turn_idx]

        # 构建对话历史
        conversation_history = build_qwen_conversation_history(
            conversations, 
            turn_idx,
            max_turns=max_history_turns
        )

        grpo_output = current_entry['value']

        training_samples.append({
            "instruction": system_instruction,  # 系统指令已包含在input中
            "input": conversation_history,
            "output": grpo_output
        })
    
    return training_samples
    
def satisfy_condition(conversations, idx):
    entry = conversations[idx]
    
    if entry['from'] != 'assistant' or not entry['value'].startswith('[') or not entry['value'].endswith(']') or len(entry['value'])<=2:
        return False
    if idx+1 < len(conversations):
        next_entry = conversations[idx+1]
        assert next_entry['from'] == 'tool'
        if 'error' in json.dumps(next_entry['value']).lower():
            return False
    return True

# Example usage:
if __name__ == "__main__":
    all_result = []
    skipped_samples = 0
    processed_samples = 0
    
    # 加载数据集
    with open("xxx.json", "r") as f:
        multi_turn_samples = json.load(f)
    
    # 处理每个样本
    for idx, sample in enumerate(multi_turn_samples):
        try:
            grpo_samples = convert_sample_toolcall(sample, max_history_turns=8)
            # grpo_samples = convert_sample_response(sample, max_history_turns=6)
            if grpo_samples:
                all_result.extend(grpo_samples)
                processed_samples += 1
            else:
                skipped_samples += 1
                print(f"跳过第{idx+1}个样本：处理后无有效训练数据")
        except Exception as e:
            skipped_samples += 1
            print(f"处理第{idx+1}个样本时出错: {str(e)}")
            continue
    
    print(f"成功处理{processed_samples}个函数调用样本，跳过{skipped_samples}个样本")
    print(f"总共得到{len(all_result)}条函数调用的训练数据")

    with open("xxx.json", "w") as f:
        json.dump(all_result, f, indent=2)