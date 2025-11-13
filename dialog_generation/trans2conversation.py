import json
import os
import collections

from utils import normalize_unicode_text, clean_json_content

def load_data_file(file_path):
    """加载JSON文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        if file_path.endswith('.jsonl'):
            data = [json.loads(line.strip()) for line in f if line.strip()]
        else:
            data = json.load(f)

    return clean_json_content(data)
    
def dump_data_file(data, file_path):
    """保存JSON文件"""
    # 在保存前清理数据
    cleaned_data = clean_json_content(data)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

def get_all_json_files(directory):
    """获取目录下所有JSON文件"""
    json_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(('.json', '.jsonl')):
                json_files.append(os.path.join(root, file))
    return json_files

def format_tool_for_system(tool):
    """将工具信息格式化为系统提示词中的格式"""
    tool_dict = {
        "name": tool["name"],
        "description": tool["description"],
        "parameters": tool["parameters"]
    }
    if "required" in tool:
        tool_dict["required"] = tool["required"]
    return tool_dict

def convert_tool_usage_to_function_call(tool_usage_list):
    """将tool_usage转换为函数调用格式"""
    if not tool_usage_list:
        return ""
    
    function_calls = []
    for tool_usage in tool_usage_list:
        name = tool_usage["name"]
        params = tool_usage.get("parameters", {})
        function_calls.append({
            "name": name,
            "arguments": params
        })
    
    return json.dumps(function_calls, ensure_ascii=False)

def convert_tool_response_to_tool_message(tool_responses):
    """将tool_response转换为tool消息格式"""
    if not tool_responses:
        return ""
    
    tool_results = []
    for response in tool_responses:
        tool_results.append({
            "name": response["name"],
            "results": response["results"]
        })
    
    return json.dumps(tool_results, ensure_ascii=False)

def create_system_prompt(tools):
    """创建系统提示词"""
    base_prompt = """You are a helpful multi-turn dialogue assistant capable of leveraging tool calls. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.

# Tool
In your response, you can use the following tools:
<tools>
[tools_str]
</tools>

Steps for Each Turn:
1. Think about the reasoning process and enclosed your reasoning within <think> </think> tags.
2. Provide a json object with function names and arguments within <tool_call> </tool_call>.
3. Make sure both the reasoning and the tool call steps are included together in one single reply.

**Output Format**
```plaintext
<think>Your thoughts</think>
<tool_call>
[{"name": "tool_call_name", "arguments": {"arg1": "value1", "arg2": "value2"}}, ... (additional parallel tool calls as needed)]
</tool_call>
<response>Your final response</response>
```

**Important Notes**
1. The output must strictly comply with the Output Format. Only one of <tool_call> and <response> is required. No additional content is permitted.
2. If there's no appropriate tools to apply or required parameters are missing, please directly inform me within <response> </response> tag without any tool call. Otherwise, you should use one or more necessary tool calls to complete the given task.
3. In multi-turn dialogs, if you encounter an error and the task remains unfinished, retry with more necessary tool calls until completion. You could perform tool calls for multiple rounds so you can try and error. Based on the tool feedback, reflect on if understanding or selectioin of tool is wrong, what tool calling step is missing, and how to achieve the task goal from now on.(e.g., File system tools are limited to the current directory. No path is allowed. You should consider whether the current directory is correct.)
4. Refer to the previous dialogue records in the history, including the user's queries noted as `<user>`, previous `<tool_call>`, `<response>`, and any tool feedback noted as `<obs>`.
"""

    # formatted_tools = [format_tool_for_system(tool) for tool in tools]
    tools_json = "\n".join(json.dumps(tool) for tool in formatted_tools)

    return base_prompt.replace('[tools_json_str]', tools_json)

def convert_to_conversation_format(data_item): 
    """将单个数据项转换为对话格式""" 
    tools = data_item["tools"] 
    messages = data_item["curr_messages"]

    # 创建系统提示词, 现在对话不需要system_prompt
    # system_prompt = create_system_prompt(tools)

    # 转换对话
    conversations = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        
        if msg["role"] == "user":
            conversations.append({
                "from": "user",
                "value": normalize_unicode_text(msg["content"])
            })
        
        elif msg["role"] == "assistant":
            # 处理助手消息
            content = normalize_unicode_text(msg["content"])
            
            # 如果包含CoT，移除CoT部分
            if "<cot>" in content and "</cot>" in content:
                content = content.split("</cot>")[-1].strip()
            
            # 如果有工具使用，替换为函数调用格式
            if "tool_usage" in msg and msg["tool_usage"]:
                function_call = convert_tool_usage_to_function_call(msg["tool_usage"])
                if function_call:
                    conversations.append({
                        "from": "assistant", 
                        "value": function_call
                    })
                    
                    # 查找对应的tool响应
                    if i + 1 < len(messages) and messages[i + 1]["role"] == "tool":
                        tool_msg = messages[i + 1]
                        # tool_content = convert_tool_response_to_tool_message(
                        #     tool_msg.get("tool_response", [])
                        # )
                        tool_content = tool_msg.get("tool_response", [])
                        conversations.append({
                            "from": "tool",
                            "value": tool_content
                        })
                        i += 1  # 跳过tool消息
                    
                    # # 有工具使用的情况下, 不允许添加额外的非工具使用的对话
                    # if content and not content.isspace():
                    #     conversations.append({
                    #         "from": "assistant",
                    #         "value": content
                    #     })
            else:
                # 没有工具使用的普通助手消息
                if content and not content.isspace():
                    conversations.append({
                        "from": "assistant",
                        "value": content
                    })
        
        elif msg["role"] == "tool":
            # 工具消息通常在上面的assistant处理中已经处理了
            pass
        
        i += 1

    result = {
        "tools": tools,
        "conversations": conversations
    }

    # 添加其他元信息
    if "curr_day" in data_item and data_item["curr_day"]:
        result["date"] = data_item["curr_day"]

    user_num = sum(msg["role"] == "user" for msg in data_item["curr_messages"])
    result["user_turns"] = user_num

    # 添加文件的initial_config and plan
    result['plan'] = data_item.get("plan", "None")
    result['initial_config'] = data_item.get('initial_config', "None")

    # if user_num >= 5:
    #     result["is_extreme_multi_turn"] = True
        
    return result
def post_process_to_conversation_format(crawled_files_path, tools_path, group_tool, output_path): 
    """ 将爬取的数据转换为对话格式的训练数据集
    Args:
        crawled_files_path: 爬取数据的路径
        tools_path: 工具定义文件路径
        group_tool: 是否按组处理工具
        output_path: 输出路径
    """
    print("开始处理数据...")

    # 加载工具信息
    curr_tools = []
    for file in get_all_json_files(tools_path):
        curr_tools.extend(load_data_file(file))

    # 加载爬取的数据
    crawled_data = []
    for file in get_all_json_files(crawled_files_path):
        crawled_data.extend(load_data_file(file))

    print(f"总共加载了 {len(crawled_data)} 条原始数据")

    # 找到每个cid的最终数据（最长的有效对话）
    data_current = collections.defaultdict(dict)
    for d in crawled_data:
        if d['cid'] not in data_current:
            data_current[d['cid']] = d
        elif d['valid'] and len(data_current[d['cid']].get('curr_messages',[])) <= len(d.get('curr_messages',[])):
            data_current[d['cid']] = d

    print(f"去重后有 {len(data_current)} 条对话")

    # 过滤和转换数据
    valid_conversations = []
    invalid_count = 0
    only_query_count = 0
    no_usage_count = 0

    for cid in data_current:
        d = data_current[cid]
        
        # 跳过无效数据
        if not d['valid']:
            invalid_count += 1
            continue
        
        # 跳过只有一条消息的数据
        if len(d.get('curr_messages', [])) <= 1:
            only_query_count += 1
            continue
        
        # 检查是否有工具使用
        has_tool_usage = any("tool_usage" in msg and msg["tool_usage"] for msg in d.get('curr_messages', []))
        
        # 跳过没有工具使用的数据
        if not has_tool_usage:
            no_usage_count += 1
            continue
        
        # 确保对话以assistant 或者 tool 结尾
        while d["curr_messages"] and d["curr_messages"][-1]["role"] not in ["assistant", "tool"]:
            d["curr_messages"].pop()
        
        # 如果处理后没有消息了，跳过
        if not d["curr_messages"]:
            invalid_count += 1
            continue
        
        # 转换为对话格式
        try:
            conversation_data = convert_to_conversation_format(d)
            valid_conversations.append(conversation_data)
            
        except Exception as e:
            print(f"转换对话 {cid} 时出错: {e}")
            invalid_count += 1
            continue

    print(f"统计信息:")
    print(f"  - 有效对话: {len(valid_conversations)}")
    print(f"  - 无效数据: {invalid_count}")
    print(f"  - 仅查询数据: {only_query_count}")
    print(f"  - 无工具使用: {no_usage_count}")

    # 保存结果
    os.makedirs(output_path, exist_ok=True)

    # 保存所有有效对话
    output_file = os.path.join(output_path, "conversation_dataset.json")
    dump_data_file(valid_conversations, output_file)
    print(f"已保存 {len(valid_conversations)} 条对话数据到 {output_file}")

    print("数据处理完成！")
    return valid_conversations

# 使用示例
if __name__ == "__main__":
    # 调用新的处理函数
    crawled_files_path = "./dialogs/11_06_toolbench"  # 替换为你的爬取文件路径
    tools_path = "tools/toolbench"                   # 替换为你的工具文件路径
    output_path = "./conversations/toolbench_1106"                 # 替换为你的输出路径
    
    conversations = post_process_to_conversation_format(
        crawled_files_path=crawled_files_path,
        tools_path=tools_path,
        group_tool=False,  
        output_path=output_path
    )