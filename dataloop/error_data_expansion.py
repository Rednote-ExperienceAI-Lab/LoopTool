#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import os
import json
import numpy as np
import pandas as pd
import argparse
import ast
import builtins
import copy
import operator
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Optional, Type, Union, Dict, Tuple, Any
from collections import Counter
from openai import OpenAI

single_tool_example = {
    'input': "USER\nI'd appreciate a breakdown on the current stock market trends so I can determine the suitability of executing a trade right now. Could you provide the latest market status for me?",
    'output': "<tool_call>\n{\"name\": \"get_current_time\", \"arguments\": {}}\n</tool_call>"
}

multi_tool_example = {
    "input": "USER\nAdd Microsoft (MSFT) to my watchlist and display the updated watchlist.",
    "output": "<tool_call>\n{\"name\": \"add_to_watchlist\", \"arguments\": {\"stock\": \"MSFT\"}}\n</tool_call>\n<tool_call>\n{\"name\": \"get_watchlist\", \"arguments\": {}}\n</tool_call>"
}

multi_turn_example = {
    "input": "USER\nCan you check if I'm currently logged in?\n\nASSISTANT\n<tool_call>\n{\"name\": \"message_get_login_status\", \"arguments\": {}}\n</tool_call>\n\nUSER\n<tool_response>\n{'login_status': True}\n</tool_response>\n\nASSISTANT\nYou are currently logged in.\n\nUSER\nCould you show me all the messages I've sent to other users in my workspace?",
    "output": "<tool_call>\n{\"name\": \"view_messages_sent\", \"arguments\": {}}\n</tool_call>"
}

SCENARIO_VARIATIONS = [
    {
        "focus": "complexity variation", 
        "instruction": "Create a sample with different complexity level - if original is simple, make it more complex with multiple steps",
        "parameters": "Include more detailed parameters and edge cases"
    },
    {
        "focus": "user persona",
        "instruction": "Create a sample from a different user perspective (e.g., business user vs. technical user, beginner vs. expert)",
        "parameters": "Adjust language style and request specificity accordingly"
    },
    {
        "focus": "interaction style",
        "instruction": "Create a sample with different interaction patterns - vary the conversational flow and request style",
        "parameters": "Change how the user phrases requests and provides context"
    },
    {
        "focus": "data variation",
        "instruction": "Create a sample with significantly different data types, ranges, and formats",
        "parameters": "Use different currencies, naming conventions, different parameters, etc."
    }
]


def convert_to_simple_format(input_str):
    """
    将输入格式转换为指定的简单格式
    """
    # 使用正则表达式匹配 <|im_start|>角色...内容...<|im_end|> 的模式
    pattern = r'<\|im_start\|>(user|assistant)\n(.*?)<\|im_end\|>'
    matches = re.findall(pattern, input_str, re.DOTALL)
    
    result = []
    for i, (role, content) in enumerate(matches):
        # 转换角色名
        role_upper = role.upper()
        
        # 处理内容，去除首尾空白
        content = content.strip()
        
        # 添加角色和内容
        if i == len(matches) - 1:  # 最后一个不添加额外换行
            result.append(f"{role_upper}\n{content}")
        else:
            result.append(f"{role_upper}\n{content}\n")
    
    return '\n'.join(result)

def clean_tool_tags_spaces(text):
    """
    清理tool标签周围的多余空格
    """
    if not text:
        return text
    
    # 清理 <tool_call> 标签周围的空格
    # 模式1: <tool_call>  \n 或 <tool_call>   \n 等
    text = re.sub(r'<tool_call>\s*\n', '<tool_call>\n', text)
    
    # 模式2: \n  </tool_call> 或 \n   </tool_call> 等  
    text = re.sub(r'\n\s*</tool_call>', '\n</tool_call>', text)
    
    # 模式3: </tool_call>  \n 或 </tool_call>   \n 等
    text = re.sub(r'</tool_call>\s*\n', '</tool_call>\n', text)
    
    # 清理 <tool_response> 标签周围的空格
    # 模式4: <tool_response>  \n 或 <tool_response>   \n 等
    text = re.sub(r'<tool_response>\s*\n', '<tool_response>\n', text)
    
    # 模式5: \n  </tool_response> 或 \n   </tool_response> 等
    text = re.sub(r'\n\s*</tool_response>', '\n</tool_response>', text)
    
    # 模式6: </tool_response>  \n 或 </tool_response>   \n 等
    text = re.sub(r'</tool_response>\s*\n', '</tool_response>\n', text)
    
    # 清理行尾多余空格（但保留必要的换行符）
    lines = text.split('\n')
    cleaned_lines = [line.rstrip() for line in lines]
    text = '\n'.join(cleaned_lines)
    
    return text

def convert_from_simple_format(simple_str):
    """
    将简单格式转换回输入格式
    """
    # 按双换行符分割不同的角色消息
    parts = simple_str.split('\n\n')
    
    result = []
    for part in parts:
        if not part.strip():
            continue
            
        lines = part.split('\n', 1)  # 只分割第一个换行符
        if len(lines) < 2:
            continue
            
        role_line = lines[0].strip()
        content = lines[1].strip() if len(lines) > 1 else ""

        # 清理content中的tool标签周围的多余空格
        content = clean_tool_tags_spaces(content)
        # 转换角色名
        if role_line == "USER":
            role = "user"
        elif role_line == "ASSISTANT":
            role = "assistant"
        else:
            continue
        
        # 构造输入格式
        result.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    
    return '\n'.join(result) + '\n'

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

def create_multiple_clients(api_keys: List[str], base_url: str) -> List[OpenAI]:
    """创建多个OpenAI客户端"""
    clients = []
    for api_key in api_keys:
        client = OpenAI(base_url=base_url, api_key=api_key)
        clients.append(client)
    return clients

def remove_reasoning_content(model_response):
    """移除思考内容"""
    if "</think>" in model_response:
        parts = model_response.split("</think>")
        cleaned_response = parts[-1].strip("\n")
        return cleaned_response
    else:
        return model_response

def call_generation_api(messages: List[Dict], clients: List[OpenAI], models: list) -> str:
    """调用API生成新样本"""
    try:
        # 随机选择一个客户端
        client = random.choice(clients)
        model = random.choice(models)
        response = client.chat.completions.create(
            model=model,
            temperature=1.0,  # 增加温度以获得更多样化的输出
            max_tokens=8192,
            messages=messages,
            timeout=72000,
            top_p=0.9,
            presence_penalty=1.2,
            extra_body={
                "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": True}
            })

        response_content = response.choices[0].message.content.strip().strip("\n")
        # return remove_reasoning_content(response_content)
        return response_content
    except Exception as e:
        print(f"API_ERROR: {str(e)}")
        return f"API_ERROR: {str(e)}"

def extract_tools_from_instruction(instruction: str) -> str:
    """从instruction中提取工具定义"""
    tools_pattern = r'<tools>\n(.*?)\n</tools>'
    tools_match = re.search(tools_pattern, instruction, re.DOTALL)
    
    if tools_match:
        return tools_match.group(1).strip()
    return ""

def extract_date_from_instruction(instruction: str) -> str:
    """从instruction中提取日期"""
    patterns = [
        r'The current time is (\d{4}-\d{2}-\d{2})',
        r'Today is (\d{4}-\d{2}-\d{2})',
        r'Today is (\d{2}/\d{2}/\d{4})',
        r'Today is (\d{2}-\d{2}-\d{4})',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, instruction)
        if match:
            return match.group(1)
    return "2024-04-14"  # 默认日期

def create_sample_generation_prompt(error_sample: Dict) -> List[Dict]:
    """创建用于生成新样本的prompt"""
    start_token = "<|im_start|>user\n"
    end_token = "<|im_end|>\n"
    # 提取必要信息
    tools_content = extract_tools_from_instruction(error_sample['instruction'])
    current_date = extract_date_from_instruction(error_sample['instruction'])
    error_status = error_sample.get('status', '')
    error_analysis = error_sample.get('error_message', '')
    original_input = start_token + error_sample.get('input', '') + end_token
    converted_input = convert_to_simple_format(original_input)
    
    # 根据错误状态确定正确和错误的响应
    if error_status == 'model_incorrect':
        correct_response = error_sample.get('output', '')
        incorrect_response = error_sample.get('response', '')
    elif error_status == 'original_incorrect':
        correct_response = error_sample.get('response', '')
        incorrect_response = error_sample.get('output', '')
    elif error_status == 'both_incorrect':
        correct_response = ""
        incorrect_response = error_sample.get('response', '')
    else:
        # 默认处理
        # correct_response = error_sample.get('output', '')
        # incorrect_response = error_sample.get('response', '')
        raise NotImplementedError

    variation_index = random.randint(0,6)
    scenario = SCENARIO_VARIATIONS[variation_index % len(SCENARIO_VARIATIONS)]

    system_message = {
        "role": "system",
        "content": f"""You are an expert data engineer specialized in creating high-quality training samples for tool-calling language models. Your mission is to generate NEW training samples that help models learn correct tool usage patterns.

## Context Information
**Current Date:** {current_date}

**Available Tools:**
<tools>
{tools_content}
</tools>

## Your Task
You will analyze an error case and create a completely NEW sample that:
1. **Demonstrates CORRECT tool usage** in a similar scenario
2. **Uses different parameters** (names, amounts, symbols, etc.)
3. **Maintains similar potential error patterns but change the conversation complexity or the dialog flow** 
4. **Follows exact format requirements**

## Critical Requirements
- Generate a realistic but DIFFERENT scenario
- Show the CORRECT approach to handle such requests
- Use proper conversation format with role markers
- Output format MUST be: INPUT: [content] \n OUTPUT: [content]
- INPUT could be a single turn or multiple turn request (When the original sample is single-turn, the INPUT may be multi-turn sample; when the original sample is multi-turn, the INPUT may be a single-turn sample.)
- OUTPUT must use <tool_call>\n...\n</tool_call> format strictly
- NO additional explanations or text outside INPUT/OUTPUT sections"""
    }
    
    user_message = {
        "role": "user", 
        "content": f"""According to the original sample, generate a NEW training sample:

## Original Sample

**Original Input:**
{converted_input}

**Correct Response{' (Expected)' if correct_response else ' (Not Available)'}:**
{correct_response if correct_response else 'No correct response provided - you need to infer the correct approach'}

**Incorrect Response (What went wrong):**
{incorrect_response}

**Error Analysis:**
{error_analysis}

## Generation Instructions

Create a NEW sample that addresses the same type of error but with maximum diversity:
- **Apply the scenario focus**: {scenario['focus']} 
- **Follow parameter strategy**: {scenario['parameters']}
- **Different scenario details**: Change user names, company names, stock symbols, amounts, locations, etc.
- **Different conversation context**: Vary the dialogue flow and user requests
- **Correct solution**: Show the RIGHT way to handle such requests

## Format Requirements

Your response must ONLY contain:

INPUT: [New conversation with proper role markers like USER, ASSISTANT] OUTPUT: [Correct tool calls in <tool_call>\n...\n</tool_call> format]

## Three Output Examples:
**Single Tool**
INPUT: {single_tool_example['input']}
OUTPUT: {single_tool_example['output']}

**Multi Tools**
INPUT: {multi_tool_example['input']}
OUTPUT: {multi_tool_example['output']}

**Multi Turn**
INPUT: {multi_turn_example['input']}
OUTPUT: {multi_turn_example['output']}


## Important Notes
1. The new sample should help the model recognize the error pattern from the original case and learn the correct approach for similar scenarios
2. The correct invocation of new samples does not need to use exactly the same tools as the original samples, and can further involve more complex user requests.
3. The OUTPUT Should cover the error causes and ensure the correctness of function calls.
4. The output format MUST strictly match the output format of the examples.
5. **MAXIMIZE DIVERSITY**: Your sample should be as different as possible from other potential generations while maintaining the core learning objective.

**Generate the new sample now (INPUT and OUTPUT sections only):**"""
    }
    
    return [system_message, user_message]


def parse_generated_sample(api_response: str) -> Tuple[Optional[str], Optional[str]]:
    """解析API生成的样本，提取INPUT和OUTPUT部分"""
    try:
        # 查找INPUT和OUTPUT部分
        input_match = re.search(r'INPUT:\s*(.*?)(?=OUTPUT:|$)', api_response, re.DOTALL)
        output_match = re.search(r'OUTPUT:\s*(.*?)$', api_response, re.DOTALL)
        
        input_content = input_match.group(1).strip() if input_match else None
        output_content = output_match.group(1).strip() if output_match else None
        
        return input_content, output_content
    except Exception as e:
        print(f"Parse error: {str(e)}")
        return None, None

def validate_generated_sample(input_content: str, output_content: str, original_tools: str) -> bool:
    """验证生成的样本是否合理"""
    if not input_content or not output_content:
        return False
    
    # 检查INPUT是否包含对话格式
    # if "<|im_start|>" not in input_content or "<|im_end|>" not in input_content:
    #     return False
        
    # 检查OUTPUT是否包含tool_call格式
    if "<tool_call>" not in output_content:
        return False
        
    # 可以添加更多验证逻辑
    return True

def generate_single_sample(index: int, error_sample: dict, clients: List[OpenAI], models: list, 
                         progress_lock: threading.Lock, processed_count: list, max_retries: int = 3) -> Tuple[int, Dict[str, Any]]:
    """生成单个新样本"""
    
    result = {
        'original_index': index,
        'status': 'failed',
        'new_sample': None,
        'error_message': ''
    }
    
    for attempt in range(max_retries):
        try:
            # 创建生成prompt
            messages = create_sample_generation_prompt(error_sample)
            
            # 调用API生成
            api_response = call_generation_api(messages, clients, models)
            if api_response.startswith("API_ERROR"):
                result['error_message'] = api_response
                continue
                
            # 解析生成的样本
            input_content, output_content = parse_generated_sample(api_response)
            if not input_content or not output_content:
                result['error_message'] = f"Failed to parse generated sample (attempt {attempt + 1})"
                continue

            input_content = input_content.strip(" ").strip("\n")
            output_content = output_content.strip(" ").strip("\n")
            output_content = clean_tool_tags_spaces(output_content)
            
            # 验证生成的样本
            tools_content = extract_tools_from_instruction(error_sample['instruction'])
            if not validate_generated_sample(input_content, output_content, tools_content):
                result['error_message'] = f"Generated sample validation failed (attempt {attempt + 1})"
                continue

            try:
                parsed_input_content = convert_from_simple_format(input_content)
                parsed_input_content = remove_chat_tokens(parsed_input_content)
            except:
                result['error_message'] = f"Generated sample validation failed (attempt {attempt + 1})"
                continue
                
            # 构建新样本
            new_sample = {
                'instruction': error_sample['instruction'],  # 保持相同的instruction
                'input': parsed_input_content,
                'output': output_content,
                'original_sample': error_sample
            }
            
            result['status'] = 'success'
            result['new_sample'] = new_sample
            break
            
        except Exception as e:
            result['error_message'] = f"Generation exception (attempt {attempt + 1}): {str(e)}"
            continue
    
    # 更新进度
    with progress_lock:
        processed_count[0] += 1
        if processed_count[0] % 5 == 0:
            print(f"Processed {processed_count[0]} error samples...")
    
    return index, result

def generate_new_samples_from_errors_multi_thread(error_samples_path: str, api_keys: List[str], base_url: str, 
                                   models: list = ['Qwen3-32B'], max_workers: int = 6, 
                                   samples_per_error: int = 1) -> List[Dict[str, Any]]:
    """基于错误样本生成新样本 - 多线程版本"""
    
    # 创建多个客户端
    clients = create_multiple_clients(api_keys, base_url)
    print(f"Created {len(clients)} API clients")
    
    # 加载错误样本
    with open(error_samples_path, 'r', encoding='utf-8') as f:
        all_samples = json.load(f)
    
    # 筛选错误样本
    # error_samples = [sample for sample in all_samples if sample.get('status') in ['incorrect', 'model_incorrect']]
    error_samples = []
    for sample in all_samples:
        status = sample['status']
        consensus_achieved = sample['judge_details'].get('consensus_achieved', False)
        if 'incorrect' in status and consensus_achieved:
            if status == 'model_incorrect' and '<tool_call>' in sample['output']:
                error_samples.append(sample)
            elif status == 'original_incorrect' and '<tool_call>' in sample['response']:
                error_samples.append(sample)
            elif status == 'both_incorrect' and '<tool_call>' in sample['response']:
                error_samples.append(sample)
            
    print(f"Found {len(error_samples)} error samples from {len(all_samples)} total samples")
    
    if not error_samples:
        print("No error samples found!")
        return []
    
    # error_samples = error_samples[:10]
    # 为每个错误样本生成多个新样本
    generation_tasks = []
    for i, error_sample in enumerate(error_samples):
        for j in range(samples_per_error):
            generation_tasks.append((f"{i}_{j}", error_sample))
    
    print(f"Will generate {len(generation_tasks)} new samples")
    
    # 创建线程锁和进度计数器
    progress_lock = threading.Lock()
    processed_count = [0]
    
    # 准备结果列表
    results = [None] * len(generation_tasks)
    
    # 使用线程池处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_index = {
            executor.submit(generate_single_sample, i, task[1], clients, models, progress_lock, processed_count): i 
            for i, task in enumerate(generation_tasks)
        }
        
        # 收集结果
        for future in as_completed(future_to_index):
            try:
                task_index, result = future.result()
                results[future_to_index[future]] = result
            except Exception as exc:
                original_index = future_to_index[future]
                print(f'Generation task {original_index} generated an exception: {exc}')
                results[original_index] = {
                    'original_index': original_index,
                    'status': 'exception',
                    'new_sample': None,
                    'error_message': f'Processing exception: {str(exc)}'
                }
    
    # 统计和整理结果
    successful_samples = []
    status_counts = Counter()
    
    for result in results:
        if result:
            status_counts[result['status']] += 1
            if result['status'] == 'success' and result['new_sample']:
                successful_samples.append(result['new_sample'])
    
    # 打印统计结果
    print("\n" + "="*60)
    print("GENERATION SUMMARY")
    print("="*60)
    print(f"Total generation tasks: {len(generation_tasks)}")
    print(f"Successful generations: {len(successful_samples)}")
    print(f"Success rate: {len(successful_samples)/len(generation_tasks)*100:.1f}%")
    print(f"Used {len(clients)} API clients with {max_workers} threads")
    print("\nStatus distribution:")
    for status, count in status_counts.items():
        percentage = (count / len(generation_tasks)) * 100
        print(f"  {status}: {count} ({percentage:.1f}%)")
    
    return successful_samples


def main():
    # 配置参数
    error_samples_path = "xxx.json"
    
    # API配置
    model_name = "Qwen3-32B"
    api_keys = [""]
    base_url = "xxx"
    
    # 生成参数
    max_workers = min(12, len(api_keys)) 
    samples_per_error = 3  # 每个错误样本生成3个新样本
    
    print(f"Starting new sample generation with {model_name} model...")
    print(f"Using {len(api_keys)} API keys with {max_workers} threads")
    print(f"Will generate {samples_per_error} samples per error sample")
    
    # 生成新样本
    new_samples = generate_new_samples_from_errors_multi_thread(
        error_samples_path, 
        api_keys, 
        base_url, 
        model_name, 
        max_workers,
        samples_per_error
    )
    
    # 保存结果
    if new_samples:
        output_path = "xxx.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(new_samples, f, indent=2, ensure_ascii=False)
        print(f"\n{len(new_samples)} new samples saved to: {output_path}")
    
    else:
        print("No new samples were generated successfully.")

if __name__ == "__main__":
    main()
