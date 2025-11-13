# 将在训练数据上推理错误的样本再收集起来，重新作为下一轮样本的生成, 生成使用的是官方给的脚本，本脚本整理生成的结果，并与原结果进行对比。
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
import time
from datetime import datetime, timedelta
from typing import Callable, List, Optional, Type, Union, Dict, Tuple, Any
from collections import Counter
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

import random
from collections import Counter

class ClientManager:
    """管理多个OpenAI client的类"""
    def __init__(self, base_url: str, api_keys: List[str]):
        self.clients = []
        for api_key in api_keys:
            try:
                client = OpenAI(base_url=base_url, api_key=api_key)
                self.clients.append(client)
                print(f"Successfully created client with API key: {api_key[:10]}...")
            except Exception as e:
                print(f"Failed to create client with API key {api_key[:10]}...: {e}")
        
        if not self.clients:
            raise ValueError("No valid clients were created")
        
        print(f"Total {len(self.clients)} clients created successfully")
    
    def get_random_client(self):
        """随机返回一个client"""
        return random.choice(self.clients)


def compare_parsed_content(parsed1, parsed2):
    """
    比较两个解析后的内容是否一致，忽略列表中元素的顺序以及字典中键的顺序。
    
    参数:
    parsed1 (list of dict): 第一个解析后的内容
    parsed2 (list of dict): 第二个解析后的内容
    
    返回:
    bool: 如果两个解析后的内容一致，返回 True；否则返回 False
    """
    if len(parsed1) != len(parsed2):
        return False
        
    def convert_to_hashable(data):
        """
        将字典转换为可哈希的 frozenset，以便进行比较。
        """
        if isinstance(data, dict):
            return frozenset((key, convert_to_hashable(value)) for key, value in data.items())
        elif isinstance(data, list):
            return frozenset(convert_to_hashable(item) for item in data)
        else:
            return data

    # 将每个字典转换为 frozenset，并对列表进行 Counter 计数
    counter1 = Counter(convert_to_hashable(parsed1))
    counter2 = Counter(convert_to_hashable(parsed2))

    # 比较两个 Counter 是否相等
    return counter1 == counter2

def extract_tools_from_instruction(instruction: str) -> Dict[str, Dict]:
    """
    从instruction中提取工具定义信息
    
    Args:
        instruction: 包含工具定义的指令文本
        
    Returns:
        Dict[str, Dict]: 工具名称到工具定义的映射
    """
    tools_dict = {}
    
    # 提取<tools></tools>标签内的内容
    tools_pattern = r'<tools>\n(.*?)\n</tools>'
    tools_match = re.search(tools_pattern, instruction, re.DOTALL)
    
    if not tools_match:
        return tools_dict
    
    tools_content = tools_match.group(1).strip()
    
    # 按行分割，每行是一个JSON工具定义
    lines = tools_content.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue  
        try:
            tool_def = json.loads(line)
            if 'name' in tool_def:
                tools_dict[tool_def['name']] = tool_def
        except json.JSONDecodeError:
            # 跳过无法解析的行
            continue
    
    return tools_dict

def validate_function_call(call_dict: Dict, tool_set: Dict[str, Dict]) -> Tuple[bool, str]:
    """验证函数调用是否符合规则"""
    try:
        func_name = call_dict['name']
        params = call_dict['arguments']
            # 检查函数是否存在
        if func_name not in tool_set:
            return False, f"Function '{func_name}' not found in tool set"
        
        func_def = tool_set[func_name]
        
        # 获取参数定义
        # if 'parameters' not in func_def['parameters']['properties']:
        #     return False, f"Function '{func_name}' has no the parameter definition of {}"
        
        param_def = func_def['parameters']
        required_params = param_def.get('required', [])
        properties = param_def.get('properties', {})
        
        # 检查必需参数是否存在
        for req_param in required_params:
            if req_param not in params:
                return False, f"Required parameter '{req_param}' missing for function '{func_name}'"
        
        # 检查是否有额外的不合规字段
        for param_name in params:
            if param_name not in properties:
                return False, f"Invalid parameter '{param_name}' for function '{func_name}'"
        
        # 检查参数类型
        for param_name, param_value in params.items():
            if param_name in properties:
                expected_type = properties[param_name].get('type')
                if expected_type:
                    if not validate_parameter_type(param_value, expected_type):
                        return False, f"Parameter '{param_name}' has invalid type for function '{func_name}'. Expected {expected_type}, got {type(param_value).__name__}"
    
        return True, ""
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def validate_parameter_type(value: Any, expected_type: str) -> bool:
    """验证参数类型"""
    type_mapping = {
        'string': str,
        'integer': int,
        'number': (int, float),
        'boolean': bool,
        'array': list,
        'object': dict
    }
    
    expected_python_type = type_mapping.get(expected_type.lower())
    if expected_python_type is None:
        return True  # 未知类型，跳过检查
    
    return isinstance(value, expected_python_type)

def extract_date_from_instruction(prompt):
    pattern = r"The current time is (\d{4}-\d{2}-\d{2})"
    
    match = re.search(pattern, prompt)
    
    if match:
        return match.group(1)
    else:
        return 'Unknown'

def create_llm_judge_prompt(original_row: Dict, output: str, response: str, tool_set: Dict) -> str:
    """创建适配Qwen3模型的判断prompt，返回聊天格式的消息列表"""
    date = extract_date_from_instruction(original_row['instruction'])
    conversation_text = "<|im_start|>user\n" + original_row['input'] + "<|im_end|>\n"
    tool_str = ""
    for tool in tool_set.values():
        tool_str += f"\n{json.dumps(tool)}"

    # 创建系统消息 - 针对Qwen3优化
    system_message = {
        "role": "system",
        "content": f"""You are an expert evaluator specialized in assessing function call responses. Your task is to compare two different function call responses and determine their correctness. Here are the available tools in the conversation:
<tools>{tool_str}\n</tools>

Evaluation Criteria:
1. Correctness: Whether the function calls properly address the user's request
2. Parameter Accuracy: Whether all parameters are correct and appropriate  
3. Function Selection: Whether the chosen functions are suitable for the task
4. Completeness: Whether the response fully satisfies the user's needs.
5. Irrelevance: Whether the response unnecessarily call functions without addressing the user's request.

Please provide objective and thorough evaluations based on these criteria."""
    }
    
    # 创建用户消息 - 针对Qwen3优化格式
    user_message = {
        "role": "user", 
        "content": f"""## Task
Please evaluate two function call responses for the following conversation. The time of the conversation is {date}.

**Original Conversation:**
{conversation_text}

## Responses to Compare

**Response 1**
{output}

**Response 2**
{response}

## Output Format
Strictly respond with the following formats (no additional text):
"[RESPONSE1_INCORRECT/RESPONSE2_INCORRECT/BOTH_CORRECT/BOTH_INCORRECT].\nError Analysis: [Only briefly describe the cause of the incorrect response, without comparing it to the correct response - max 2 sentences] Correct Approach: [Explain the right way to handle this task - max 2 sentences]."

The analysis should be brief and focus on the key differentiating factor."""
    }
    
    return [system_message, user_message]

def remove_reasoning_content(model_response):
    if "</think>" in model_response:
        parts = model_response.split("</think>")
        reasoning_content = parts[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
        cleaned_response = parts[-1].strip("\n")
        return cleaned_response
    else:
        return model_response

def _extract_tool_calls(input_string):
    pattern = r"<tool_call>\n(.*?)\n</tool_call>"
    matches = re.findall(pattern, input_string, re.DOTALL)

    # Process matches into a list of dictionaries
    result = []
    for match in matches:
        try:
            match = json.loads(match)
        except Exception as e:
            pass
        result.append(match)
    return result
        

def call_llm_judge(messages: List[Dict], client, model: str = "Qwen3-32b") -> str:
    """调用Qwen3-8B模型进行判断"""
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0.0,  # 降低温度以获得更稳定的输出
            max_tokens=5000,   # 限制输出长度，因为我们只需要简短判断
            messages=messages,
            timeout=72000,
            top_p=0.8)
            # presence_penalty=1.5,
            # extra_body={
            #     "top_k": 20,
            #     "chat_template_kwargs": {"enable_thinking": True}
            # })
        response_content = response.choices[0].message.content.strip().strip("\n")
        return remove_reasoning_content(response_content)
    except Exception as e:
        print(f"API_ERROR: {str(e)}")
        return f"API_ERROR: {str(e)}"

def call_llm_judge_with_consensus(
    messages: List[Dict], 
    client_manager: ClientManager, 
    models: List[str], 
    max_attempts: int = 3
) -> Tuple[str, bool, Dict]:
    """
    使用多个LLM进行判断，返回多数投票结果
    
    Returns:
        Tuple[str, bool, Dict]: (最终判断结果, 是否达成一致, 详细信息)
    """
    
    results = []
    full_responses = []
    
    # 进行3次判断
    for i in range(len(models)):
        try:
            client = client_manager.get_random_client()
            # model = random.choice(models)
            model = models[i]
            
            result = call_llm_judge(messages, client, model)
            full_responses.append(result)
            
            # 提取核心判断结果
            if 'RESPONSE2_INCORRECT' in result:
                core_result = 'RESPONSE2_INCORRECT'
            elif 'RESPONSE1_INCORRECT' in result:
                core_result = 'RESPONSE1_INCORRECT'  
            elif 'BOTH_CORRECT' in result:
                core_result = 'BOTH_CORRECT'
            elif 'BOTH_INCORRECT' in result:
                core_result = 'BOTH_INCORRECT'
            else:
                core_result = 'UNKNOWN'
            
            results.append(core_result)
            
        except Exception as e:
            results.append('ERROR')
            full_responses.append(f"API_ERROR: {str(e)}")
    
    # 多数投票
    result_counter = Counter(results)
    most_common_result, most_common_count = result_counter.most_common(1)[0]
    
    # 判断是否达成一致（至少2/3同意）
    consensus_achieved = most_common_count >= 2
    
    # 找到对应的完整回复
    final_response = "No valid response"
    for i, core_result in enumerate(results):
        if core_result == most_common_result:
            final_response = full_responses[i]
            break
    
    # 构建详细信息
    consensus_info = {
        'vote_results': results,
        'vote_distribution': dict(result_counter),
        'consensus_achieved': consensus_achieved,
        'final_result': most_common_result,
        'vote_count': f"{most_common_count}/{len(models)}"
    }
    
    return final_response, consensus_achieved, consensus_info
    
    
def extract_error_message(judge_result, result):
    if result['status'] == 'llm_judge_failed':
        return f"Unexpected judge result: {judge_result}"
    else:
        start_pos = judge_result.find("Error Analysis")
        if start_pos != -1:
            return judge_result[start_pos:]
        return judge_result

def evaluate_single_sample_enhanced(
    original_row: dict, 
    client_manager: ClientManager, 
    models: List[str]
) -> Dict[str, Any]:
    """增强版评估单个样本，支持多LLM一致性判断"""

    result = copy.deepcopy(original_row)
    result['status'] = ""
    result['error_message'] = ""
    result['judge_details'] = {}

    try:
        # 获取原始output和模型response
        output = original_row['output']
        response = original_row['response']
        
        if output == response:
            result['status'] = 'both_correct'
            return result
        
        # 检查原始output和response是否均以"<tool_call>"开头
        if not str(output).strip().startswith("<tool_call>") and not str(response).strip().startswith("<tool_call>"):
            result['status'] = 'not_function_call'
            return result
        
        # 尝试解析原始output和模型response
        original_calls = _extract_tool_calls(output)
        model_calls = _extract_tool_calls(response)
        
        # 比较解析内容
        calls_match = compare_parsed_content(original_calls, model_calls)
        
        if calls_match == True:
            result['status'] = 'both_correct'
            return result
        
        # 如果不匹配，调用多LLM判断
        tool_set = extract_tools_from_instruction(original_row['instruction'])
        judge_messages = create_llm_judge_prompt(original_row, output, response, tool_set)
        
        # 使用多LLM一致性判断
        judge_result, consensus_achieved, judge_details = call_llm_judge_with_consensus(
            judge_messages, client_manager, models
        )
        
        # 保存判断详情
        result['judge_details'] = judge_details
        result['consensus_achieved'] = consensus_achieved
        
        # 根据判断结果设置状态
        if 'RESPONSE1_INCORRECT' in judge_result:
            result['status'] = 'original_incorrect'
        elif 'RESPONSE2_INCORRECT' in judge_result:
            result['status'] = 'model_incorrect'
        elif 'BOTH_CORRECT' in judge_result:
            result['status'] = 'both_correct'
        elif 'BOTH_INCORRECT' in judge_result:
            result['status'] = 'both_incorrect'
        else:
            result['status'] = 'llm_judge_failed'

        result['error_message'] = extract_error_message(judge_result, result)
            
    except Exception as e:
        result['status'] = 'evaluation_error'
        result['error_message'] = str(e)
    
    return result

def evaluate_single_sample_with_retry_enhanced(
    original_row: dict, 
    client_manager: ClientManager, 
    models: List[str], 
    max_retries: int = 3
) -> Dict[str, Any]:
    """带重试机制的增强版评估单个样本"""
    for attempt in range(max_retries):
        try:
            return evaluate_single_sample_enhanced(original_row, client_manager, models)
        except Exception as e:
            if attempt == max_retries - 1:
                # 最后一次尝试失败，返回错误结果
                result = copy.deepcopy(original_row)
                result['status'] = 'evaluation_error'
                result['error_message'] = f"Failed after {max_retries} attempts: {str(e)}"
                return result
            else:
                # 等待后重试
                print(f"Attempt {attempt + 1} failed, retrying in {attempt + 1}s...")
                time.sleep(1 * (attempt + 1))  # 递增等待时间
                continue



def comprehensive_evaluation_threaded_enhanced(
    json_path: str, 
    client_manager: ClientManager,
    models: List[str],
    max_workers: int = 4,
    max_retries: int = 3
) -> Dict[str, Any]:
    """增强版多线程综合评估"""
    
    with open(json_path) as f:
        original_data = json.load(f)
    
    total_samples = len(original_data)
    start_time = datetime.now()
    
    print("=" * 60)
    print("STARTING ENHANCED MULTI-THREADED EVALUATION")
    print("=" * 60)
    print(f"Total samples: {total_samples}")
    print(f"Max workers: {max_workers}")
    print(f"Models: {models}")
    print(f"Judge consensus required: 2/3 attempts")
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # 使用线程池执行器
    results = [None] * total_samples
    completed_count = 0
    error_count = 0
    consensus_count = 0
    api_call_count = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_index = {
            executor.submit(evaluate_single_sample_with_retry_enhanced, original_data[i], client_manager, models, max_retries): i
            for i in range(total_samples)
        }
        
        print(f"Submitted {len(future_to_index)} tasks to thread pool")
        print("Processing...")
        
        # 处理完成的任务
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                result = future.result()
                results[index] = result
                
                # 统计错误情况
                if result['status'] in ['evaluation_error', 'thread_error', 'llm_judge_failed']:
                    error_count += 1
                
                # 统计一致性情况
                # if result.get('consensus_achieved', False):
                #     consensus_count += 1
                if 'consensus_achieved' in result:  # 表明该样本进行了API调用来判别
                    api_call_count += 1
                    if result['consensus_achieved']:
                        consensus_count += 1
                    
            except Exception as e:
                # 创建错误结果
                error_result = copy.deepcopy(original_data[index])
                error_result['status'] = 'thread_error'
                error_result['error_message'] = str(e)
                results[index] = error_result
                error_count += 1
                print(f"Thread error on sample {index}: {str(e)}")
            
            completed_count += 1

            # 每完成1000个样本保存一次临时结果
            if completed_count % 3000 == 0:
                # 收集所有已完成的结果
                completed_results = [r for r in results if r is not None]
                tmp_output_path = json_path.replace('.json', f'_tmp_{completed_count}.json')
                try:
                    with open(tmp_output_path, 'w', encoding='utf-8') as f:
                        json.dump(completed_results, f, indent=2, ensure_ascii=False)
                    print(f"✓ Temporary results saved ({len(completed_results)} samples): {tmp_output_path}")
                except Exception as save_error:
                    print(f"✗ Failed to save temporary results: {save_error}")
            
            # 每处理20个样本输出一次进度
            if completed_count % 20 == 0 or completed_count == total_samples:
                elapsed_time = datetime.now() - start_time
                progress_percent = (completed_count / total_samples) * 100
                consensus_rate = (consensus_count / api_call_count) * 100 if api_call_count > 0 else 0
                
                print(f"Progress: {completed_count}/{total_samples} ({progress_percent:.1f}%) | "
                      f"Errors: {error_count} | Consensus: {consensus_count} ({consensus_rate:.1f}%)")          

    # 最终统计
    end_time = datetime.now()
    total_duration = end_time - start_time
    
    status_counts = Counter([r['status'] for r in results])
    
    print("\n" + "="*60)
    print("ENHANCED EVALUATION COMPLETED")
    print("="*60)
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total duration: {str(total_duration).split('.')[0]}")
    print(f"Average time per sample: {total_duration.total_seconds()/total_samples:.2f}s")
    print(f"Total samples processed: {total_samples}")
    print(f"Total errors: {error_count}")
    print(f"Consensus achieved: {consensus_count}/{api_call_count} ({(consensus_count/api_call_count)*100:.1f}%)")
    
    print("\nStatus distribution:")
    for status, count in status_counts.items():
        percentage = (count / total_samples) * 100
        print(f"  {status}: {count} ({percentage:.1f}%)")
    
    evaluation_summary = {
        'total_samples': total_samples,
        'status_distribution': dict(status_counts),
        'execution_time': str(total_duration),
        'errors': error_count,
        'consensus_achieved': consensus_count,
        'consensus_rate': (consensus_count/api_call_count)*100
    }
    
    return results

def main():
    json_path = "xxx.json"

    # 配置多个模型用于判断
    models = ["Qwen3-32B"]  # 可以添加更多模型
    # 配置API信息
    base_url = "xxx"
    api_keys = [""]
    
    # 创建客户端管理器
    try:
        client_manager = ClientManager(base_url, api_keys)
    except ValueError as e:
        print(f"Error creating client manager: {e}")
        return
    
    # 进行增强版综合评估
    print(f"\nStarting enhanced comprehensive evaluation...")
    print(f"Using models: {models}")
    
    evaluation_results = comprehensive_evaluation_threaded_enhanced(
        json_path, 
        client_manager, 
        models,
        max_workers=11,
        max_retries=3
    )
    
    # 保存评估结果
    output_path = "xxx.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
    print(f"\nEnhanced evaluation results saved to: {output_path}")

if __name__ == "__main__":
    main()