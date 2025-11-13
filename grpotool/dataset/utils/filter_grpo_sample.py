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
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Optional, Type, Union, Dict, Tuple, Any
from collections import Counter
from openai import OpenAI


class ClientManager:
    """管理多个OpenAI客户端"""
    def __init__(self, api_keys: List[str], base_url: str):
        self.clients = []
        for api_key in api_keys:
            client = OpenAI(base_url=base_url, api_key=api_key)
            self.clients.append(client)
        self.lock = threading.Lock()
    
    def get_random_client(self) -> OpenAI:
        """随机获取一个客户端"""
        with self.lock:
            return random.choice(self.clients)

def resolve_ast_by_type(value):
    if isinstance(value, ast.Constant):
        if value.value is Ellipsis:
            output = "..."
        else:
            output = value.value
    elif isinstance(value, ast.UnaryOp):
        output = -value.operand.value
    elif isinstance(value, ast.List):
        output = [resolve_ast_by_type(v) for v in value.elts]
    elif isinstance(value, ast.Dict):
        output = {
            resolve_ast_by_type(k): resolve_ast_by_type(v)
            for k, v in zip(value.keys, value.values)
        }
    elif isinstance(
        value, ast.NameConstant
    ):  # Added this condition to handle boolean values
        output = value.value
    elif isinstance(
        value, ast.BinOp
    ):  # Added this condition to handle function calls as arguments
        output = eval(ast.unparse(value))
    elif isinstance(value, ast.Name):
        output = value.id
    elif isinstance(value, ast.Call):
        if len(value.keywords) == 0:
            output = ast.unparse(value)
        else:
            output = resolve_ast_call(value)
    elif isinstance(value, ast.Tuple):
        output = tuple(resolve_ast_by_type(v) for v in value.elts)
    elif isinstance(value, ast.Lambda):
        output = eval(ast.unparse(value.body[0].value))
    elif isinstance(value, ast.Ellipsis):
        output = "..."
    elif isinstance(value, ast.Subscript):
        try:
            output = ast.unparse(value.body[0].value)
        except:
            output = ast.unparse(value.value) + "[" + ast.unparse(value.slice) + "]"
    else:
        raise Exception(f"Unsupported AST type: {type(value)}")
    return output

def resolve_ast_call(elem):
    # Handle nested attributes for deeply nested module paths
    func_parts = []
    func_part = elem.func
    while isinstance(func_part, ast.Attribute):
        func_parts.append(func_part.attr)
        func_part = func_part.value
    if isinstance(func_part, ast.Name):
        func_parts.append(func_part.id)
    func_name = ".".join(reversed(func_parts))
    args_dict = {}
    for arg in elem.keywords:
        output = resolve_ast_by_type(arg.value)
        args_dict[arg.arg] = output
    return {func_name: args_dict}

def ast_parse(input_str: str, language: str="Python") -> list[dict]:
    try:
        cleaned_input = input_str.strip("[]'")
        parsed = ast.parse(cleaned_input, mode="eval")
        extracted = []
        if isinstance(parsed.body, ast.Call):
            extracted.append(resolve_ast_call(parsed.body))
        else:
            for elem in parsed.body.elts:
                assert isinstance(elem, ast.Call)
                extracted.append(resolve_ast_call(elem))
        return extracted
    except Exception as e:
        # print("解析出现异常")
        return []

def default_decode_execute_prompting(result: str):
    result = result.strip("`\n ")
    if not result.startswith("["):
        result = "[" + result
    if not result.endswith("]"):
        result = result + "]"
    return ast_parse(result)

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

def load_and_show_content(original_path: str, inference_path: str):
    """加载并展示output和response内容"""
    # 加载数据
    original_df = pd.read_parquet(original_path)
    inference_df = pd.read_parquet(inference_path)
    
    print(f"Original dataset shape: {original_df.shape}")
    print(f"Inference dataset shape: {inference_df.shape}")
    print(f"Original columns: {list(original_df.columns)}")
    print(f"Inference columns: {list(inference_df.columns)}")
    
    # 展示几个样本的内容
    n_samples = 10
    
    for i in range(min(n_samples, len(original_df))):
        print(f"\n{'='*60}")
        print(f"SAMPLE {i}")
        print('='*60)

        print("🔹 PROMPT:")
        prompt = original_df['prompt'].iloc[i]
        if isinstance(prompt, list):
            for msg in prompt:
                print(f"  {msg['role'].upper()}: {msg['content']}")
        else:
            print(prompt)
        
        print("🔹 ORIGINAL OUTPUT:")
        print(inference_df['extra_info'].iloc[i].get("output", "None"))
        
        print("\n🔹 INFERENCE RESPONSE:")
        response = inference_df['responses'].iloc[i]

        # 处理numpy array
        if isinstance(response, np.ndarray):
            print(response[0])  # 取array的第一个元素
        elif isinstance(response, list):
            print(response[0])  # 取list的第一个元素
        else:
            print(response)
        print("\n" + "-"*80)

def load_specific_id_content(original_path: str, inference_path: str, indexs: list):
    """加载并展示output和response内容"""
    # 加载数据
    original_df = pd.read_parquet(original_path)
    inference_df = pd.read_parquet(inference_path)
    
    print(f"Original dataset shape: {original_df.shape}")
    print(f"Inference dataset shape: {inference_df.shape}")
    print(f"Original columns: {list(original_df.columns)}")
    print(f"Inference columns: {list(inference_df.columns)}")
    
    for i in indexs:
        print(f"\n{'='*60}")
        print(f"SAMPLE {i}")
        print('='*60)

        print("🔹 PROMPT:")
        prompt = original_df['prompt'].iloc[i]
        if isinstance(prompt, list):
            for msg in prompt:
                print(f"  {msg['role'].upper()}: {msg['content']}")
        else:
            print(prompt)
        
        print("🔹 ORIGINAL OUTPUT:")
        print(inference_df['extra_info'].iloc[i].get("output", "None"))
        
        print("\n🔹 INFERENCE RESPONSE:")
        response = inference_df['responses'].iloc[i]

        # 处理numpy array
        if isinstance(response, np.ndarray):
            print(response[0])  # 取array的第一个元素
        elif isinstance(response, list):
            print(response[0])  # 取list的第一个元素
        else:
            print(response)
        print("\n" + "-"*80)
        

def extract_tools_from_system(system_content: str) -> Dict[str, Dict]:
    """从system字段中提取工具定义"""
    # 查找 "Here is a list of functions in JSON format that you can invoke.\n" 后面的部分
    pattern = r"Here is a list of functions in JSON format that you can invoke\.\s*\n(.+?)(?=\n\n|\Z)"
    match = re.search(pattern, system_content, re.DOTALL)
    
    if not match:
        return {}
    
    functions_text = match.group(1).strip()

    functions = json.loads(functions_text)

    return {func['name']: func for func in functions}  # 函数名为key

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

def create_llm_judge_prompt(original_row: Dict, output: str, tool_set: Dict, date: str) -> str:
    """创建适配Qwen3模型的判断prompt，返回聊天格式的消息列表"""
    
    conversation_text = "<|im_start|>user\n" + original_row['input'] + "<|im_end|>\n"
    tool_str = ""
    for tool in tool_set.values():
        tool_str += f"\n{json.dumps(tool)}"

    # 创建系统消息 - 针对Qwen3优化
    system_message = {
        "role": "system",
        "content": f"""You are a strict evaluator for tool call correctness in dialogues. Please evaluate whether the assistant's tool invocation in this turn is appropriate, based on the provided user query, the definition of the toolset, and the dialogue context. 
Here are the available tools in the conversation:
<tools>{tool_str}\n</tools>

The date is {date}.

Evaluation Criteria:
1. Correctness: Whether the function calls properly address the user's request
2. Parameter Accuracy: Whether all parameters are correct and appropriate  
3. Function Selection: Whether the chosen functions are suitable for the task
4. Completeness: Whether the response fully satisfies the user's needs

Please provide objective and thorough evaluations based on these criteria."""
    }
    
    # 创建用户消息 - 针对Qwen3优化格式
    user_message = {
        "role": "user", 
        "content": f"""## Task
Please evaluate the last function call responses for the following conversation:

**Original Conversation:**
{conversation_text}

**Responses to Evaluate:**
{output}

## Output Format
Strictly respond with the following formats (no additional text):
"[INCORRECT/CORRECT].\nError Analysis: [When the resposne is judged to be INCORRECT, an analysis is provided - max 2 sentences] "
"""
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

def call_llm_judge(messages: List[Dict], client: OpenAI, model: str = "Qwen3-32b") -> str:
    """调用单个模型进行判断"""
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0,  # 降低温度以获得更稳定的输出
            max_tokens=2048,   # 限制输出长度，因为我们只需要简短判断
            messages=messages,
            timeout=72000,
            top_p=0.6,
            presence_penalty=1.5,
            extra_body={
                "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": True}
            })

        response_content = response.choices[0].message.content.strip().strip("\n")
        return remove_reasoning_content(response_content)
    except Exception as e:
        print(f"API_ERROR: {str(e)}")
        return f"API_ERROR: {str(e)}"

def call_llm_judge_with_consensus(
    messages: List[Dict], 
    client_manager: ClientManager, 
    models: List[str]
) -> Tuple[str, bool, Dict]:
    """
    使用多个LLM进行判断，返回多数投票结果
    
    Returns:
        Tuple[str, bool, Dict]: (最终判断结果, 是否达成一致, 详细信息)
    """
    
    results = []
    full_responses = []
    
    # 对每个模型进行判断
    for model in models:
        try:
            client = client_manager.get_random_client()
            
            result = call_llm_judge(messages, client, model)
            full_responses.append(result)
            
            # 提取核心判断结果
            if 'INCORRECT' in result:
                core_result = 'INCORRECT'
            elif 'CORRECT' in result:  
                core_result = 'CORRECT'
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
        'vote_count': f"{most_common_count}/{len(models)}",
        'all_responses': full_responses
    }
    
    return final_response, consensus_achieved, consensus_info
    
def extract_error_message(judge_result, result):
    if result['status'] == 'llm_judge_failed':
        return f"Unexpected judge result: {judge_result}"
    else:
        start_pos = judge_result.find("Error Analysis")
        if start_pos != -1:
            return judge_result[start_pos:]

def extract_date_from_instruction(text):
    patterns = [
        r'Today is (\d{4}-\d{2}-\d{2})', # YYYY-MM-DD
        r'Today is (\d{2}/\d{2}/\d{4})', # MM/DD/YYYY
        r'Today is (\d{2}-\d{2}-\d{4})', # MM-DD-YYYY
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return None

def evaluate_single_sample(index: int, original_row: dict, client_manager: ClientManager, models: List[str], progress_lock: threading.Lock, processed_count: list) -> Tuple[int, Dict[str, Any]]:
    """评估单个样本，返回index和结果 - 使用多模型consensus"""
    
    result = copy.deepcopy(original_row)
    result['status'] = ""
    result['error_message'] = ""
    result['consensus_info'] = {}

    output = original_row['output']
        
    # 检查原始output是否以"<tool_call>"开头
    if not str(output).strip().startswith("<tool_call>"):
        result['status'] = 'not_function_call'
        with progress_lock:
            processed_count[0] += 1
            if processed_count[0] % 10 == 0:
                print(f"Processed {processed_count[0]} samples...")
        return index, result
        
    # 尝试解析原始output
    original_calls = _extract_tool_calls(output)
    
    tool_set = extract_tools_from_instruction(original_row['instruction'])
    date = extract_date_from_instruction(original_row['instruction'])

    # 验证模型调用是否符合规则
    for call in original_calls:
        is_valid, violation_msg = validate_function_call(call, tool_set)

        if not is_valid:
            result['status'] = 'rule_violation'
            result['error_message'] += violation_msg

    if result['status'] == 'rule_violation':
        with progress_lock:
            processed_count[0] += 1
            if processed_count[0] % 10 == 0:
                print(f"Processed {processed_count[0]} samples...")
        return index, result
                
    # 使用多模型consensus进行判断
    judge_messages = create_llm_judge_prompt(original_row, output, tool_set, date)
    judge_result, consensus_achieved, consensus_info = call_llm_judge_with_consensus(
        judge_messages, client_manager, models
    )
    
    # 保存consensus信息
    result['consensus_info'] = consensus_info
    
    # 根据最终判断结果设置状态
    final_result = consensus_info['final_result']
    if final_result == 'INCORRECT':
        result['status'] = 'incorrect'
        result['error_message'] = extract_error_message(judge_result, result)
    elif final_result == 'CORRECT':
        result['status'] = 'correct'
    else:
        result['status'] = 'llm_judge_failed'
        result['error_message'] = f"Unexpected judge result: {final_result}"
    
    # 如果没有达成consensus，标记状态
    if not consensus_achieved:
        result['status'] += '_no_consensus'

    # 更新进度
    with progress_lock:
        processed_count[0] += 1
        if processed_count[0] % 10 == 0:
            print(f"Processed {processed_count[0]} samples...")
    
    return index, result

def comprehensive_evaluation(json_path: str, api_keys: List[str], base_url: str, models: List[str], max_workers: int = 8) -> Dict[str, Any]:
    """对该次推理进行综合评估 - 多模型consensus版本"""
    
    # 创建客户端管理器
    client_manager = ClientManager(api_keys, base_url)
    print(f"Created client manager with {len(client_manager.clients)} API clients")
    print(f"Using models: {models}")
    
    with open(json_path) as f:
        original_data = json.load(f)
    
    # 创建线程锁和进度计数器
    progress_lock = threading.Lock()
    processed_count = [0]  # 使用列表以便在函数间共享
    
    # 准备结果列表
    results = [None] * len(original_data)
    
    # 使用线程池处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_index = {
            executor.submit(evaluate_single_sample, i, original_data[i], client_manager, models, progress_lock, processed_count): i 
            for i in range(len(original_data))
        }
        
        # 收集结果
        for future in as_completed(future_to_index):
            try:
                index, result = future.result()
                results[index] = result
            except Exception as exc:
                original_index = future_to_index[future]
                print(f'Sample {original_index} generated an exception: {exc}')
                # 创建错误结果
                error_result = copy.deepcopy(original_data[original_index])
                error_result['status'] = 'processing_error'
                error_result['error_message'] = f'Processing exception: {str(exc)}'
                error_result['consensus_info'] = {}
                results[original_index] = error_result
    
    # 统计结果
    status_counts = Counter([r['status'] for r in results])
    
    # 统计consensus信息
    consensus_stats = {
        'total_consensus': 0,
        'no_consensus': 0,
        'vote_distribution': Counter()
    }
    
    for result in results:
        if 'consensus_info' in result and result['consensus_info']:
            if result['consensus_info'].get('consensus_achieved', False):
                consensus_stats['total_consensus'] += 1
            else:
                consensus_stats['no_consensus'] += 1
            
            # 统计投票分布
            vote_count = result['consensus_info'].get('vote_count', 'unknown')
            consensus_stats['vote_distribution'][vote_count] += 1
    
    evaluation_summary = {
        'total_samples': len(results),
        'status_distribution': dict(status_counts),
        'consensus_statistics': consensus_stats,
        'models_used': models
    }
    
    # 打印统计结果
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Total samples: {evaluation_summary['total_samples']}")
    print(f"Used {len(client_manager.clients)} API clients with {max_workers} threads")
    print(f"Models used: {', '.join(models)}")
    print("\nStatus distribution:")
    for status, count in status_counts.items():
        percentage = (count / len(results)) * 100
        print(f"  {status}: {count} ({percentage:.1f}%)")
    
    print(f"\nConsensus Statistics:")
    print(f"  Achieved consensus: {consensus_stats['total_consensus']}")
    print(f"  No consensus: {consensus_stats['no_consensus']}")
    print(f"  Vote distribution: {dict(consensus_stats['vote_distribution'])}")
    
    return results

def main():
    json_path = "xxx.json"


    models = ["Qwen3-32B"]  # 根据实际可用模型调整
    # 多个API key
    api_keys = []
    
    base_url = "xxx"
    
    max_workers = min(12, 2*len(api_keys))  # 降低并发数以避免API限流
    
    # 进行综合评估
    print(f"\nStarting comprehensive evaluation with consensus approach...")
    print(f"Using {len(api_keys)} API keys with {max_workers} threads")
    
    evaluation_results = comprehensive_evaluation(
        json_path, 
        api_keys, 
        base_url, 
        models,
        max_workers
    )
    
    # 保存评估结果
    output_path = "xxx.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
    print(f"\nEvaluation results saved to: {output_path}")

if __name__ == "__main__":
    main()
