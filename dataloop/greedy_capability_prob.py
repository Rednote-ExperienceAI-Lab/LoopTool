import json
import re
from vllm import LLM, SamplingParams
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoConfig
import numpy as np
import math

def extract_think_tool_call(model_response):
    if "</think>" in model_response:
        parts = model_response.split("</think>")
        reasoning_content = parts[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
        cleaned_response = parts[-1].strip("\n")
    else:
        reasoning_content = model_response.split("<think>")[1].strip("\n")
        cleaned_response = ""

    return reasoning_content, cleaned_response

def calculate_difficulty_metrics(output):
    """
    使用vLLM输出中的cumulative_logprob计算困难度指标
    
    Args:
        output: vLLM的CompletionOutput对象
    
    Returns:
        dict: 包含各种困难度指标的字典
    """
    try:
        # 直接使用vLLM提供的cumulative_logprob
        cumulative_logprob = output.cumulative_logprob
        num_tokens = len(output.token_ids)
        
        if cumulative_logprob is None or num_tokens == 0:
            return {
                "avg_log_prob": None,
                "avg_negative_log_likelihood": None,
                "perplexity": None,
                "total_tokens": num_tokens,
                "cumulative_logprob": None,
            }
        
        # 计算平均log概率
        avg_log_prob = cumulative_logprob / num_tokens
        
        # 计算其他指标
        avg_nll = -avg_log_prob  # 负对数似然
        perplexity = math.exp(avg_nll)  # 困惑度
        
        return {
            "avg_log_prob": float(avg_log_prob),
            "avg_negative_log_likelihood": float(avg_nll),
            "perplexity": float(perplexity),
            "total_tokens": num_tokens,
            "cumulative_logprob": float(cumulative_logprob),
        }
        
    except Exception as e:
        print(f"计算困难度指标时出错: {e}")
        return {
            "avg_log_prob": None,
            "avg_negative_log_likelihood": None,
            "perplexity": None,
            "total_tokens": len(output.token_ids) if hasattr(output, 'token_ids') else 0,
            "cumulative_logprob": None,
        }

def get_qwen3_max_length(model_path):
    """获取Qwen3模型的最大长度配置"""
    try:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        
        if hasattr(config, 'max_position_embeddings'):
            max_len = config.max_position_embeddings
            print(f"从config.max_position_embeddings获取: {max_len}")
            return max_len
        elif hasattr(config, 'max_sequence_length'):
            max_len = config.max_sequence_length
            print(f"从config.max_sequence_length获取: {max_len}")
            return max_len
        
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if hasattr(tokenizer, 'model_max_length') and tokenizer.model_max_length < float('inf'):
            max_len = tokenizer.model_max_length
            print(f"从tokenizer.model_max_length获取: {max_len}")
            return max_len
        
        print("使用Qwen3默认长度: 40960")
        return 40960
        
    except Exception as e:
        print(f"获取Qwen3最大长度时出错: {e}")
        return 40960

def truncate_qwen3_prompt_from_left(prompt, tokenizer, max_length, reserve_tokens=1024):
    """专门为Qwen3优化的从左截断函数"""
    available_length = max_length - reserve_tokens
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    
    if len(tokens) <= available_length:
        return prompt
    
    print(f"截断Qwen3 prompt: 原长度 {len(tokens)} tokens -> {available_length} tokens")
    truncated_tokens = tokens[-available_length:]
    
    try:
        truncated_prompt = tokenizer.decode(truncated_tokens, skip_special_tokens=False)
    except:
        truncated_prompt = tokenizer.decode(truncated_tokens, skip_special_tokens=True)

    return truncated_prompt

def process_samples_vllm_batch_multi_gpu(input_file, output_file, model_path, batch_size=64, max_length=8192):
    """使用vLLM多GPU batch推理处理样本，并收集困难度指标"""
    
    num_gpus = torch.cuda.device_count()
    print(f"检测到 {num_gpus} 个GPU")

    print("初始化Qwen3 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    llm = LLM(
        model=model_path,
        tensor_parallel_size=num_gpus,
        max_model_len=max_length,
        gpu_memory_utilization=0.8,
        swap_space=2,
        max_num_batched_tokens=max_length * batch_size,
    )
    
    # 修改SamplingParams以收集logprobs
    sampling_params = SamplingParams(
        temperature=0.001,
        max_tokens=1024,
        stop=["<|im_end|>"],
        logprobs=1,  # 只需要1个就够了，因为我们用cumulative_logprob
        prompt_logprobs=None,  # 不收集prompt的logprobs以节省内存
    )
    
    # 读取数据
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        data = [data]
    
    # 准备所有prompts
    prompts = []
    for sample in data:
        formatted_prompt = "<|im_start|>system\n" + sample.get('instruction') + "<|im_end|>\n" + "<|im_start|>user\n" + \
            sample.get('input') + "<|im_end|>\n" + "<|im_start|>assistant\n"
        
        formatted_prompt = truncate_qwen3_prompt_from_left(formatted_prompt, tokenizer, max_length=max_length, reserve_tokens=1024)
        prompts.append(formatted_prompt)
    
    results = []
    
    # 批处理推理
    for i in tqdm(range(0, len(prompts), batch_size), desc="多GPU Batch推理中"):
        batch_prompts = prompts[i:i+batch_size]
        batch_samples = data[i:i+batch_size]
        
        # 批量生成
        outputs = llm.generate(batch_prompts, sampling_params)
        
        # 处理结果
        for j, output in enumerate(outputs):
            sample = batch_samples[j]
            raw_text = output.outputs[0].text
            reason_content, cleaned_text = extract_think_tool_call(raw_text)
            
            # 计算困难度指标（使用cumulative_logprob）
            difficulty_metrics = calculate_difficulty_metrics(output.outputs[0])
            
            result = {
                "instruction": sample.get("instruction", ""),
                "input": sample.get("input", ""),
                "output": sample.get("output", ""),
                "response": cleaned_text,
                "think": reason_content,
                # 添加困难度指标
                "avg_log_prob": difficulty_metrics["avg_log_prob"],
                "avg_negative_log_likelihood": difficulty_metrics["avg_negative_log_likelihood"],
                "perplexity": difficulty_metrics["perplexity"],
                "response_tokens": difficulty_metrics["total_tokens"],
                "cumulative_logprob": difficulty_metrics["cumulative_logprob"],
            }
            results.append(result)
    
    # 计算整体统计信息
    valid_samples = [r for r in results if r["avg_log_prob"] is not None]
    
    print(f"\n=== 困难度统计信息 ===")
    print(f"总样本数: {len(results)}")
    print(f"有效样本数: {len(valid_samples)}")
    
    if valid_samples:
        avg_log_probs = [r["avg_log_prob"] for r in valid_samples]
        avg_nlls = [r["avg_negative_log_likelihood"] for r in valid_samples]
        perplexities = [r["perplexity"] for r in valid_samples]
        token_counts = [r["response_tokens"] for r in valid_samples]
        
        print(f"平均log概率: {np.mean(avg_log_probs):.4f} (std: {np.std(avg_log_probs):.4f})")
        print(f"平均负对数似然: {np.mean(avg_nlls):.4f} (std: {np.std(avg_nlls):.4f})")
        print(f"平均困惑度: {np.mean(perplexities):.4f} (std: {np.std(perplexities):.4f})")
        print(f"平均响应长度: {np.mean(token_counts):.1f} tokens")
        print(f"Log概率范围: [{min(avg_log_probs):.4f}, {max(avg_log_probs):.4f}]")
        print(f"困惑度范围: [{min(perplexities):.4f}, {max(perplexities):.4f}]")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n完成! 处理了 {len(results)} 个样本")
    print(f"结果保存到: {output_file}")

# 使用方式
if __name__ == "__main__":
    model = "Qwen/Qwen3-8B" # 当前训练模型的checkpoints
    input_file = "xxx.json"
    output_file = "xxx_response.json"
    
    process_samples_vllm_batch_multi_gpu(input_file, output_file, model, batch_size=64, max_length=12288)
