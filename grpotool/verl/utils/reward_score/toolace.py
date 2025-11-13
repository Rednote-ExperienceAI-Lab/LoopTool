# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import json
import os
from collections import Counter
import ast
from typing import List, Dict, Any


def match_score(list1, list2):
    """Compute a similarity score considering element frequency, ignoring order."""
    if list1 == list2:
        return 1.0
    
    if os.getenv("REFINEDREWARD", 0) == "1":
        print("REFINEDREWARD is set to 1, so strict match is used")
        if list1 != list2:
            return 0.0
    
    if not list1 or not list2:
        return 0.0

    count1 = Counter(list1)  # Frequency count for list1
    count2 = Counter(list2)  # Frequency count for list2

    intersection = sum(min(count1[k], count2[k]) for k in count1.keys() & count2.keys())
    max_possible = len(list1) + len(list2) - intersection

    return intersection / max_possible if max_possible > 0 else 0.0
    

# custoimzed reward functions: format
def customize_format_reward_func(completions, answer, step, max_possible_reward, min_possible_reward, **kwargs):
    if str(os.getenv("MAX1STEP30MAX3", 0)) == "1":
        print("MAX1STEP30MAX3 is set to 1, so max 1 -> 30 steps -> max 3")
        if step >= 30:
            max_possible_reward = max_possible_reward / 2
            min_possible_reward = min_possible_reward / 2
        else:
            max_possible_reward = max_possible_reward
            min_possible_reward = min_possible_reward
    
    # schedule reward
    if str(os.getenv("SCHEDULEREWARD", 0)) == "1":  # 奖励区间随步数动态调整
        print("SCHEDULEREWARD is set to 1, so schedule reward is used")
        max_possible_reward = 2 - (2 - max_possible_reward) * step / 150
        min_possible_reward = -2 + (2 + min_possible_reward) * step / 150
        if max_possible_reward < 1.0:
            max_possible_reward = 1.0
        if min_possible_reward > -1.0:
            min_possible_reward = -1.0
    
    rewards = []
    responses = [completion[0]['content'] for completion in completions]
    
    print("\n======= Answer ======= ")
    print(answer[0])
    print("\n======= Responses ======= ")
    for idx, response in enumerate(responses):
        print(f"*** Response {idx+1}***\n{response}")

    for response, ans in zip(responses, answer):
        reward = min_possible_reward
        if "<response>" in ans and "<tool_call>" not in ans:
            pattern = r"^<think>.*?</think>\n<response>.*?</response>$"
            if re.search(pattern, response, re.DOTALL) and response.count("<response>") == 1 and response.count("</response>") == 1:
                reward = max_possible_reward
        elif "<response>" not in ans and "<tool_call>" in ans:
            pattern = r"^<think>.*?</think>\n<tool_call>\n.*?\n</tool_call>$" 
            if re.search(pattern, response, re.DOTALL) and response.count("<tool_call>") == 1 and response.count("</tool_call>") == 1:
                reward = max_possible_reward
        elif "<response>" in ans and "<tool_call>" in ans:
            pattern = r"^<think>.*?</think>\n<tool_call>\n.*?\n</tool_call>\n<response>.*?</response>$"
            if re.search(pattern, response, re.DOTALL) and response.count("<tool_call>") == 1 and response.count("</tool_call>") == 1 and response.count("<response>") == 1 and response.count("</response>") == 1:
                reward = max_possible_reward
        else:
            pattern = r"^<think>.*?</think>$"
            if re.search(pattern, response, re.DOTALL):
                reward = max_possible_reward
        
        rewards.append(reward)
        
    print("\n======= Reward for <format> =======")
    print("Reward function for <format> is called ...")
    print(rewards)
    return rewards


# customized reward functions: length
def customize_length_reward_func(completions, answer, step, max_possible_reward, min_possible_reward, **kwargs):
    # schedule length
    if os.getenv("SCHEDULELENGTH", 0) == "1":
        print("SCHEDULELENGTH is set to 1, so schedule max reward for length is used")
        max_reward_len = (640 - 384) * step / 105 + 384
    else:
        max_reward_len = 512
    
    """Reward function that gives higher scores to longer completions."""
    responses = [completion[0]['content'] for completion in completions]
    rewards = []
    
    for response, ans in zip(responses, answer):
        if "<think>" not in response or "</think>" not in response:
            rewards.append(min_possible_reward)
            continue
        think_responses = response.split("<think>")[-1].split("</think>")[0].strip()
        reward = round(len(think_responses.split()) / max_reward_len, 2)
        if reward > 1.0:
            reward = 1.0
        
        final_reward = reward * (max_possible_reward - min_possible_reward) + min_possible_reward
        rewards.append(final_reward)
    
    print("\n======= Reward for <length> =======")
    print("Reward function for <length> is called ...")
    print(rewards)
    return rewards
                
def compare_parsed_content(parsed1, parsed2):
    """
    比较两个解析后的内容是否一致，忽略列表中元素的顺序以及字典中键的顺序。
    
    参数:
    parsed1 (list of dict): 第一个解析后的内容
    parsed2 (list of dict): 第二个解析后的内容
    
    返回:
    bool: 如果两个解析后的内容一致，返回 True；否则返回 False
    """
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

def compute_tool_call_reward_json(gt_tools, pd_tools, max_possible_reward, min_possible_reward):
    if len(gt_tools) != len(pd_tools):  # 8_03 add
        return min_possible_reward
    if compare_parsed_content(gt_tools, pd_tools):
        print("Exact Match of final parsed tool calls.")
        print("gt_tools", gt_tools)
        print("pd_tools", pd_tools)
        print("Score:", max_possible_reward)
        return max_possible_reward
    else:
        return min_possible_reward

def compute_tool_call_reward_json_v2(gt_tools, pd_tools, max_possible_reward, min_possible_reward):
    tool_calls_len = min(len(gt_tools), len(pd_tools))
    final_reward = min_possible_reward
    single_call_reward = 1.0/len(gt_tools)
    for i in range(tool_calls_len):
        if gt_tools[i] == pd_tools[i]:
            final_reward += single_call_reward
        else:
            return final_reward
            
    return final_reward

def extract_tool_calls(input_string):
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

def customize_correctness_reward_tool_json(completions, answer, step, max_possible_reward, min_possible_reward, **kwargs):
    if str(os.getenv("MAX1STEP30MAX3", 0)) == "1":
        print("MAX1STEP30MAX3 is set to 1, so max 1 -> 30 steps -> max 3")
        if step < 30:
            max_possible_reward = max_possible_reward / 3
            min_possible_reward = min_possible_reward / 3
        else:
            max_possible_reward = max_possible_reward
            min_possible_reward = min_possible_reward
    
    if str(os.getenv("SCHEDULEREWARD", 0)) == "1":
        print("SCHEDULEREWARD is set to 1, so schedule reward is used")
        max_possible_reward = (max_possible_reward - 2) * step / 150 + 2
        min_possible_reward = (min_possible_reward + 2) * step / 150 - 2
        if max_possible_reward > 3.0:
            max_possible_reward = 3.0
        if min_possible_reward < -3.0:
            min_possible_reward = -3.0
    
    responses = [completion[0]['content'] for completion in completions]
    rewards = []
    
    for response, ans in zip(responses, answer):
        reward = 0.0
        
        print(f"The ground truth is {ans}.")
        if "<tool_call>" not in ans:  # 说明是普通回应
            if "<tool_call>" not in response and "</tool_call>" not in response:
                print("The ans and response are pure text")
                if str(os.getenv("RESPONSE_HALF_REWARD", 0)) == "1":  # 表示将Response的奖励变为0.5
                    reward = max_possible_reward * 0.5
                # print(f"The response is {response.split("<response>")[1].split("</response>")[0].strip().strip("\n")}")
                else:
                    reward = max_possible_reward
            else:
                print("The ground truth is pure text, but response is tool call.")
                if str(os.getenv("RESPONSE_HALF_REWARD", 0)) == "1":  # 表示将Response的奖励变为0.5
                    reward = min_possible_reward * 0.5
                else:
                    reward = min_possible_reward
            rewards.append(reward)
            continue
        
        else:
            gt_tools = extract_tool_calls(ans)  # list of dict
            print(f"The ground truth is {json.dumps(gt_tools)}.")
            
            try:
                # Change here as a constrint in training: if the format is not correct, directly give the lowest possible score
                assert "<tool_call>" in response
                assert "</tool_call>" in response
                pd_tools = extract_tool_calls(response)
                print(f"The predicted tool is {json.dumps(pd_tools)}")
                if str(os.getenv("TOOL_REWARD_VERSION", 1)) == "1":
                    reward = compute_tool_call_reward_json(gt_tools, pd_tools, max_possible_reward, min_possible_reward) # top reward is 2
                elif str(os.getenv("TOOL_REWARD_VERSION", 1)) == "2":
                    reward = compute_tool_call_reward_json_v2(gt_tools, pd_tools, max_possible_reward, min_possible_reward)
                else:
                    raise NotImplementedError
            except:
                reward = min_possible_reward
        
        rewards.append(reward)
    
    print("\n======= Reward for <tool call> =======")
    print("Reward function for <tool call> correctness is called ...")
    print(rewards)
    return rewards


def compute_score(solution_str, ground_truth, step=0):
    """The scoring function for ToolACE.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    exp_name = str(os.getenv("EXPERIMENT_NAME", ""))
    if "llama" in exp_name:
        predict_str = solution_str.split("<|start_header_id|>assistant<|end_header_id|>")[-1].split("<|eot_id|>")[0].strip()
    elif "qwen" in exp_name:
        predict_str = solution_str.split("<|im_start|>assistant")[-1].split("<|im_end|>")[0].strip()
    else:
        raise NotImplementedError(f"Unknown model name: {exp_name}")
    if str(os.getenv("CORRECTMAX1", 0)) == "1":
        print("====================")
        print("CORRECTMAX1 is set to 1, so max score is set to 1")
        tool_max_possible = 1.0
        if str(os.getenv("ERRORMAX", -1)) == "0":
            tool_min_possible = 0
        else:
            tool_min_possible = -1.0
        # tool_min_possible =
    else:
        tool_max_possible = 3.0
        tool_min_possible = -3.0
    
    format_max_possible = 1.0
    format_min_possible = 0.0
    
    length_max_possible = 1.0
    length_min_possible = 0.0
    
    completions = [[{"role": "assistant", "content": predict_str}]]
    answer = [ground_truth]
    
    # format_score = customize_format_reward_func(completions, answer, step, format_max_possible, format_min_possible)[0]
    format_score = 0
    correctness_score = customize_correctness_reward_tool_json(completions, answer, step, tool_max_possible, tool_min_possible)[0]
    
    if str(os.getenv("WITHLENGTH", 0)) == "1":
        print("WITHLENGTH is set to 1, so length score is set!")
        length_score = customize_length_reward_func(completions, answer, step, length_max_possible, length_min_possible)[0]
    else:
        length_score = 0
    
    # pdb.set_trace()
    score = format_score + correctness_score + length_score
    
    return score, format_score, correctness_score, length_score
    