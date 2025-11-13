# -*- coding: utf-8 -*-
import time
from utils import *
from prompts import tool_prompts
from api_call import call_gpt


def convert_messages_for_tool_role(tool_defs, parameters, moc_prompt=None, sys_prompt=None, curr_time=None, curr_day=None):
    if moc_prompt is None:
        moc_prompt = tool_prompts.tool_mock_prompt
    if sys_prompt is None:
        sys_prompt = tool_prompts.tool_role_en_system_prompt
    new_messages = [{"role": "system", "content": sys_prompt}]
    if curr_time:
        new_messages[0]["content"] += f"现在的时间是{curr_time}."
    if curr_day:
        new_messages[0]["content"] += f"Today is {curr_time}."
    tool_call = json.dumps(parameters, ensure_ascii=False)
    prompt = moc_prompt.replace("[tool_defs]", tool_defs).replace("[tool_calls]", tool_call)
    new_messages.append({"role": "user", "content": prompt})
    return new_messages


def moc_tool_call(tool_defs, parameters, moc_prompt=None, sys_prompt=None, curr_time=None, used_model=None, try_num=5):
    """
    tool_defs：工具定义，每一行是一个工具的定义的json.dumps，每个工具定义默认为符合json schema的dict
    parameters：入参，list of dict，每个list元素为一次工具调用
    curr_time：数据的当前时间，如无则为None
    used_model: 使用什么gpt模型，默认为gpt-4-turbo-2024-04-09
    try_num：尝试爬取最多次数，如超过仍无法爬取成功返回None
    """
    messages = convert_messages_for_tool_role(tool_defs, parameters, moc_prompt, sys_prompt, curr_time)
    if used_model is None:
        used_model = "gpt-4-turbo-2024-04-09"
    success, valid, response = call_gpt(used_model, messages, tools=None, try_num=try_num)
    if success and valid:
        response_message = response['choices'][0]['message']
        if response_message is None or response_message["content"] is None:
            return None
        return convert_gpt_tool_role_output_to_toolace(response_message)
    else:
        return None




