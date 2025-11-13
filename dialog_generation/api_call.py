# -*- coding: utf-8 -*-
import json
import time
import random
from gpt_key import clients
# from gpt_key import gpt_client
from utils import *


def call_gpt(used_model, messages, tools=None, try_num=5, current_role='user'):
    success = False
    valid = True
    response = None
    extra_body = {}
    client = random.choice(clients) 
    if "Qwen" in used_model:
        extra_body = {
            "top_k": 20, 
            "chat_template_kwargs": {"enable_thinking": True},
        }
    for attempt in range(try_num):
        # time.sleep(random.randint(20,30))
        time.sleep(random.randint(1, 3))
        try:
            if tools is None:
                # response = gpt_client.chat.completions.create(model=used_model, messages=messages, temperature=0, extra_query={"api-version": "2024-12-01-preview"})
                response = client.chat.completions.create(model=used_model, messages=messages, temperature=0.6, extra_body=extra_body)
            else:
                
                response = client.chat.completions.create(model=used_model, messages=messages, tools=tools, tool_choice="auto", temperature=0.6, extra_body=extra_body)
            # response_message = response['choices'][0]['message']
        except Exception as e:
            print(f"{e}")
            # print(messages)
            if "string does not match pattern" in f"{e}" or "Invalid schema for function" in f"{e}":  # function call出错，数据不可用
                valid = False
                return success, valid, response
            time.sleep(random.randint(1, 20))
        else: # try 执行成功时执行
            success = True
            break
    if success:
        response = json.loads(response.model_dump_json())
    return success, valid, response



