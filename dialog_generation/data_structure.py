# -*- coding: utf-8 -*-
import json
import random
import time
from utils import *
from prompts import user_prompts, assistant_prompts, tool_prompts, planner_prompts
from tool_call import convert_messages_for_tool_role


class MTData:
    def __init__(
        self,
        cid=None,
        tool_list=None,
        curr_time=None,
        curr_day=None,
        tgt_turn_num=None,
        use_cot=True,
        use_func_call=False,
        use_plan=False,
        need_chat=False,
        is_en=False
    ):
        self.cid = cid if cid is not None else int(time.time()*100000)
        self.curr_messages = []
        self.curr_time = curr_time
        self.curr_day = curr_day
        assert self.curr_time is None or self.curr_day is None
        self.tools = tool_list
        self.tgt_turn_num = tgt_turn_num
        self.use_cot = use_cot
        self.use_func_call = use_func_call
        self.possible_wrong_turns = []
        self.possible_assistant_answers = []
        self.valid = True
        self.use_plan = use_plan
        self.need_chat = need_chat
        self.plan = ""
        self.possible_plans = []
        self.is_en = is_en
        self.wrong_turn_count = 0

    def get_tool_desc_text(self):
        return "\n".join([json.dumps(t, ensure_ascii=False) for t in self.tools])

    def get_tool_names(self):
        return [t["name"] for t in self.tools]

    def get_planner_messages(self, system_prompt):
        tool_desc = self.get_tool_desc_text()
        if self.need_chat:   # 这里包含一些非工具调用的对话, 先不使用
            if self.possible_plans:
                chosen_plan_example = random.choice(self.possible_plans)
                plan_example_turn_num = len(chosen_plan_example.split("\n"))
                if self.is_en:
                    raise NotImplementedError
                else:
                    used_prompt = planner_prompts.planner_prompt_customized_example.replace(
                        "[tool_defs]", tool_desc).replace("[tgt_turn_num]", str(int(self.tgt_turn_num / 4))).replace(
                        "[example_plan]", chosen_plan_example).replace("[example_turn_num]", str(plan_example_turn_num))
            else:
                if self.is_en:
                    raise NotImplementedError
                used_prompt = planner_prompts.planner_prompt.replace(
                    "[tool_defs]", tool_desc).replace("[tgt_turn_num]", str(int(self.tgt_turn_num/4)))
        else:
            if self.possible_plans: 
                chosen_plan_example = random.choice(self.possible_plans)
                plan_example_turn_num = len(chosen_plan_example.split("\n"))
                if self.is_en:
                    used_prompt = planner_prompts.planner_prompt_en_no_chat_customized_example.replace(
                        "[tool_defs]", tool_desc).replace("[tgt_turn_num]", str(int(self.tgt_turn_num / 4))).replace(
                        "[example_plan]", chosen_plan_example).replace("[example_turn_num]", str(plan_example_turn_num))
                else:
                    used_prompt = planner_prompts.planner_prompt_no_chat_customized_example.replace(
                        "[tool_defs]", tool_desc).replace("[tgt_turn_num]", str(int(self.tgt_turn_num / 4))).replace(
                        "[example_plan]", chosen_plan_example).replace("[example_turn_num]", str(plan_example_turn_num))
            else:
                if self.is_en:
                    used_prompt = planner_prompts.planner_prompt_en_no_chat.replace(
                    "[tool_defs]", tool_desc).replace("[tgt_turn_num]", str(int(self.tgt_turn_num/4)))
                else:
                    used_prompt = planner_prompts.planner_prompt_no_chat.replace(
                    "[tool_defs]", tool_desc).replace("[tgt_turn_num]", str(int(self.tgt_turn_num/4)))
        new_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": used_prompt}
        ]
        return new_messages

    def convert_messages_for_user_role(self, system_prompt, style=None, requirement=None):
        tool_desc = self.get_tool_desc_text()
        conv_begin = convert_toolace_data_to_text(self.curr_messages)
        if self.curr_time:
            if self.is_en:
                conv_begin = f"<system>Current time is {self.curr_time}</system>\n" + conv_begin
            else:
                conv_begin = f"<system>现在的时间是{self.curr_time}</system>\n" + conv_begin
        if self.curr_day:
            if self.is_en:
                conv_begin = f"<system>Today is {self.curr_day}</system>\n" + conv_begin
            else:
                conv_begin = f"<system>今天是{self.curr_day}</system>\n" + conv_begin
        if self.use_plan: 
            if self.is_en:
                used_prompt = user_prompts.user_with_plan_en_prompt  # 在这个setting下，不会使用[requirement]
            else:
                used_prompt = user_prompts.user_with_plan_prompt
        elif len(self.curr_messages) > 5 and self.curr_messages[-2]["role"] == "tool" and random.random() > 0.4:
            # 不调用
            if self.is_en:
                raise NotImplementedError
            used_prompt = user_prompts.user_chat_prompt
        else:
            # 不调用
            if self.is_en:
                raise NotImplementedError
            used_prompt = user_prompts.user_require_prompt
        user_msg = used_prompt.replace("[tool_definition]", tool_desc).replace("[conversation_begin]", conv_begin)
        if "[requirement]" in system_prompt:
            system_prompt = system_prompt.replace("[requirement]", requirement)
        if "[style]" in system_prompt:
            system_prompt = system_prompt.replace("[style]", style)
        if "[conversation_plan]" in user_msg:
            user_msg = user_msg.replace("[conversation_plan]", self.plan)
        new_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg}
        ]
        return new_messages

    def convert_messages_for_assistant_role(self, system_prompt, func_call=True):
        new_messages = [{"role": "system", "content": system_prompt}]
        if self.curr_time:
            if self.is_en:
                new_messages[0]["content"] += f"Current time is {self.curr_time}."
            else:
                new_messages[0]["content"] += f"现在的时间是{self.curr_time}。"
        if self.curr_day:
            if self.is_en:
                new_messages[0]["content"] += f"Today is {self.curr_day}."
            else:
                new_messages[0]["content"] += f"今天是{self.curr_day}。"
        if func_call:
            gpt_tools = convert_toolace_tools_to_gpt_tools(self.tools)
        else:
            gpt_tools = None
            if self.tools:
                if self.is_en:
                    new_messages[0]["content"] += "\n\nYou can use the following tools: \n" + "\n".join([json.dumps(t, ensure_ascii=False) for t in self.tools])
                    new_messages[0]["content"] += "\n\nStrictly Return the tool invocation in the following format: \n<tool_call>ToolName1|{\"parameter1\": \"value1\", \"parameter2\": \"value2\"}</tool_call><tool_call>ToolName2|{\"parameter1\": \"value1\", \"parameter2\": \"value2\"}</tool_call>"
                else:
                    new_messages[0]["content"] += "\n\nYou can use the following tools: \n" + "\n".join([json.dumps(t, ensure_ascii=False) for t in self.tools])
                    new_messages[0]["content"] += "\n\nReturn the tool invocation in the following format: \n<tool_call>工具名1|{\"参数1\": \"值1\", \"参数2\": \"值2\"}</tool_call><tool_call>工具名2|{\"参数1\": \"值1\", \"参数2\": \"值2\"}</tool_call>"

        for i, message in enumerate(self.curr_messages):
            if i == 0:
                assert message["role"] == "user"
            if message["role"] == "user":
                new_messages.append(message)
            elif message["role"] == "assistant":
                assert i != 0
                turn = {"role": "assistant", "content": message["content"]}
                if "tool_usage" in message:
                    if func_call: #不执行
                        turn["tool_calls"] = convert_toolace_tool_usages_to_gpt_tool_calls(message["tool_usage"], i)
                    else:
                        for tool_usage in message["tool_usage"]:
                            t_usage = tool_usage["name"] + "|" + json.dumps(tool_usage["parameters"], ensure_ascii=False)
                            turn["content"] += "<tool_call>" + t_usage + "</tool_call>"
                new_messages.append(turn)
            elif message["role"] == "tool":
                assert 'tool_response' in message and message["content"] == "" and new_messages[-1]["role"] == "assistant"
                if func_call:
                    tool_calls = new_messages[-1]['tool_calls']
                    for j, tool_res in enumerate(message['tool_response']):
                        tool_call_id = tool_calls[j]['id']
                        new_messages.append({
                            'role': 'tool',
                            'name': tool_res['name'],
                            'content': json.dumps(tool_res['results'], ensure_ascii=False),
                            'tool_call_id': tool_call_id
                        })
                else:
                    content = "The response are as follows: \n\n" + json.dumps(message['tool_response'], ensure_ascii=False)
                    new_messages.append({'role': 'user', 'content': content})
            else:
                assert False
        return new_messages, gpt_tools

    def convert_messages_for_tool_role(self, system_prompt):
        assert self.curr_messages[-1]["role"] == "assistant" and "tool_usage" in self.curr_messages[-1]
        tool_defs = self.get_tool_desc_text()
        parameters = json.dumps(self.curr_messages[-1]["tool_usage"], ensure_ascii=False)
        tool_role_prompt = tool_prompts.tool_mock_en_prompt
        return convert_messages_for_tool_role(tool_defs, parameters, tool_role_prompt, system_prompt, self.curr_time, self.curr_day)

    def convert_to_dict(self):
        return {key: value for key, value in vars(self).items() if not key.startswith('__') and not callable(value)}

    def init_from_dict(self, dic):
        for k, v in dic.items():
            self.__dict__[k] = v

    def valid_new_turn(self, new_turn):
        tool_names = self.get_tool_names()
        if new_turn["role"] == "assistant":
            if (len(self.curr_messages)+1) % 2 != 0:
                return False
            if "tool_usage" in new_turn:
                for tu in new_turn["tool_usage"]:
                    if tu['name'] not in tool_names:
                        return False
        elif new_turn["role"] == "tool":
            if len(self.curr_messages) % 2 != 0 or len(self.curr_messages) == 0:
                return False
            if self.curr_messages[-1]["role"] != "assistant" or "tool_usage" not in self.curr_messages[-1]:
                return False
            if len(self.curr_messages[-1]["tool_usage"]) != len(new_turn["tool_response"]):
                return False
            for tu, tr in zip(self.curr_messages[-1]["tool_usage"], new_turn["tool_response"]):
                if tu["name"] != tr["name"]:
                    return False
        return True

    def check_assistant_answers(self):
        if len(self.possible_assistant_answers) < 2:
            return 0
        elif len(self.possible_assistant_answers) == 2:
            if all(["tool_usage" not in ans for ans in self.possible_assistant_answers]):
                self.curr_messages.append(random.choice(self.possible_assistant_answers))
                self.possible_assistant_answers = []
                return 1
            elif not all(["tool_usage" in ans for ans in self.possible_assistant_answers]):
                return 0
            else:
                if self.possible_assistant_answers[0]["tool_usage"] == self.possible_assistant_answers[1]["tool_usage"]:
                    self.curr_messages.append(random.choice(self.possible_assistant_answers))
                    self.possible_assistant_answers = []
                    return 1
                else:
                    return 0
        else: # 三个candidates
            majority = sum(["tool_usage" in ans for ans in self.possible_assistant_answers])
            if majority < 2:  # 两个candidates均无工具调用
                self.curr_messages.append(random.choice([ans for ans in self.possible_assistant_answers if "tool_usage" not in ans]))
                self.possible_assistant_answers = []
                return 1
            else:  # 两个以上candidates有工具调用
                possible_answers = [ans for ans in self.possible_assistant_answers if "tool_usage" in ans]
                if possible_answers[0]["tool_usage"] == possible_answers[1]["tool_usage"]:
                    self.curr_messages.append(possible_answers[0])
                    self.possible_assistant_answers = []
                    return 1
                elif possible_answers[0]["tool_usage"] == possible_answers[-1]["tool_usage"]:
                    self.curr_messages.append(possible_answers[0])
                    self.possible_assistant_answers = []
                    return 1
                elif len(possible_answers) == 3 and possible_answers[1]["tool_usage"] == possible_answers[-1]["tool_usage"]:
                    self.curr_messages.append(possible_answers[-1])
                    self.possible_assistant_answers = []
                    return 1
                else:
                    self.curr_messages.append(random.choice(possible_answers))
                    self.possible_wrong_turns.append(len(self.curr_messages))
                    self.possible_assistant_answers = []
                    return -1


class FlexQueueStack:
    def __init__(self):
        self.items = []

    def __len__(self):
        return len(self.items)

    def push_back(self, item):
        self.items.append(item)

    def push_front(self, item):
        self.items.insert(0, item)

    def pop_back(self):
        if not self.is_empty():
            return self.items.pop()

    def pop_front(self):
        if not self.is_empty():
            return self.items.pop(0)

    def is_empty(self):
        return len(self.items) == 0
