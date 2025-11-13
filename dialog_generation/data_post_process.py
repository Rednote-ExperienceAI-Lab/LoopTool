# -*- coding: utf-8 -*-
import copy
import collections
from utils import *


def post_process(crawled_files_path, tools_path, group_tool, fixed_meta_prompt, out_put_path, need_split=True):
    # get tools
    curr_tools = []
    for file in get_all_json_files(tools_path):
        curr_tools.extend(load_data_file(file))
    name2tags = {}
    if group_tool:
        for tools in curr_tools:
            for tool in tools["apis"]:
                name2tags[tool['name']] = tools['tag']
    else:
        for tool in curr_tools:
            name2tags[tool['name']] = tool['tag']

    # get crawled data
    crawled_data = []
    for file in get_all_json_files(crawled_files_path):
        crawled_data.extend(load_data_file(file))

    # find final data
    data_current = collections.defaultdict(dict)
    length = {}
    for d in crawled_data:
        if d['cid'] not in data_current:
            data_current[d['cid']] = d
            length[d['cid']] = len(d["curr_messages"])
        elif d['valid'] and len(data_current[d['cid']]["curr_messages"]) <= len(d["curr_messages"]):
            data_current[d['cid']] = d
            length[d['cid']] = len(d["curr_messages"])
    print(len(data_current))
    # print(f"共有{len([d for d in data_current if len(data_current[d]["curr_messages"]) > 10])}个对话轮次长于10次交互")

    data_only_query = []
    data_turn_less_3 = []
    data_turn_less_5 = []
    data_turn_extreme = []
    data_no_usage = []
    data_invalid = []
    for cid in data_current:
        d = data_current[cid]
        if not d['valid']:
            data_invalid.append(d)
            continue
        if len(d["curr_messages"]) == 1:
            data_only_query.append(d)
            continue
        new_d = {
            "meta_prompt": copy.deepcopy(fixed_meta_prompt),
            "data": d["curr_messages"],
            "tools": d["tools"],
            "tag": []
        }
        if "curr_time" in d and d["curr_time"]:
            new_d["time_set"] = d["curr_time"]

        if "curr_day" in d and d["curr_day"]:
            if "is_en" in d and d["is_en"]:
                new_d["meta_prompt"].append({"persona": "free", "content": f"Today is {d['curr_day']}"})
            else:
                new_d["meta_prompt"].append({"persona": "free", "content": f"今天是{d['curr_day']}"})

        while new_d["data"][-1]["role"] != "assistant":
            del new_d["data"][-1]
        if "<cot>" in new_d["data"][1]["content"] and "</cot>" in new_d["data"][1]["content"]:
            new_d["meta_prompt"].append(
                {
                    "persona": "formalized_cot",
                    "content": "Please think through the following 5 steps: 1. Analyze the user's problem; 2. Analyze the tool list; 3. Analyze tool parameter requirements; 4. Extract parameters; 5. Make action decision."
                }
            )
        else:
            for new_t in new_d["data"]:
                if "<cot>" in new_t["content"] and "</cot>" in new_t["content"]:
                    new_t["content"] = new_t["content"].split("</cot>")[-1].strip()
        if 'possible_wrong_turns' in d and len(d['possible_wrong_turns']) > 0:
            new_d["loss_mask"] = [0 if t["role"] == "assistant" else 1 for t in new_d["data"]]
            for t in d['possible_wrong_turns']:
                if t <= len(new_d["loss_mask"]):
                    new_d["loss_mask"][t - 1] = 1
            if sum(new_d['loss_mask']) == len(new_d['loss_mask']):
                data_invalid.append(new_d)
                continue
        no_usage = True
        for t in new_d["data"]:
            if "tool_usage" not in t:
                continue
            no_usage = False
            for tu in t["tool_usage"]:
                try:
                    tags = name2tags[tu['name']]  # 每个工具只对应一个tag(domain)
                    # import pdb; pdb.set_trace()
                    new_d["tag"].append(tags)
                    break
                except KeyError:
                    continue
            break
        # new_d["tag"] = list(set(new_d["tag"]))
        user_num = sum([t["role"] == "user" for t in d["curr_messages"]])
        # import pdb; pdb.set_trace()
        if no_usage:
            data_no_usage.append(new_d)
        elif user_num < 3:
            # print(new_d)
            data_turn_less_3.append(new_d)
        elif user_num < 5:
            data_turn_less_5.append(new_d)
        else:
            new_d["tag"].append("Extreme_Multi_Turn")
            data_turn_extreme.append(new_d)

    if need_split:
        dump_data_file(data_invalid, os.path.join(out_put_path, "data_invalid.json"))
        dump_data_file(data_turn_less_3, os.path.join(out_put_path, "data_turn_less_3.json"))
        dump_data_file(data_turn_less_5, os.path.join(out_put_path, "data_turn_less_5.json"))
        dump_data_file(data_turn_extreme, os.path.join(out_put_path, "data_turn_extreme.json"))
        dump_data_file(data_no_usage, os.path.join(out_put_path, "data_no_tool_usage.json"))
        dump_data_file(data_only_query, os.path.join(out_put_path, "data_only_query.json"))
    else:
        data_turn_extreme.extend(data_turn_less_3)
        data_turn_extreme.extend(data_turn_less_5)
        dump_data_file(data_invalid, os.path.join(out_put_path, "data_invalid.json"))
        dump_data_file(data_turn_extreme, os.path.join(out_put_path, "data_valid_all.json"))
        dump_data_file(data_no_usage, os.path.join(out_put_path, "data_no_tool_usage.json"))
        dump_data_file(data_only_query, os.path.join(out_put_path, "data_only_query.json"))



if __name__ == "__main__":
    meta_prompt = [
        {
            "persona": "good",
            "content": "你是一个智能助手，你为用户生成综合质量很好的回复。"
        },
        {
            "persona": "plugins",
            "content": "你可以调用各种用户自定义的工具来解决用户的问题。"
        }
    ]
    post_process(
        crawled_files_path="raw_data/res_0503",
        tools_path=r"D:\Data\LLM\Plugins\api_generate\tools_all_0408\all",
        fixed_meta_prompt=meta_prompt,
        out_put_path="processed_data/0506")



