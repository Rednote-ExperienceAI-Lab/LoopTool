# -*- coding: utf-8 -*-
import time
from tqdm import tqdm
from utils import *
from prompts import user_prompts, assistant_prompts, tool_prompts, planner_prompts
from data_structure import MTData, FlexQueueStack
from api_call import call_gpt



class InteractiveCrawlThread(threading.Thread):
    """
    抓取线程类，注意需要继承线程类Thread
    """

    def __init__(
            self, thread_id, flex_queue, counts, crawled, out_file_name,
            used_models=["gpt-4-turbo-2024-04-09"], error_return_queue=False
    ):
        threading.Thread.__init__(self)  # 需要对父类的构造函数进行初始化
        self.thread_id = thread_id
        self.flex_queue = flex_queue  # 任务队列
        self.crawled = crawled # dict, Counter()
        self.out_file = out_file_name
        self.counts = counts  # total_count & finish_count
        self.print_count = counts['print_count']
        self.try_count = counts['try_count']
        self.used_models = used_models if used_models else ["gpt-4-turbo-2024-04-09"]
        self.error_return_queue = error_return_queue

    def run(self):
        """
        线程在调用过程中就会调用对应的run方法
        :return:
        """
        print('启动线程：', self.thread_id)
        self.crawl_spider()
        print('退出了该线程：', self.thread_id)

    def crawl_spider(self):
        while True:
            if self.flex_queue.is_empty():  # 如果队列为空，则跳出
                break
            else:
                row = self.flex_queue.pop_front()
                # print(row.convert_to_dict())
                # input()
                if self.crawled[f"{row.cid}-{len(row.curr_messages)}-{len(row.possible_assistant_answers)}"]:
                    continue
                messages = row.curr_messages
                use_func_call = row.use_func_call
                if len(messages) == 0 and row.use_plan and row.plan == "":
                    current_role = "planner"
                    if row.is_en:
                        new_messages = row.get_planner_messages(planner_prompts.planner_role_en_system_prompt)
                    else:
                        new_messages = row.get_planner_messages(planner_prompts.planner_role_system_prompt)
                    gpt_tools = None
                elif len(messages) == 0 or (messages[-1]["role"] == "assistant" and "tool_usage" not in messages[-1]): 
                    # 可能的情况是1. 对话开头 2. 助手总结回复之后，应是user轮次, 新的query 3. 助手无法正确调用工具（例如需要额外的参数）
                    current_role = "user"
                    system_en_prompt = user_prompts.user_agent_system_prmpt
                    req = random.choice(user_prompts.requirements)
                    style = random.choice(user_prompts.styles)
                    new_messages = row.convert_messages_for_user_role(system_en_prompt, style=style, requirement=req)
                    gpt_tools = None
                elif messages[-1]["role"] == "user":  # 应是助手轮次, 这里应包含三种情况
                    # 1. 用户给出了新的query, 需要完成
                    # 2. 用户补充了一些调用细节，重新调用
                    # 3. 用户增加了追问，但不涉及工具调用
                    current_role = "assistant"
                    if len(messages) > 3 and "tool_usage" not in messages[-2] and messages[-2]["content"].startswith("<cot>"):  # 回复追问不需要再cot
                        if row.is_en:
                            new_messages, gpt_tools = row.convert_messages_for_assistant_role(
                                assistant_prompts.assistant_role_no_cot_en_system_prompt, use_func_call
                            )
                        else:
                            new_messages, gpt_tools = row.convert_messages_for_assistant_role(
                                assistant_prompts.assistant_role_no_cot_system_prompt, use_func_call
                            )
                    elif row.use_cot:
                        if row.is_en:
                            new_messages, gpt_tools = row.convert_messages_for_assistant_role(
                                assistant_prompts.assistant_role_cot_en_system_prompt, use_func_call
                            )
                        else:
                            new_messages, gpt_tools = row.convert_messages_for_assistant_role(
                                assistant_prompts.assistant_role_cot_system_prompt, use_func_call
                            )
                    else:
                        if row.is_en:
                            new_messages, gpt_tools = row.convert_messages_for_assistant_role(
                                assistant_prompts.assistant_role_no_cot_en_system_prompt, use_func_call
                            )
                        else:
                            new_messages, gpt_tools = row.convert_messages_for_assistant_role(
                                assistant_prompts.assistant_role_no_cot_system_prompt, use_func_call
                            )
                elif messages[-1]["role"] == "tool":  # 应是助手轮次，该轮次为根据工具返回的结果组织答案
                    current_role = "assistant"
                    if row.is_en:
                        new_messages, gpt_tools = row.convert_messages_for_assistant_role(
                            assistant_prompts.assistant_role_no_cot_en_system_prompt, use_func_call
                        )
                    else:
                        new_messages, gpt_tools = row.convert_messages_for_assistant_role(
                            assistant_prompts.assistant_role_no_cot_system_prompt, use_func_call
                        )
                elif messages[-1]["role"] == "assistant" and "tool_usage" in messages[-1]:  # 应是工具轮次
                    current_role = "tool"
                    if row.is_en:
                        new_messages = row.convert_messages_for_tool_role(
                            tool_prompts.tool_role_en_system_prompt
                        )
                    else:
                        new_messages = row.convert_messages_for_tool_role(
                            tool_prompts.tool_role_system_prompt
                        )
                    gpt_tools = None
                else:
                    assert False
                used_model = random.choice(self.used_models)
                # print(new_messages)
                # print(used_model)
                # input()
                if new_messages is None:
                    print("new messages is none:", row['cid'], row['curr_turn'])
                    continue
                success, valid, response = call_gpt(used_model, new_messages, gpt_tools, self.try_count, current_role)
                # import pdb; pdb.set_trace()
                # print(success, valid)
                # print(response)
                # input()
                if not valid:
                    row.valid = False
                if success:
                    response_message = response['choices'][0]['message']
                    if response_message is None or (response_message["content"] is None and not response_message['tool_calls']):
                        self.flex_queue.push_front(row)
                        continue
                    if current_role == "user":
                        new_turn = convert_gpt_user_role_output_to_toolace(response_message)
                    elif current_role == "assistant":
                        new_turn = convert_gpt_assistant_role_output_to_toolace(response_message, used_model=used_model)
                    elif current_role == "tool":
                        new_turn = convert_gpt_tool_role_output_to_toolace(response_message)
                    elif current_role == "planner":
                        new_plan = response_message['content'].strip()
                        row.plan = new_plan
                        self.flex_queue.push_front(row)
                        with open(f"{self.out_file}_{self.thread_id}.jsonl", 'a', encoding="utf-8") as f:
                            f.write(json.dumps(row.convert_to_dict(), ensure_ascii=False) + "\n")
                        continue
                    else:
                        assert False
                    # print(new_turn)
                    # print(row.valid_new_turn(new_turn))
                    # input()
                    if new_turn is None or not row.valid_new_turn(new_turn):
                        print("new turn is None with response:", response_message)
                        row.wrong_turn_count += 1
                        self.flex_queue.push_back(row)  # 响应出错的话，放到最后
                        if row.wrong_turn_count > 10:  # 错误轮次超过10次，放弃该对话的生成
                            self.flex_queue.pop_back()
                        continue
                    self.crawled[f"{row.cid}-{len(row.curr_messages)}-{len(row.possible_assistant_answers)}"] += 1
                    new_row = copy.deepcopy(row)
                    if current_role == "assistant":
                        # 如是助手轮次，需至少爬取两次进行验证，规则：
                        # 1. 两次都是非工具调用，验证通过，随机取其中之一（后续可增强为非随机规则）
                        # 2. 两次都是工具调用，且调用的工具和入参完全相同，验证通过
                        # 3. 两次工具调用不完全相同或一次是工具调用，一次不是，爬取第三次
                        # 4. 三次中，若其中两次都是非工具调用，随机取其中之一验证通过；若存在两次工具调用完全相同，验证通过；否则加loss mask
                        new_row.possible_assistant_answers.append(new_turn)
                        new_success = new_row.check_assistant_answers()
                    else:
                        new_row.curr_messages.append(new_turn)
                        new_success = 1
                    # print(new_row.convert_to_dict())
                    # input()
                    with open(f"{self.out_file}_{self.thread_id}.jsonl", 'a', encoding="utf-8") as f:
                        f.write(json.dumps(new_row.convert_to_dict(), ensure_ascii=False) + "\n")
                    if new_success == 1:
                        self.counts['finish_count'] += 1
                    if new_row.valid and len(new_row.curr_messages) < new_row.tgt_turn_num:
                        self.flex_queue.push_front(new_row)
                    # print(self.counts)
                    # input()
                else:
                    if self.error_return_queue and row.valid:
                        row.wrong_turn_count += 1
                        self.flex_queue.push_back(row)
                        if row.wrong_turn_count > 10:  # 错误轮次超过10次，放弃该对话的生成
                            self.flex_queue.pop_back()

                    elif not row.valid:
                        with open(f"{self.out_file}_{self.thread_id}.jsonl", 'a', encoding="utf-8") as f:
                            f.write(json.dumps(row, ensure_ascii=False) + "\n")

                if self.counts['finish_count'] % self.print_count == 0 or len(self.flex_queue.items) == 0:
                    print(f"Finish {self.counts['finish_count']}/{self.counts['total_count']} Queue Len: {len(self.flex_queue)}")


def get_available_tools(tools_path, group_tool=False):
    curr_tools = []
    # import pdb; pdb.set_trace()
    files = get_all_json_files(tools_path)
    tags = [f"tag_{i}" for i in range(1, len(files)+1)]
    for file, tag in zip(files, tags):
        # curr_tools.extend(load_data_file(file))
        this_file_tools = load_data_file(file)
        for tool in this_file_tools:
            tool['tag'] = tag
            curr_tools.append(tool)
        
    print(len(curr_tools))
    # exit()
    available_tools = []
    tag2tool = collections.defaultdict(list)
    tool2tag = {}
    if not group_tool:
        for tool in curr_tools:
            tag = tool.pop("tag")
            available_tools.append(tool)
            # # for tag in tags:
            #     tag = tag.split("-")[0]
            tag2tool[tag].append(tool)
        print(len(available_tools))
    else:
        for tools in curr_tools:
            available_tools.append(tools)
            tool2tag[json.dumps(tools["apis"], ensure_ascii=False)] = tools["tag"]
            for tag in tools["tag"]:
                tag2tool[tag].append(tools)
    return available_tools, tool2tag, tag2tool

def get_available_tools_for_toolbench(tools_path, group_tool=False):
    curr_tools = []
    files = get_all_json_files(tools_path)
    # import pdb; pdb.set_trace()
    for file in files:
        this_file_tools = load_data_file(file)
        for tool in this_file_tools:
            tool['tag'] = tool['category']
            curr_tools.append(tool)
        
    print(len(curr_tools))
    # exit()
    available_tools = []
    tag2tool = collections.defaultdict(list)
    tool2tag = {}
    if not group_tool:
        for tool in curr_tools:
            tag = tool.pop("tag")
            available_tools.append(tool)
            tag2tool[tag].append(tool)
        print(len(available_tools))
    else:
        for tools in curr_tools:
            available_tools.append(tools)
            tool2tag[json.dumps(tools["apis"], ensure_ascii=False)] = tools["tag"]
            for tag in tools["tag"]:
                tag2tool[tag].append(tools)
    return available_tools, tool2tag, tag2tool


if __name__ == "__main__":

    available_tools, tool2tag, tag2tool = get_available_tools(r"D:\Data\LLM\Plugins\api_generate\tools_all_0408\all")

    all_tags = list(tag2tool.keys())
    all_tag_weights = [np.log(len(tag2tool[tag])) if tag not in ['办公', '金融'] else 1.0 for tag in all_tags]
    print(all_tags)
    print(all_tag_weights)
    # exit()
    generated_datas = []
    for num in tqdm(range(0, 400)):
        time.sleep(0.001)
        selected_tag = random.choices(all_tags, weights=all_tag_weights, k=1)[0]
        select_num = random.choice([3, 4, 5, 6, 7, 8, 9, 10])
        selected_tools = random.choices(tag2tool[selected_tag], k=select_num)
        current_time = generate_random_time(year=random.choice([2020, 2021, 2022, 2023, 2024, 2025, 2026]))
        each_data = MTData(curr_time=current_time, tool_list=selected_tools, tgt_turn_num=int(random.randint(14, 40)/2)*2)
        each_data.use_cot = False  # random.choices([True, False], weights=[4, 1])[0]
        each_data.use_func_call = True
        each_data.use_plan = True
        generated_datas.append(each_data)

    print(len(generated_datas))
    print(generated_datas[0].convert_to_dict())
    print(generated_datas[1].convert_to_dict())
    print(generated_datas[2].convert_to_dict())
    # exit()

    pageQueue = FlexQueueStack()  # 任务队列
    crawled = collections.Counter()
    crawled_data = []
    for n in range(50):
        if not os.path.exists(f"raw_data/res_0503/generated_multi_turn_0503_{n}.jsonl"):
            continue
        crawled_data.extend(load_data_file(f"raw_data/res_0503/generated_multi_turn_0503_{n}.jsonl"))
    for d in crawled_data:
        if d['valid'] is False:
            for n in range(3):
                crawled[f"{d['cid']}-{len(d['curr_messages'])}-{n}"] += 1
        elif len(d['possible_assistant_answers']) == 0:
            for n in range(3):
                crawled[f"{d['cid']}-{len(d['curr_messages'])-1}-{n}"] += 1
        else:
            for n in range(len(d['possible_assistant_answers'])):
                crawled[f"{d['cid']}-{len(d['curr_messages'])}-{n}"] += 1
    total_count = 0
    for d in crawled_data:
        if len(d['curr_messages']) >= d['tgt_turn_num']:
            continue
        if crawled[f"{d['cid']}-{len(d['curr_messages'])}-{len(d['possible_assistant_answers'])}"] == 0:
            nd = MTData()
            nd.init_from_dict(d)
            pageQueue.push_back(nd)
            total_count += d['tgt_turn_num'] -  len(d['curr_messages'])
    for d in generated_datas:
        if crawled[f"{d.cid}-{len(d.curr_messages)}-{len(d.possible_assistant_answers)}"] == 0 and d.valid:
            pageQueue.push_back(d)
            total_count += d.tgt_turn_num
    print(len(pageQueue.items))
    # exit()

    crawl_threads = []
    crawl_name_list = range(50)
    counts = {"try_count": 5, "print_count": 20, "total_count": total_count, "finish_count": 0}
    print(counts)
    for thread_id in crawl_name_list:
        thread = InteractiveCrawlThread(
            thread_id, pageQueue, counts, crawled, "raw_data/res_0503/generated_multi_turn_0503",
            error_return_queue=True, used_models=["gpt-4-turbo-2024-04-09"]
        )  # 启动爬虫线程
        time.sleep(0.5)
        thread.start()  # 启动线程
        crawl_threads.append(thread)

