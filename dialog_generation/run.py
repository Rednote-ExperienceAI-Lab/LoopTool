# -*- coding: utf-8 -*-
import time
from tqdm import tqdm
import argparse
from utils import *
from data_structure import MTData, FlexQueueStack
from data_crawl import InteractiveCrawlThread, get_available_tools, get_available_tools_for_toolbench
from data_post_process import post_process
from utils import difficulty_config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--func', type=str, default="", help="data_crawl或data_post_process")
    parser.add_argument('--raw_data_path', type=str, default="", help="爬取的raw data保存路径")
    parser.add_argument('--processed_data_path', type=str, default="",  help="处理后数据保存路径")
    parser.add_argument('--tools_path', type=str, default="", help="使用的工具列表路径")
    parser.add_argument('--used_models', type=str, nargs="+", default="llama-3.3-70b-instruct", help="使用的gpt模型")
    parser.add_argument('--use_func_call', action="store_true", help="助手是否使用function call")
    parser.add_argument('--use_cot', action="store_true", help="助手是否使用规范化cot")
    parser.add_argument('--use_plan', action="store_true", help="是否在对话开始前采用planner规划对话流程")
    parser.add_argument('--need_chat', action="store_true", help="对话中插入闲聊等非工具调用对话")
    parser.add_argument('--crawl_num', type=int, default=100, help="爬取对话数量，0表示不增加新对话数量，仅完成爬取中的")
    parser.add_argument('--thread_num', type=int, default=1, help="爬取线程数量")
    parser.add_argument('--try_num', type=int, default=5, help="爬取失败重试次数")
    parser.add_argument('--print_num', type=int, default=20, help="每爬取多少数据打印一次进程")
    parser.add_argument('--tgt_turn_num_a', type=int, default=4, help="每个对话轮数最小值")
    parser.add_argument('--tgt_turn_num_b', type=int, default=20, help="每个对话轮数最大值")
    parser.add_argument('--tool_num_a', type=int, default=3, help="每个对话工具数最小值")
    parser.add_argument('--tool_num_b', type=int, default=11, help="每个对话工具数最大值")
    parser.add_argument('--group_tool', action="store_true", help="是否候选工具已分组")
    parser.add_argument('--is_en', action="store_true", help="是否构造英文数据集")
    parser.add_argument('--use_time_or_day', type=str, default='day', choices=['time', 'day'])

    parser.add_argument('--processed_data_extreme_split', action="store_true", help="后处理根据数据长度区分文件")
    args = parser.parse_args()

    # args.func = "data_crawl"
    # args.raw_data_path = r"D:\Data\LLM\Plugins\data_gen_interact\raw_data\res_multiturn_bfcl_250123"
    # args.tools_path = r"D:\Data\LLM\Plugins\api_generate\tools_multiturn_0110\multi-turn-revised\APIs-BFCL-withplan"
    # args.used_models = ["gpt-4o-2024-05-13", "gpt-4o-2024-11-20", "gpt-4-turbo-2024-04-09"]  # ["gpt-4-turbo-2024-04-09"]
    # args.use_func_call = False
    # # args.use_cot = True
    # args.use_plan = True
    # # args.need_chat = True
    # args.group_tool = True
    # args.crawl_num = 6000
    # args.thread_num = 50
    # args.tgt_turn_num_a = 16
    # args.tgt_turn_num_b = 40
    # args.is_en = True
    # args.use_time_or_day = "day"
    # # args.tool_num_a = 1
    # # args.tool_num_b = 6
    # args.func = "data_post_process"
    # args.raw_data_path = r"D:\Data\LLM\Plugins\data_gen_interact\raw_data\res_multiturn_bfcl_250123"
    # args.tools_path = r"D:\Data\LLM\Plugins\api_generate\tools_multiturn_0110\multi-turn-revised\APIs-BFCL-withplan"
    # args.group_tool = True
    # args.processed_data_path = r"D:\Data\LLM\Plugins\data_gen_interact\processed_data\250124"

    if args.func == "data_crawl":
        assert args.raw_data_path and args.tools_path
        os.makedirs(args.raw_data_path, exist_ok=True)
        available_tools, tool2tag, tag2tool = get_available_tools_for_toolbench(args.tools_path, args.group_tool)
        all_tags = list(tag2tool.keys())
        all_tag_weights = [np.log(len(tag2tool[tag])+1) if tag not in ['办公', '金融'] else 1.0 for tag in all_tags]
        # print(all_tags)
        print(all_tag_weights)
        # import pdb; pdb.set_trace()
        difficulty_levels = list(difficulty_config.keys())
        difficulty_weights = [difficulty_config[level]['probability'] for level in difficulty_levels]
        # exit()
        generated_datas = []
        for num in tqdm(range(0, args.crawl_num)):
        # 抽样难度级别
            selected_difficulty = random.choices(difficulty_levels, weights=difficulty_weights, k=1)[0]
            difficulty_cfg = difficulty_config[selected_difficulty]
            tag_nums = difficulty_cfg['tag_num']
            selected_tags = random.choices(all_tags, weights=all_tag_weights, k=tag_nums)

            if args.group_tool:
                selected = random.choice(tag2tool[selected_tag])
                selected_tools = selected["apis"]
                possible_plans = selected["plans"]
            # else:
            #     selected_tools = random.choices(available_tools, k=select_num)
            else:
                # 根据难度配置选择工具数量
                tool_num_min, tool_num_max = difficulty_cfg['tool_num_range']
                select_num = random.choice([n for n in range(tool_num_min, tool_num_max + 1)])
                tool_pool = []
                for tag in selected_tags:
                    tool_pool.extend(tag2tool[tag])

                selected_tools = random.choices(tool_pool, k=select_num)

            
            # 根据难度配置设置对话轮数
            if selected_difficulty == 'easy':
                target_turn_num = difficulty_cfg['turn_num']
            else:
                turn_num_min, turn_num_max = difficulty_cfg['turn_num_range']
                target_turn_num = int(random.randint(turn_num_min, turn_num_max) / 2) * 2  # 确保为偶数
            
            if args.use_time_or_day == 'time':
                current_time = generate_random_time(year=random.choice([2020, 2021, 2022, 2023, 2024, 2025, 2026]))
                each_data = MTData(curr_time=current_time, tool_list=selected_tools,
                                tgt_turn_num=target_turn_num)
            else:
                if args.is_en:
                    current_day = generate_random_date_en(year=random.choice([2020, 2021, 2022, 2023, 2024, 2025, 2026]))
                else:
                    current_day = generate_random_date(year=random.choice([2020, 2021, 2022, 2023, 2024, 2025, 2026]))
                each_data = MTData(curr_day=current_day, tool_list=selected_tools,
                                tgt_turn_num=target_turn_num)
            
            each_data.use_cot = args.use_cot
            each_data.use_func_call = args.use_func_call
            each_data.use_plan = args.use_plan
            each_data.need_chat = args.need_chat
            each_data.is_en = args.is_en
            if args.group_tool and args.use_plan and possible_plans:
                each_data.possible_plans = possible_plans
            generated_datas.append(each_data)
        # import pdb; pdb.set_trace()
        print(len(generated_datas))
        # print(generated_datas[0].convert_to_dict())
        '''
        dict_keys(['cid', 'curr_messages', 'curr_time', 'curr_day', 'tools', 'tgt_turn_num', 'use_cot', 'use_func_call', 'possible_wrong_turns', 
        'possible_assistant_answers', 'valid', 'use_plan', 'need_chat', 'plan', 'possible_plans', 'is_en'])
        '''
        # print(generated_datas[1].convert_to_dict())
        # print(generated_datas[2].convert_to_dict())
        # exit()
        pageQueue = FlexQueueStack()  # 任务队列
        crawled = collections.Counter()
        crawled_data = []
        # 筛选已开始爬取但未完成的对话
        for n in range(100): # For debug
            if not os.path.exists(f"{args.raw_data_path}/generated_multi_turn_{n}.jsonl"):
                continue
            crawled_data.extend(load_data_file(f"{args.raw_data_path}/generated_multi_turn_{n}.jsonl"))
        for d in crawled_data:
            if d['valid'] is False:
                for n in range(3):
                    crawled[f"{d['cid']}-{len(d['curr_messages'])}-{n}"] += 1
            elif len(d['possible_assistant_answers']) == 0:
                for n in range(3):
                    crawled[f"{d['cid']}-{len(d['curr_messages']) - 1}-{n}"] += 1
            else:
                for n in range(len(d['possible_assistant_answers'])):
                    crawled[f"{d['cid']}-{len(d['curr_messages'])}-{n}"] += 1
        total_count = 0
        for d in crawled_data:
            if not args.use_func_call and d["use_func_call"]:
                continue
            if len(d['curr_messages']) >= d['tgt_turn_num']:
                continue
            if crawled[f"{d['cid']}-{len(d['curr_messages'])}-{len(d['possible_assistant_answers'])}"] == 0:
                nd = MTData()
                nd.init_from_dict(d)
                pageQueue.push_back(nd)
                total_count += d['tgt_turn_num'] - len(d['curr_messages'])
        for d in generated_datas:  # generated_datas 里面存储每一次生成的对话记录
            if crawled[f"{d.cid}-{len(d.curr_messages)}-{len(d.possible_assistant_answers)}"] == 0 and d.valid:
                pageQueue.push_back(d)
                total_count += d.tgt_turn_num
        print(len(pageQueue.items))
        # exit()
        crawl_threads = []
        crawl_name_list = range(args.thread_num)
        counts = {"try_count": args.try_num, "print_count": args.print_num, "total_count": total_count, "finish_count": 0}
        print(counts)
        # pdb.set_trace()
        for thread_id in crawl_name_list:
            thread = InteractiveCrawlThread(
                thread_id, pageQueue, counts, crawled, f"{args.raw_data_path}/generated_multi_turn",
                error_return_queue=True, used_models=args.used_models
            )  # 启动爬虫线程
            time.sleep(0.5)
            thread.start()  # 启动线程
            crawl_threads.append(thread)
    elif args.func == "data_post_process":
        assert args.raw_data_path and args.processed_data_path and args.tools_path
        meta_prompt = [
            {
                "persona": "good",
                "content": "You generate replies of excellent overall quality."
            },
            {
                "persona": "plugins", 
                "content": "You can call various user-defined tools to solve user problems."
            }
        ]
        os.makedirs(args.processed_data_path, exist_ok=True)
        post_process(
            crawled_files_path=args.raw_data_path,
            tools_path=args.tools_path,
            group_tool=args.group_tool,
            fixed_meta_prompt=meta_prompt,
            out_put_path=args.processed_data_path,
            need_split=args.processed_data_extreme_split
        )
    else:
        raise NotImplementedError
