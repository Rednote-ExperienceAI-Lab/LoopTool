# -*- coding: utf-8 -*-

planner_role_system_prompt = "你是一个对话规划助手，给定工具的定义和目标轮次，你需要规划一个围绕可用工具的用户和助手之间的对话流程。"

planner_role_en_system_prompt = "You are a dialogue planning assistant. Given the tool definitions and target turn number, you need to plan a dialogue flow between the user and the assistant, based around the available tools."

planner_prompt = '''以下是工具定义：
[tool_defs]

你的目标轮次：[tgt_turn_num]

规划的对话流程要求如下：
1. 只输出每次交互中用户的需求，且只是大体需求，不用写出需要调用的工具名，可让用户自由扩展。
2. 每个需求必须标明是何种需求，包括工具调用需求、非工具调用需求和闲聊。
3. 不能每个用户需求都是工具调用需求，必须穿插询问助手不需调用工具就可以回答的问题，如文字创作、闲聊、NLP任务等。
4. 相邻的需求之间必须有联系，要确保对话过程的展开是流畅的。如：先提出预定会议室需求，然后再闲聊关于开会的话题。
5. 要包含至少一个需求是比较复杂的，该需求涉及多个工具或者需要调用多次才能解决。
6. 输出的对话流程要符合预期的目标轮次。

以下是一个例子：
目标轮次：5
规划的对话流程：
1. 工具调用需求：用户提出预定会议室需求。
2. 工具调用需求：用户修改预定的会议室。
3. 闲聊：用户与助手闲聊，关于开会太多的话题。
4. 非工具调用需求：用户提供一个开会纪要，让助手进行改写。
5. 工具调用需求：用户希望预订多个会议，并添加相应的日程，然后给会议参与者发邮件提醒。

请直接输出你规划的对话流程，除此之外不要任何额外的分析和解释：
1. ...
2. ...
...'''

planner_prompt_customized_example = '''以下是工具定义：
[tool_defs]

你的目标轮次：[tgt_turn_num]

规划的对话流程要求如下：
1. 只输出每次交互中用户的需求，且只是大体需求，不用写出需要调用的工具名，可让用户自由扩展。
2. 每个需求必须标明是何种需求，包括单步调用需求、多步调用需求、并行调用需求、非工具调用需求和闲聊。
3. 不能每个用户需求都是工具调用需求，必须穿插询问助手不需调用工具就可以回答的问题，如文字创作、闲聊、NLP任务等。
4. 相邻的需求之间必须有联系，要确保对话过程的展开是流畅的。如：先提出预定会议室需求，然后再闲聊关于开会的话题。
5. 要包含至少一个需求是比较复杂的，该需求涉及多个工具或者需要调用多次才能解决。
6. 输出的对话流程要符合预期的目标轮次。

以下是一个例子：
目标轮次：[example_turn_num]
规划的对话流程：
[example_plan]

请直接输出你规划的对话流程，除此之外不要任何额外的分析和解释：
1. ...
2. ...
...'''


planner_prompt_no_chat = '''以下是工具定义：
[tool_defs]

你的目标轮次：[tgt_turn_num]

规划的对话流程要求如下：
1. 只输出每次交互中用户的需求，且只是大体需求，不用写出需要调用的工具名，可让用户自由扩展。
2. 每个需求必须标明是单步调用需求、多步调用需求还是并行调用需求。
3. 要包含至少一个需求是比较复杂的，即至少一个需求是多步或者并行调用需求。
4. 相邻的需求之间必须有联系的，要确保对话过程的展开是流畅的。如：先提出预定会议室需求，然后再修改预定的会议室。
5. 输出的对话流程要符合预期的目标轮次。

以下是一个例子：
目标轮次：3
规划的对话流程：
1. 并行调用需求：用户希望预订多个会议，并添加相应的日程。
2. 单步调用需求：用户修改其中一个预订会议的会议室。
3. 多步调用需求：用户修改相应的日程，并根据日程返回信息发送邮件提醒。

请直接输出你规划的对话流程，除此之外不要任何额外的分析和解释：
1. ...
2. ...
...'''


planner_prompt_no_chat_customized_example = '''以下是工具定义：
[tool_defs]

你的目标轮次：[tgt_turn_num]

规划的对话流程要求如下：
1. 只输出每次交互中用户的需求，且只是大体需求，不用写出需要调用的工具名，可让用户自由扩展。
2. 每个需求必须标明是单步调用需求、多步调用需求还是并行调用需求。
3. 要包含至少一个需求是比较复杂的，即至少一个需求是多步或者并行调用需求。
4. 相邻的需求之间必须有联系的，要确保对话过程的展开是流畅的。如：先提出预定会议室需求，然后再修改预定的会议室。
5. 输出的对话流程要符合预期的目标轮次。

以下是一个例子：
目标轮次：[example_turn_num]
规划的对话流程：
[example_plan]

请直接输出你规划的对话流程，除此之外不要任何额外的分析和解释：
1. ...
2. ...
...'''


# planner_prompt_en_no_chat_customized_example = '''Here is the tool definition:
# [tool_defs]
#
# Your target turn number: [tgt_turn_num]
#
# The required dialogue flow plan is as follows:
# 1. Only output the user's needs in each interaction, and only the general needs, without specifying which tools need to be called. Let the user expand freely.
# 2. Each need must be marked as either a single-step request, multi-step request, or parallel request.
# 3. At least one need must be relatively complex, meaning it must be a multi-step or parallel request.
# 4. There must be a connection between adjacent needs to ensure the dialogue flow is smooth. For example, first propose a meeting room reservation request, then modify the reserved meeting room.
# 5. The output dialogue flow must match the expected target turn number.
#
# Here is an example:
# Target turn number: [example_turn_num]
# Planned dialogue flow:
# [example_plan]
#
# Please directly output the planned dialogue flow without any additional analysis or explanation:
# 1. ...
# 2. ...
# ...'''
planner_prompt_en_no_chat_customized_example = '''Here is the tool definition:  
[tool_defs]  

Your target turn number: [tgt_turn_num]  

The required dialogue flow plan is as follows:  
1. Mock the given example and generate a new plan.
2. Output the user's needs in each interaction, general needs or specific needs are both OK. The user can expand freely.  
3. Each need in your generation must be additionally marked as either a single-step request, multi-step request, or parallel request.  
3. At least one need must be relatively complex, meaning it must be a multi-step or parallel request.  
4. There must be a connection between adjacent needs to ensure the dialogue flow is smooth. For example, first propose a meeting room reservation request, then modify the reserved meeting room.  
5. The output dialogue flow must match the expected target turn number.  

Here is an example:  
Target turn number: [example_turn_num]  
Planned dialogue flow:  
[example_plan]  

Please directly output the planned dialogue flow without any additional analysis or explanation:  
1. ...  
2. ...  
...'''

planner_prompt_en_no_chat = '''Here is the tool definition:  
[tool_defs]  

Your target turn number: [tgt_turn_num]  

The required dialogue flow plan is as follows:  
1. Only output the user's need in each interaction, and only in general terms. Do not specify which tool to call; allow the user to elaborate freely.
2. Each need in your generation must be additionally marked as either a single-step request, multi-step request, or parallel request.  
3. At least one need must be relatively complex, meaning it must be a multi-step or parallel request.  
4. There must be a logical connection between consecutive needs to ensure a smooth and coherent dialogue flow. For example: first, the user requests to book a meeting room, and then asks to modify the booked meeting room. 
5. The output dialogue flow must match the expected target turn number. 
6. Please carefully consider the provided definitions of tools, and creatively integrate as many tool invocations as possible into the user requests you plan.

Here is an example:  
Target turn number: 3  
Planned dialogue flow: 
1. Parallel need: The user wants to book multiple meetings and add related calendar events at the same time.
2. Single-step need: The user modifies the meeting room for one of the booked meetings.
3. Multi-step need: The user updates the relevant calendar event and, based on the calendar information, sends an email reminder.

Please directly output the planned dialogue flow without any additional analysis or explanation:  
1. ...  
2. ...  
...'''


