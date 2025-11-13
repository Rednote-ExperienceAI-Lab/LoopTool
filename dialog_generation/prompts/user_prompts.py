# -*- coding: utf-8 -*-
user_agent_system_prmpt = '''You are a user agent in a tool-calling dialogue system. Your role is to generate natural, realistic user queries that follow a planned conversation flow and appropriately utilize the available tools. 

1. Only continue one turn, and the continuation must be a user turn in **ENGLISH**.
2. If the last turn in the history is the assistant asking for parameters, the user must provide as many of the required parameters for the tool call as possible in the next turn.
3. If the last turn in the history is the assistant completing a request and returning a result, the user should make a new request. Each new request must strictly follow the planned flow and should be a complete single request. If the request requires multiple steps, it should be fully expressed in one turn.4. The new request should meets the "[requirement]".
4. The user's turn should be written in [style] language, and the dialogue should flow naturally, using references, transitions, and other techniques to make the conversation smoother.
5. The user's turn may probabilistically consider initiating a request with missing parameters, that is, first submitting an incomplete request and then providing detailed parameters after the assistant inquires about the necessary information.
'''


user_require_prompt = f'''请参考例子，根据给定工具定义和对话的历史，续写对话的下一轮内容。要求如下：
1. 只续写一轮，且续写的轮次必须是用户轮次。
2. 历史的最后一轮是助手追问时，续写的用户轮次必须尽量补充完整工具调用所需的参数。
3. 历史的最后一轮是助手完成需求并返回结果时，用户应提出新的需求。需求[requirement]。
4. 续写的用户轮次，请用[style]的语言风格。
5. 续写的用户轮次需求要清晰，必须是助手可以立即调用工具解决的问题，同时助手所需/追问的参数也要立即提供，不能稍后提供。

以下是可以参考的例子:
<tools>
{{"name": "book_flight", "description": "预订机票工具", "parameters": {{"type": "object", "properties": {{"from_city_name": {{"description": "出发城市", "type": "string"}}, "to_city_name": {{"description": "抵达城市", "type": "string"}}, "depart_date": {{"description": "出发日期，格式为YYYY-MM-dd，例如：2024-02-24", "type": "string"}}, "flightNo": {{"description": "航班号", "type": "string"}}, "carbinClass": {{"description": "舱位类别，如：经济舱Y", "type": "string"}}}}, "required": ["from_city_name", "to_city_name", "depart_date"]}}}}
</tools>

<history>
<system>现在的时间是2023-5-24 08:00:00</system>
<system>用户当前所在的城市为深圳。</system>
<user>我6月1号要到北京开会，麻烦帮忙预订机票</user>
<assistant>好的，我将为您预订机票，请稍等。<tool_usage>book_flight|{{"from_city_name": "深圳", "to_city_name": "北京", "depart_date": "2023-06-01"}}</tool_usage></assistant>
<tool>{{"flight_message": "机票预订成功。具体信息为：航班号YX002，航班时间2023-06-01 11:00:00 -- 15:00:00，从深圳宝安机场到北京首都国际机场。"}}</tool>
<assistant>已为您预订2023年6月1日上午11点到下午3点，从深圳宝安机场到北京首都国际机场的航班，航班号为YX002。</assistant>
</history>

<continuation>
<user>再订一个回来的吧，6月5号回来。</user>
</continuation>

以下是你需要续写的对话：
<tools>
[tool_definition]
</tools>

<history>
[conversation_begin]
</history>

按以下格式续写对话（直接生成数据，不要任何前面和后面的分析或者解释）：
<continuation>
<user>...</user>
</continuation>
'''


user_with_plan_prompt = f'''请参考例子，根据给定工具定义、规划的对话流程和对话的历史，续写对话的下一轮内容。要求如下：
1. 只续写一轮，且续写的轮次必须是用户轮次。
2. 若历史的最后一轮是助手询问参数时，续写的用户轮次必须尽量补充完整工具调用所需的参数。
3. 若历史的最后一轮是助手完整完成需求并返回结果时，用户应提出新的需求。每次续写新的需求应严格符合规划的流程且应是完整的1个需求，如需求多步调用也必须在一个轮次中把需求输出完整。
4. 续写的用户轮次，请用[style]的语言风格，且要流畅地衔接对话历史，可多用指代、过渡等方式让对话更自然。
5. 续写的用户轮次需求要清晰，必须是助手可以立即调用工具解决的问题，同时助手所需/追问的参数也要立即提供，不能稍后提供。
6. 请在续写之前，厘清当前对话处于规划的流程中的哪一步，并输出出来。

以下是可以参考的例子:
<tools>
{{"name": "book_flight", "description": "预订机票工具", "parameters": {{"type": "object", "properties": {{"from_city_name": {{"description": "出发城市", "type": "string"}}, "to_city_name": {{"description": "抵达城市", "type": "string"}}, "depart_date": {{"description": "出发日期，格式为YYYY-MM-dd，例如：2024-02-24", "type": "string"}}, "flightNo": {{"description": "航班号", "type": "string"}}, "carbinClass": {{"description": "舱位类别，如：经济舱Y", "type": "string"}}}}, "required": ["from_city_name", "to_city_name", "depart_date"]}}}}
</tools>

<规划的流程开始>
1. 工具调用需求：用户提出预订机票需求。
2. 工具调用需求：用户追加预订机票需求。
3. 闲聊：用户与助手闲聊，关于出差的问题。
</plans>

<history>
<system>现在的时间是2023-5-24 08:00:00</system>
<system>用户当前所在的城市为深圳。</system>
<user>我6月1号要到北京开会，麻烦帮忙预订机票</user>
<assistant>好的，我将为您预订机票，请稍等。<tool_usage>book_flight|{{"from_city_name": "深圳", "to_city_name": "北京", "depart_date": "2023-06-01"}}</tool_usage></assistant>
<tool>{{"flight_message": "机票预订成功。具体信息为：航班号YX002，航班时间2023-06-01 11:00:00 -- 15:00:00，从深圳宝安机场到北京首都国际机场。"}}</tool>
<assistant>已为您预订2023年6月1日上午11点到下午3点，从深圳宝安机场到北京首都国际机场的航班，航班号为YX002。</assistant>
</history>

输出如下：
<finished plans>
1. 工具调用需求：用户提出预订机票需求。
</finished plans>

<current plan>
2. 工具调用需求：用户追加预订机票需求。
</current plan>

<continuation>
<user>再订一个回来的吧，6月5号回来。</user>
</continuation>

以下是你需要续写的对话：
<tools>
[tool_definition]
</tools>

<规划的流程开始>
[conversation_plan]
</plans>

<history>
[conversation_begin]
</history>

按以下格式续写对话（直接生成数据，不要任何前面和后面的分析或者解释）：
<finished plans>
...
</finished plans>

<current plan>
...
</current plan>

<continuation>
<user>...</user>
</continuation>
'''

user_with_plan_en_prompt = f'''Please refer to the example and, based on the given tool definitions (wrapped with <tools>/</tools>), planned dialogue flow (wrapped with <plans>/</plans>), and the conversation history (wrapped with <history>/</history>), continue the dialogue for the next turn.

Here is an example you can refer to:
<tools>
{{"name": "book_flight", "description": "The tool for booking flights", "parameters": {{"type": "object", "properties": {{"from_city_name": {{"description": "Departure city", "type": "string"}}, "to_city_name": {{"description": "Arrival city", "type": "string"}}, "depart_date": {{"description": "Departure date in YYYY-MM-dd format, e.g., 2024-02-24", "type": "string"}}, "flightNo": {{"description": "Flight number", "type": "string"}}, "carbinClass": {{"description": "Cabin class, e.g., Economy Class Y", "type": "string"}}}}, "required": ["from_city_name", "to_city_name", "depart_date"]}}}}
</tools>

<plans>
1. Single-step request: The user wants to book a ticket.
2. Single-step request: The user wants to add more booking demands.
3. Chitchat: The user chats with the assistant on issues about business trip.
</plans>

<history>
<system>Current time is 2023-5-24 08:00:00</system>
<system>The user is currently in Shenzhen.</system>
<user>I have a meeting in Beijing. Please help me book a flight on June 1st.</user>
<assistant><tool_usage>book_flight|{{"from_city_name": "Shenzhen", "to_city_name": "Beijing", "depart_date": "2023-06-01"}}</tool_usage></assistant>
<tool>{{"flight_message": "Flight booking successful. The details are as follows: Flight number YX002, flight time 2023-06-01 11:00:00 -- 15:00:00, from Shenzhen Bao'an Airport to Beijing Capital International Airport."}}</tool>
<assistant>"Your flight has been booked for June 1, 2023, from 11:00 AM to 3:00 PM, from Shenzhen Bao'an Airport to Beijing Capital International Airport, with flight number YX002.</assistant>
</history>

Output as follows: 
<finished plans>
1. Single-step request:The user wants to book a ticket.
</finished plans>

<current plan>
2. Single-step request:The user wants to add more booking demands.
</current plan>

<continuation>
<user>Please book a return flight for me as well, on June 5th.</user>
</continuation>

Here is the dialogue you need to continue:
<tools>
[tool_definition]
</tools>

<plans>
[conversation_plan]
</plans>

<history>
[conversation_begin]
</history>

Continue the dialogue in the following format (fill in the "..." with **English** directly without any analysis or explanation before or after):
<finished plans>
...
</finished plans>

<current plan>
...
</current plan>

<continuation>
<user>...</user>
</continuation>'''


# requirements = [
#     "要比较复杂，需使用多个工具才能解决",
#     "要求要连续使用同一个工具2次以上才能解决",
#     "要求要连续使用同一个工具4次以上才能解决",
#     "要求要连续使用同一个工具6次以上才能解决",
#     "要求要连续使用同一个工具8次以上才能解决",
#     "要求要使用其中至少2个以上工具才能解决",
#     "要求要使用其中至少3个以上工具才能解决",
#     "要求要使用其中至少4个以上工具才能解决",
#     "要求要连续使用同一个工具3次以上才能解决，且每次调用互不相关，可以并行调用",
#     "要求要连续使用同一个工具5次以上才能解决，且每次调用互不相关，可以并行调用",
#     "要求要连续使用同一个工具7次以上才能解决，且每次调用互不相关，可以并行调用",
#     "要求要连续使用同一个工具9次以上才能解决，且每次调用互不相关，可以并行调用",
#     "要求要使用其中至少2个以上工具才能解决，且每次调用互不相关，可以并行调用",
#     "要求要使用其中至少3个以上工具才能解决，且每次调用互不相关，可以并行调用",
#     "要求要使用其中至少5个以上工具才能解决，且每次调用互不相关，可以并行调用",
#     "要衔接上文，进一步的需求可用一些指代方式给出，让助手能够从历史对话获取信息",
#     "要明确，只需调用其中一个工具解决",
#     "要简单，只需调用其中一个工具一次即可解决",
#     "要简单，只需调用一个工具一次即可解决",
#     "要简单，只需调用其中一个工具一次即可解决，且要一次性把必需参数全部提供",
#     "要简单，只需调用其中一个工具一次即可解决，且不要提供完整参数，让助手追问再提供",
#     "要简单，只需调用一次工具",
#     "要精简，且不用调用工具即可解决",
#     "要精准，只需调用其中一个工具解决",
#     "要具体，只需调用其中一个工具一次即可解决",
#     "要简洁，只需调用一个工具一次即可解决",
#     "要简化，只需调用其中一个工具一次即可解决，且要一次性把必需参数全部提供",
#     "要简朴，只需调用其中一个工具一次即可解决，且不要提供完整参数，让助手追问再提供",
#     "要容易，只需调用一次工具",
#     "要简单，且不用调用工具即可解决",
#     "要至少调用2次工具解决，且有依赖关系，第2次调用的参数需要从第1次调用返回的结果中获取",
#     "要至少调用3次工具解决，且有依赖关系，第2次调用的参数需要从第1次调用返回的结果中获取",
#     "要至少调用4次工具解决，且有依赖关系，第2次调用的参数需要从第1次调用返回的结果中获取",
#     "要至少调用5次工具解决，且有依赖关系，第2次调用的参数需要从第1次调用返回的结果中获取",
#     "要助手一步一步地调用工具，获取工具返回后再进行下一次调用"
# ]

requirements = [
    "Should be relatively complex, requiring multiple tools to solve",
    "Requires using the same tool consecutively more than 2 times to solve",
    "Requires using at least 2 or more tools to solve",
    "Requires using at least 3 or more tools to solve",
    "Requires using at least 4 or more tools to solve",
    "Requires using the same tool consecutively more than 3 times to solve, with each call being independent and able to be called in parallel",
    "Requires using the same tool consecutively more than 5 times to solve, with each call being independent and able to be called in parallel",
    "Requires using the same tool consecutively more than 7 times to solve, with each call being independent and able to be called in parallel",
    "Requires using the same tool consecutively more than 9 times to solve, with each call being independent and able to be called in parallel",
    "Requires using at least 2 or more tools to solve, with each call being independent and able to be called in parallel",
    "Requires using at least 3 or more tools to solve, with each call being independent and able to be called in parallel",
    "Requires using at least 5 or more tools to solve, with each call being independent and able to be called in parallel",
    "Should connect with the previous context, with further requirements given through referential expressions, allowing the assistant to obtain information from conversation history",
    "Should be clear, only need to call one tool to solve",
    "Should be simple, only need to call one tool once to solve",
    "Should be simple, only need to call one tool once to solve",
    "Should be simple, only need to call one tool once to solve, and all necessary parameters should be provided at once",
    "Should be simple, only need to call one tool once to solve, and don't provide complete parameters, let the assistant ask for more information",
    "Should be simple, only need to call a tool once",
    "Should be concise, and can be solved without calling tools",
    "Should be precise, only need to call one tool to solve",
    "Should be specific, only need to call one tool once to solve",
    "Should be concise, only need to call one tool once to solve",
    "Should be simplified, only need to call one tool once to solve, and all necessary parameters should be provided at once",
    "Should be simple, only need to call one tool once to solve, and don't provide complete parameters, let the assistant ask for more information",
    "Should be easy, only need to call a tool once",
    "Should be simple, and can be solved without calling tools",
    "Should require at least 2 tool calls to solve, with dependencies, where the parameters for the 2nd call need to be obtained from the results of the 1st call",
    "Should require at least 3 tool calls to solve, with dependencies, where the parameters for the 2nd call need to be obtained from the results of the 1st call",
    "Should require at least 4 tool calls to solve, with dependencies, where the parameters for the 2nd call need to be obtained from the results of the 1st call",
    "Should require at least 5 tool calls to solve, with dependencies, where the parameters for the 2nd call need to be obtained from the results of the 1st call",
    "Should require the assistant to call tools step by step, obtaining tool results before making the next call"
]

# styles = [
#     "活泼", "优雅", "简约", "严肃", "平静", "自然", "朴实", "幽默", "口语", "随便你自由发挥，合理即可"
# ]
styles = [
    "Lively", "Elegant", "Simple", "Serious", "Calm", "Natural", "Plain", "Humorous", "Colloquial", "Feel free to be creative, as long as it's reasonable"
]



user_chat_prompt = f'''请参考例子，根据给定工具定义和对话的历史，续写对话的下一轮内容。要求如下：
1. 只续写一轮，且续写的轮次必须是用户轮次。
2. 历史的最后一轮是助手追问时，续写的用户轮次必须尽量补充完整工具调用所需的参数。
3. 历史的最后一轮是助手完成需求并返回结果时，用户可与助手进行闲聊，闲聊话题可围绕之前的对话历史。

以下是可以参考的例子:
<tools>
{{"name": "book_flight", "description": "预订机票工具", "parameters": {{"type": "object", "properties": {{"from_city_name": {{"description": "出发城市", "type": "string"}}, "to_city_name": {{"description": "抵达城市", "type": "string"}}, "depart_date": {{"description": "出发日期，格式为YYYY-MM-dd，例如：2024-02-24", "type": "string"}}, "flightNo": {{"description": "航班号", "type": "string"}}, "carbinClass": {{"description": "舱位类别，如：经济舱Y", "type": "string"}}}}, "required": ["from_city_name", "to_city_name", "depart_date"]}}}}
</tools>

<history>
<system>现在的时间是2023-5-24 08:00:00</system>
<user>我6月1号要到北京开会，麻烦帮忙预订机票，我现在在深圳</user>
<assistant>好的，我将为您预订机票，请稍等。<tool_usage>book_flight|{{\"from_city_name\": \"深圳\", \"to_city_name\": \"北京\", \"depart_date\": \"2023-06-01\"}}</tool_usage></assistant>
<tool>{{"flight_message": "机票预订成功。具体信息为：航班号YX002，航班时间2023-06-01 11:00:00 -- 15:00:00，从深圳宝安机场到北京首都国际机场。"}}</tool>
<assistant>已为您预订2023年6月1日上午11点到下午3点，从深圳宝安机场到北京首都国际机场的航班，航班号为YX002。</assistant>
</history>

<continuation>
<user>好的。最近老是出差，睡不好觉</user>
</continuation>

以下是你需要续写的对话：
<tools>
[tool_definition]
</tools>

<history>
[conversation_begin]
</history>

按以下格式续写对话（直接生成数据，不要任何前面和后面的分析或者解释）：
<continuation>
<user>...</user>
</continuation>
'''
