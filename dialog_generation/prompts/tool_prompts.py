# -*- coding: utf-8 -*-

tool_role_system_prompt = "你的任务是扮演工具，给定工具的定义和相应的工具调用语句，你需要根据你自身知识模拟工具返回结果，然后以特定格式返回给用户。"

tool_role_en_system_prompt = "Your task is to act as the tool. Given the tool definitions and the corresponding tool invocation statements, you need to simulate the tool's return results based on your own knowledge and then return them to the user in a specific format."

tool_mock_prompt = '''以下是工具定义：
[tool_defs]

你接受到的工具调用语句如下：
[tool_calls]

请用以下Json格式返回你模拟工具生成的返回结果，其中每个列表元素为一次调用的返回。注意不要总是返回预期的正向结果，可根据情况模拟返回非预期结果、异常或错误信息等。确保可以用json.loads读取整个结果，除此之外不要任何额外的分析和解释：
[
    {
        "name": ...,
        "results": ...
    },
    {
        "name": ...,
        "results": ...
    }
    ...
]'''

tool_mock_en_prompt = '''Here are the tool definitions:  
[tool_defs]  

The tool invocation statements you received are as follows:  
[tool_calls]  

Please return the simulated tool results in the following JSON format, where each list element represents the result of one invocation. Be careful not to always return the expected positive results; depending on the situation, you can simulate returning unexpected results, exceptions, or error messages, etc. Ensure that the entire result can be read using `json.loads`, and provide no additional analysis or explanation:  
[
    {
        "name": ...,
        "results": ...
    },
    {
        "name": ...,
        "results": ...
    }
    ...
]'''

