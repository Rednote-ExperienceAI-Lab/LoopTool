from openai import OpenAI
import os

api_key = os.getenv('OPENAI_API_KEY', "xxx")
base_url = os.getenv('LLM_BASE_URL', "xxx")

client = OpenAI(
        api_key=api_key,  # 你在allin平台上申请的token
        base_url=base_url  # DirectLLM 域名
        )