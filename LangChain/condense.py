#对话压缩
import os
from typing import List, Tuple
from langchain_core.language_models import BaseLanguageModel
from langchain_huggingface import HuggingFaceEndpoint
from langchain_openai import ChatOpenAI
from config import settings


def get_condenser_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model="qwen-turbo", 
        api_key="",
        temperature=0.0,
        max_tokens=128,  # 足够生成一个句子
        timeout=10,      # 防止卡住
        max_retries=2,
        base_url=settings.base_url,
    )


CONDENSE_PROMPT_TEMPLATE = """根据以下对话历史，将最后一个问题改写成一个独立、完整、无需上下文即可理解的问题。

对话历史：
{chat_history}

最后一个问题：{question}

改写后的问题："""


def condense_question(question: str, chat_history: List[Tuple[str, str]]) -> str:
    """
    将多轮对话中的问题压缩为独立问题。
    chat_history: [("洋务运动是什么？", "洋务运动是...")]
    """
    if not chat_history:
        return question

    # 格式化历史（用户-助手交替）
    history_str = "\n".join([
        f"用户：{q}\n助手：{a}" for q, a in chat_history
    ])
    
    prompt = CONDENSE_PROMPT_TEMPLATE.format(
        chat_history=history_str,
        question=question
    )
    
    llm = get_condenser_llm()
    response = llm.invoke(prompt)
    condensed = response.content.strip() if hasattr(response, 'content') else str(response).strip()
    
    # 检查压缩后的结果是不是空，或者过长（避免跑题）
    if not condensed or len(condensed) > 200:
        return question
    return condensed




