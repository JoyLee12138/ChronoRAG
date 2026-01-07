import os
import logging
from typing import List, Tuple, Optional
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import AIMessage
from langchain_core.language_models import BaseLanguageModel
from langchain_openai import ChatOpenAI

import httpx
from config import settings
from retriever import chrono_rag_search  
from condense import condense_question   
from prompt import build_chat_rag_prompt

# 可选：开启调试日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_rag_llm() -> ChatOpenAI:
    """创建用于RAG问答的LLM实例"""
    return ChatOpenAI(
        model="qwen-turbo",
        api_key="",
        base_url=settings.base_url,
        temperature=0.7,
        max_tokens=1024,
        timeout=30,
        max_retries=2,
    )


def query_rag_multiturn(
    question: str,
    chat_history: List[Tuple[str, str]] = None,
    llm: Optional[BaseLanguageModel] = None):
    """
    多轮对话 RAG 主入口
    
    Args:
        question (str): 用户当前输入的问题
        chat_history (List[Tuple[str, str]]): 历史对话，格式 [(用户问, 助手答), ...]
        llm (Optional): 可传入自定义 LLM 实例（用于测试）
    
    Returns:
        str: 助手的回答
    """
    chat_history = chat_history or []
    
    try:
        # === Step 1: 压缩问题（关键！）===
        logger.info(f"原始问题: {question}")
        standalone_question = condense_question(question, chat_history)
        logger.info(f"压缩后问题: {standalone_question}")

        # === Step 2: RAG 检索（复用你现有的工具）===
        # 注意：chrono_rag_search 是 LangChain Tool，输入是 dict
        context = chrono_rag_search.invoke({"query": standalone_question})
        logger.info(f"检索到上下文长度: {len(context)} 字符")

        # === Step 3: 构建 Prompt ===
        # 仅保留最近 1～2 轮对话，避免超上下文
        recent_history = ""
        if chat_history:
            # 取最近 1 轮（平衡相关性与长度）
            last_q, last_a = chat_history[-1]
            recent_history = f"用户：{last_q}\n助手：{last_a}"

        prompt = build_chat_rag_prompt(
            question=question,
            context=context,
            chat_history=recent_history
        )
        logger.debug(f"最终 Prompt:\n{prompt}")

        # === Step 4: 调用主 LLM 生成回答 ===
        if llm is None:
            llm = get_rag_llm()
        
        response = llm.invoke(prompt)
        answer = response.content.strip() if hasattr(response, 'content') else str(response).strip()

        # === Step 5: 安全后处理 ===
        if not answer:
            return "抱歉，我暂时无法生成回答。"
        
        return answer

    except Exception as e:
        logger.error(f"query_rag_multiturn 发生错误: {e}", exc_info=True)
        return f"❌ 系统异常：{str(e)}"


# ===== 本地调试入口 =====
if __name__ == "__main__":
    

    # 模拟多轮对话
    history = []

    print("🤖 欢迎使用 ChronoRAG-ZH 历史问答助手！输入 '退出' 结束。\n")
    while True:
        user_input = input("👤 用户: ").strip()
        if not user_input or user_input.lower() in ["退出", "quit", "exit"]:
            break

        answer = query_rag_multiturn(user_input, history)
        print(f"\n🤖 助手: {answer}\n")

        # 保存原始问答对（用于下一轮 condense）
        history.append((user_input, answer))

        # 可选：限制历史长度（防止过长）
        if len(history) > 3:
            history = history[-3:]