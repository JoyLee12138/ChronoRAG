from string import Template

# 单轮
RAG_QA_PROMPT_TEMPLATE = Template(
    """你是一位中国近代史专家，请严格根据以下提供的参考资料回答问题。
- 如果资料中有明确答案，请直接、简洁地回答，并在句末标注引用编号（如 [1]、[2]）。
- 如果资料中没有相关信息，请回答：“根据现有资料，无法回答该问题。”
- 禁止编造事实、添加个人观点或推测。

参考资料：
$context

问题：$question

回答："""
)

# 多轮模板（包含最近一轮历史）
CHAT_RAG_PROMPT_TEMPLATE = Template(
    """你是一位中国近代史专家。请根据以下参考资料和最近的对话回答当前问题。
- 回答必须基于参考资料，禁止幻觉。
- 若参考资料不足，明确说明“无法回答”。

最近对话：
$chat_history

参考资料：
$context

当前问题：$question

回答："""
)

def build_rag_prompt(question: str, context: str) -> str:
    return RAG_QA_PROMPT_TEMPLATE.substitute(question=question, context=context)

def build_chat_rag_prompt(question: str, context: str, chat_history: str = "") -> str:
    return CHAT_RAG_PROMPT_TEMPLATE.substitute(
        question=question, context=context, chat_history=chat_history
    )