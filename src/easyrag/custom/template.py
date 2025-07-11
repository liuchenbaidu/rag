QA_TEMPLATE = """\
    上下文信息如下：
    ----------
    {context_str}
    ----------
    请你基于上下文信息而不是自己的知识，回答以下问题。

    **重要指令**：
    - 如果问题是选择题（包含A、B、C、D选项），你必须严格按照以下格式回答：
      * 单选题：只输出一个选项字母，格式为 ["A"] 或 ["B"] 或 ["C"] 或 ["D"]
      * 多选题：输出多个选项字母，格式为 ["A", "B"] 或 ["A", "C", "D"] 等
      * 绝对不要输出选项的具体内容，只输出字母！
      * 绝对不要添加任何解释或说明！
    - 如果问题是开放性问题，可以分点作答，如果上下文信息没有相关知识，可以回答不确定，不要复述上下文信息。

    {query_str}

    回答：\
    """

MERGE_TEMPLATE = """\
    上下文：
    ----------
    {context_str}
    ----------
    
    你将看到一个问题，和这个问题对应的参考答案

    请基于上下文知识而不是自己的知识补充参考答案，让其更完整地回答问题
    
    请注意，严格保留参考答案的每个字符，并将补充的内容和参考答案合理地合并，输出更长更完整的包含更多术语和分点的新答案
    
    请注意，严格保留参考答案的每个字符，并将补充的内容和参考答案合理地合并，输出更长更完整的包含更多术语和分点的新答案
    
    请注意，严格保留参考答案的每个字符，并将补充的内容和参考答案合理地合并，输出更长更完整的包含更多术语和分点的新答案

    问题：
    {query_str}

    参考答案：
    {answer_str}

    新答案：\
    """

SUMMARY_EXTRACT_TEMPLATE = """\
    这是这一小节的内容：
    {context_str}
    请用中文总结本节的关键主题和实体。

    总结：\
    """

HYDE_PROMPT_ORIGIN = """\
    Please write a passage to answer the question
    Try to include as many key details as possible
    {context_str}
    Passage:\
    """

HYDE_PROMPT_MODIFIED_V1 = """\
    你是系统运维专家，现在请你结合通信和系统运维的相关知识回答下列问题，
    请尽量包含更多你所知道的的关键细节。请详细分析可能的原因，提出有效的诊断步骤和解决方案。
    {context_str}
    请尽可能简洁的回答:\
    """

HYDE_PROMPT_MODIFIED_V2 = """\
    你是系统运维专家，现在请你结合通信和系统运维的相关知识回答下列问题，
    请详细分析可能的原因，返回有用的内容。
    {context_str}
    最终的回答请尽可能的精简:\
    """

HYDE_PROMPT_MODIFIED_MERGING = """\
    你是系统运维专家，现在请你结合通信和系统运维的相关知识回答下列问题，
    现在有给定一个问题，一个生成的可能可用的文档和一个检索出的相关的上下文信息，你需要将上述问题和信息总结为一个文档，
    要求：这个文档要包含尽可能多的关键细节，要求尽可能详细，但是不要复述上下文信息。
    {context_str}
    不需要阐述无关信息和无关注释和总结，只需要关键信息，最终的回答请尽可能的精简
    请按照要求作答：\
    """
