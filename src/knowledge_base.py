from typing import TypedDict


class FaqEntry(TypedDict):
    # 必须先满足的业务主题关键词。
    required_keywords: tuple[str, ...]

    # 至少命中一个，表示用户确实表达了这类 FAQ 的意图。
    intent_keywords: tuple[str, ...]

    # FAQ 命中后返回的可信答案。
    answer: str


# 本地 FAQ 知识库。
# 每条 FAQ 都拆成“主题”和“意图”两层关键词。
FAQ_ENTRIES: list[FaqEntry] = [
    {
        "required_keywords": ("退款",),
        "intent_keywords": (
            "多久",
            "什么时候",
            "几天",
            "工作日",
            "多长时间",
            "何时",
        ),
        "answer": "退款申请审核通过后，原路退款通常在 3 至 5 个工作日到账。",
    },
    {
        "required_keywords": ("客服",),
        "intent_keywords": (
            "工作时间",
            "上班时间",
            "几点",
            "什么时候",
            "何时",
        ),
        "answer": "人工客服工作时间为每日 9:00 至 18:00。",
    },
    {
        "required_keywords": ("密码",),
        "intent_keywords": (
            "重置",
            "忘记",
            "找回",
            "修改",
        ),
        "answer": "请在登录页选择“忘记密码”，按提示完成密码重置。",
    },
]


def find_faq_answer(query: str) -> str | None:
    # 依次检查知识库中的每一条 FAQ。
    for entry in FAQ_ENTRIES:
        # 主题关键词必须全部出现。
        required_matched = all(
            keyword in query
            for keyword in entry["required_keywords"]
        )

        # 意图关键词至少出现一个。
        intent_matched = any(
            keyword in query
            for keyword in entry["intent_keywords"]
        )

        # 主题和意图同时满足，才认为 FAQ 命中。
        if required_matched and intent_matched:
            return entry["answer"]

    # 所有 FAQ 都检查过但没有匹配时，返回 None。
    return None