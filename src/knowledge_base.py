from typing import TypedDict


class FaqEntry(TypedDict):
    # 能触发该 FAQ 的关键词集合。
    keywords: tuple[str, ...]

    # 命中 FAQ 后返回给用户的答案。
    answer: str


# 本地 FAQ 知识库。后续可以替换为数据库、文档或向量库。
FAQ_ENTRIES: list[FaqEntry] = [
    {
        "keywords": ("退款", "到账", "多久"),
        "answer": "退款申请审核通过后，原路退款通常在 3 至 5 个工作日到账。",
    },
    {
        "keywords": ("客服", "工作时间"),
        "answer": "人工客服工作时间为每日 9:00 至 18:00。",
    },
    {
        "keywords": ("密码", "重置"),
        "answer": "请在登录页选择“忘记密码”，按提示完成密码重置。",
    },
]


# def find_faq_answer(query: str) -> str | None:
#     # 依次检查知识库中的每一条 FAQ。
#     for entry in FAQ_ENTRIES:
#         # 只有当问题包含该 FAQ 的全部关键词时，才认为匹配。
#         if all(keyword in query for keyword in entry["keywords"]):
#             return entry["answer"]
#
#     # 遍历结束仍未匹配时，使用 None 表示“没有找到答案”。
#     return None

def find_faq_answer(query: str) -> str | None:
    # 依次检查知识库中的每一条 FAQ。
    for entry in FAQ_ENTRIES:
        # 统计当前问题命中了该 FAQ 的多少个关键词。
        matched_keyword_count = sum(
            keyword in query for keyword in entry["keywords"]
        )

        # 至少命中两个关键词，才认为问题与该 FAQ 足够相关。
        if matched_keyword_count >= 2:
            return entry["answer"]

    # 遍历结束仍未匹配时，使用 None 表示“没有找到答案”。
    return None




