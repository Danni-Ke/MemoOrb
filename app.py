from pathlib import Path
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gradio as gr
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, Document
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import BaseRetriever, Document
from langchain.prompts import PromptTemplate
from typing import List, Tuple
from sentence_transformers import CrossEncoder

# =========================
# 环境与常量
# =========================
load_dotenv()
DB_DIR = "./chroma_db"
BOOKS_DIR = "./books"

# =========================
# 嵌入模型
# =========================
model_name = "BAAI/bge-m3"
test_model = SentenceTransformer(model_name)
sample_vec = test_model.encode("测试向量")
print(f"✅ 当前使用嵌入模型：{model_name}，输出维度：{len(sample_vec)}")
RERANKER = CrossEncoder("BAAI/bge-reranker-base", device="cpu")
EMBED_MODEL = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# =========================
# LLM
# =========================
LLM = ChatOpenAI(
    model_name="deepseek-chat",
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=os.getenv("OPENAI_API_BASE")
)

# =========================
# VectorStore
# =========================
vectordb = Chroma(persist_directory=DB_DIR, embedding_function=EMBED_MODEL)

# ---------- 工具函数 ----------
def _is_under(base_dir: str, p: Path) -> bool:
    """仅允许 books/ 目录内的路径"""
    try:
        p_resolved = p.resolve()
        base_resolved = Path(base_dir).resolve()
        p_resolved.relative_to(base_resolved)
        return True
    except Exception:
        return False

def _pick_source_from_metadata(meta: dict) -> Tuple[Path, str, int]:
    """
    从 metadata 里提取本地文件路径、显示名与页码；
    仅允许 books/ 下真实存在的文件。
    """
    src = meta.get("book") or meta.get("source") or meta.get("file_path") or meta.get("title") or ""
    page = meta.get("page")
    p = Path(str(src))
    if not p.is_absolute():
        p = Path(BOOKS_DIR) / p.name
    if not p.exists() or not _is_under(BOOKS_DIR, p):
        raise FileNotFoundError("source not in books/")
    name = p.stem.strip()
    # page 合法化
    if isinstance(page, str) and page.isdigit():
        page = int(page)
    if not isinstance(page, int):
        page = None
    return p, name, page


def rerank(query, docs, top_k=5):
    if not docs:
        return []
    pairs = [(query, d.page_content[:3000]) for d in docs]  # 防止过长
    scores = RERANKER.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranked[:top_k]]

# 自定义一个带 rerank 的 retriever
class RerankRetriever:
    def __init__(self, vectordb, fetch_k=40, top_k=5):
        self.vectordb = vectordb
        self.fetch_k = fetch_k
        self.top_k = top_k

    def get_relevant_documents(self, query):
        docs = self.vectordb.similarity_search(query, k=self.fetch_k)
        return rerank(query, docs, self.top_k)
        
# =========================
#  QA 提示词（以文档为主，证据不足再拒答）
# =========================
QA_PROMPT = PromptTemplate(
    input_variables=["context", "question", "chat_history"],
    template=(
        "你是图书问答助手，回答时**主要依据**下面的<文档>内容；"
        "若文档没有直接给出结论，可以在不引入外部知识的前提下，"
        "对文档里的相关信息进行**严谨的概括或归纳**；"
        "如果文档未包含与问题高度相关的内容（如书名、关键概念），"
        "必须回答：“当前知识库未包含相关信息。请添加相关资料后再提问！”\n\n"
        "<历史对话>\n{chat_history}\n\n"
        "<文档>\n{context}\n\n"
        "<问题>\n{question}\n\n"
        "请给出简洁、依据充分的中文回答。"
    )
)


# =========================
# 自定义“阈值检索器”
# =========================
class ThresholdRerankRetriever(BaseRetriever):
    """先向量召回/阈值过滤，再用交叉编码器重排。"""
    vectorstore: Chroma
    k: int = 4                   # 最终返回条数
    fetch_k: int = 40            # 候选池（用于重排）
    score_threshold: float = 0.20
    min_keep: int = 3            # 阈值过严时至少保留 N 条
    reranker: CrossEncoder = None
    rerank_top_k: int = 4        # 重排后取前 N 条（一般等于 k）

    # 兼容不同分数语义：返回 0~1 相似度，越大越相关
    def _normalize_pairs(self, pairs):
        norm = []
        for d, s in pairs:
            if s is None:
                score = 0.0
            elif 0 <= s <= 1:
                score = float(s)       # 已是相似度
            else:
                score = 1.0 / (1.0 + float(s))  # 距离→相似度
            norm.append((d, score))
        return norm

    def _get_relevant_documents(self, query: str) -> List[Document]:
        # 1) 粗召回（取大一些给重排用）
        try:
            pairs = self.vectorstore.similarity_search_with_relevance_scores(query, k=self.fetch_k)
            pairs = self._normalize_pairs(pairs)
        except Exception:
            # 兜底：没有分数接口就直接取
            docs = self.vectorstore.similarity_search(query, k=self.fetch_k)
            pairs = [(d, 1.0) for d in docs]

        # 2) 阈值过滤 + 至少保底
        filtered = [(d, sc) for d, sc in pairs if sc >= self.score_threshold]
        filtered.sort(key=lambda x: x[1], reverse=True)
        if len(filtered) < self.min_keep and pairs:
            pairs.sort(key=lambda x: x[1], reverse=True)
            need = self.min_keep - len(filtered)
            for d, sc in pairs:
                if (d, sc) not in filtered:
                    filtered.append((d, sc))
                    if len(filtered) >= self.min_keep:
                        break
        candidates = [d for d, _ in filtered] or [d for d, _ in pairs]

        if not candidates:
            # 双兜底：MMR → 普通相似
            try:
                return self.vectorstore.max_marginal_relevance_search(query, k=self.k, fetch_k=self.fetch_k)
            except Exception:
                return self.vectorstore.similarity_search(query, k=self.k)

        # 3) 重排（Cross-Encoder）
        try:
            if self.reranker is not None:
                pairs_ce = [(query, d.page_content[:3000]) for d in candidates]  # 截断防超长
                scores = self.reranker.predict(pairs_ce)  # numpy array
                ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
                return [d for d, _ in ranked[: self.rerank_top_k]]
        except Exception:
            # 重排失败就忽略重排
            pass

        # 4) 不重排/重排失败时按相似度取前 k
        return candidates[: self.k]

# ✅ 实例化自定义检索器（← 你问的“retriever 在哪儿实例化”就在这里）
retriever = ThresholdRerankRetriever(
    vectorstore=vectordb,
    k=4,
    fetch_k=40,            # 给重排更大的候选池
    score_threshold=0.28,  # 0.15 更放开 / 0.30 更保守
    min_keep=3,
    reranker=RERANKER,
    rerank_top_k=4
)

# =========================
# Memory & QA Chain
# =========================
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    input_key="question",
    output_key="answer"
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=LLM,
    retriever=retriever,
    memory=memory,
    return_source_documents=True,
    combine_docs_chain_kwargs={"prompt": QA_PROMPT}
)

# =========================
# 引用格式化（仅 books/ 中且本次真的被命中的文档）
# =========================
def format_refs(docs: List[Document], max_refs: int = 3) -> str:
    refs = []
    seen = set()
    for d in docs:
        meta = (d.metadata or {})
        try:
            p, name, page = _pick_source_from_metadata(meta)
        except Exception:
            # 非 books/ 或无效源，跳过
            continue
        ref = f"《{name}》" + (f" - 第{int(page)}页" if page else "")
        if ref not in seen:
            seen.add(ref)
            refs.append(ref)
        if len(refs) >= max_refs:  # 限制最多 N 条
            break
    if not refs:
        return "参考来源：无"
    return "参考来源：" + "；".join(refs)

# =========================
# “你脑子里有什么”类问题判断 & 展示
# =========================
def is_meta_question(q: str) -> bool:
    ql = q.lower()
    keywords = [
        "你脑子里", "你都知道", "你记得什么", "你记得哪些",
        "你有哪些内容", "你有什么资料", "你能回答什么",
        "你能回答哪些", "你的知识", "你会什么", "你懂什么",
        "what do you know", "what do you remember",
        "what's in your brain", "what can you answer",
        "what info do you have", "your knowledge",
        "what topics do you know", "what books do you remember",
        "do you have any info", "what sources do you have",
    ]
    return any(k in q or k in ql for k in keywords)

def describe_knowledge_base():
    try:
        coll = getattr(vectordb, "_collection", None)
        if coll is None:
            return "⚠️ 向量库未加载，请先构建知识库。"
        metas = (coll.get(include=["metadatas"]) or {}).get("metadatas") or []
        sources = set()
        for meta in metas:
            if not meta:
                continue
            # 仅统计 books/ 下可用文件
            try:
                p, name, _ = _pick_source_from_metadata(meta)
                if p.exists():
                    sources.add(name)
            except Exception:
                continue
        if not sources:
            return "😢 向量库为空，请先导入并构建知识库。"
        return "\n\n" + "\n".join(f"• 《{s}》" for s in sorted(sources))
    except Exception as e:
        return f"读取向量库失败：{e}"

def list_books_from_folder(folder_path=BOOKS_DIR):
    supported_exts = [".pdf", ".epub", ".txt", ".md"]
    if not os.path.exists(folder_path):
        return "📂 未找到 books 文件夹。"
    files = [
        Path(f).stem.strip().replace("_", " ")
        for f in os.listdir(folder_path)
        if any(f.lower().endswith(ext) for ext in supported_exts)
    ]
    if not files:
        return "📂 books 文件夹中暂无支持的文件。"
    return "\n\n" + "\n".join(f"• 《{f}》" for f in sorted(files))

# =========================
# 主答复函数（严格拒答 & 仅追加安全引用）
# =========================
def answer_question(query, history):
    if is_meta_question(query):
        return describe_knowledge_base()

    result = qa_chain({"question": query})
    answer = (result.get("answer") or "").strip()
    source_docs = result.get("source_documents") or []

    # ✅ 如果没有命中任何文档，强制拒答
    if not source_docs:
        return "当前知识库未包含相关信息。请添加相关资料后再提问！"

    # 保存简明对话记忆
    qa_chain.memory.save_context({"question": query}, {"answer": answer})

    # ✅ 电子栅栏：如果模型调皮（比如说“虽然文档没提，但我推测…”），仍然拦掉
    reject_triggers = [
        "未包含相关信息", "无法作答", "没有找到", "资料不足",
        "not enough information", "unable to answer", "out of scope"
    ]
    if any(p in answer for p in reject_triggers):
        return "当前知识库未包含相关信息。请添加相关资料后再提问！"

    # ✅ 正常回答才附带引用
    refs = format_refs(source_docs, max_refs=3)
    return f"{answer}\n\n{refs}"




# =========================
# 运行 ingest
# =========================
def run_ingest():
    try:
        import ingest
        ingest.main()
        return "✅ 知识库构建完成！"
    except Exception as e:
        return f"❌ 构建失败：{e}"

# =========================
# Gradio UI（事件函数在前）
# =========================

# 保存聊天历史（[(user, assistant), ...]）
# 注意：真正的 gr.State 会在 Blocks 里创建；这里只是函数签名用
def on_send(user_msg, history):
    user_msg = (user_msg or "").strip()
    if not user_msg:
        return gr.update(), history, history
    answer = answer_question(user_msg, history)
    history = history + [(user_msg, answer)]
    return "", history, history

def on_retry(history):
    if not history:
        return history, history
    last_user = history[-1][0]
    new_answer = answer_question(last_user, history[:-1])
    history = history[:-1] + [(last_user, new_answer)]
    return history, history

def on_undo(history):
    if not history:
        return history, history
    history = history[:-1]
    return history, history

def on_clear():
    return [], []

def update_lang(lang):
    if lang == "中文":
        return (
            gr.update(label="📘 你的图书馆"),
            gr.update(label="📚 你的知识库"),
            gr.update(label="🛠️ 状态提示"),
            gr.update(value="🔁 重新构建知识库"),
            "### 📚口袋记忆 · MemoOrb",
            "🌿 将你要阅读的书籍（PDF / EPUB / TXT / MD）放入 `books/` 文件夹，点击左侧按钮刷新知识库，然后就可以直接和你的书本对话。🌿",
            gr.update(label="💬 对话 / Chat"),
            gr.update(label="✍️ 消息 / Message", placeholder="请输入你的问题…"),
            gr.update(value="▶️ 发送 / Send"),
            gr.update(value="🔄 重试 / Retry"),
            gr.update(value="↩️ 撤回 / Undo"),
            gr.update(value="🗑️ 清空 / Clear"),
        )
    else:
        return (
            gr.update(label="📘 Books Library"),
            gr.update(label="📚 Knowledge Base"),
            gr.update(label="🛠️ Status"),
            gr.update(value="🔁 Rebuild Knowledge Base"),
            "### 📚 MemoOrb · Pocket Memory",
            "🌿 Put your books (PDF / EPUB / TXT / MD) into `books/`, click the button on the left to rebuild, then chat with your books. 🌿",
            gr.update(label="💬 Chat"),
            gr.update(label="✍️ Message", placeholder="Type your question…"),
            gr.update(value="▶️ Send"),
            gr.update(value="🔄 Retry"),
            gr.update(value="↩️ Undo"),
            gr.update(value="🗑️ Clear"),

        )

with gr.Blocks() as demo:
    with gr.Row():
    # ===== 左栏：把语言切换放在最上面 =====
        with gr.Column(scale=1):
            lang_selector = gr.Dropdown(
                choices=["中文", "English"],
                value="中文",
                label="🌐 Language / 语言",
                scale=1,            # 跟 Column 同步
                container=True      # 使用与其他控件一致的容器，视觉统一
            )

            books_box  = gr.Textbox(value=list_books_from_folder(), lines=8, label="📘 你的图书馆")
            kb_box     = gr.Textbox(value=describe_knowledge_base(), lines=8, label="📚 你的知识库")
            status_box = gr.Textbox(label="🛠️ 状态提示", interactive=False)
            update_btn = gr.Button("🔁 重新构建知识库")

            def run_ingest_and_refresh():
                msg = run_ingest()
                return list_books_from_folder(), describe_knowledge_base(), msg
            update_btn.click(run_ingest_and_refresh, outputs=[books_box, kb_box, status_box])

        # ===== 右栏：标题/描述 + 聊天区 =====
        with gr.Column(scale=3):
            title_md = gr.Markdown("### 📚口袋记忆 · MemoOrb")
            desc_md  = gr.Markdown("🌿 将你要阅读的书籍（PDF / EPUB / TXT / MD）放入 `books/` 文件夹，点击左侧按钮刷新知识库，然后就可以直接和你的书本对话。🌿")

            chat = gr.Chatbot(label="💬 对话 / Chat", height=480)
            # ---- 顶部右侧：发送（单独一行靠右）----
            with gr.Row():
                msg = gr.Textbox(
                    label="✍️ 消息 / Message",
                    placeholder="请输入你的问题…",
                    lines=1,
                    scale=9
                )
                send_btn = gr.Button("▶️ 发送 / Send", variant="primary", scale=1)
            # —— 在输入框下面放例子
            examples = gr.Examples(
                examples=[
                    ["《乌合之众》的核心观点是什么？"],
                    ["你脑子里有什么书本啊？"],
                    ["存在主义和虚无主义的区别是什么？"]
                ],
                inputs=msg
            )

            # ---- 下一行：重试 / 撤回 / 清空 ----
            with gr.Row():
                retry_btn = gr.Button("🔄 重试 / Retry")
                undo_btn  = gr.Button("↩️ 撤回 / Undo")
                clear_btn = gr.Button("🗑️ 清空 / Clear")

            # 会话状态（必须在 Blocks 内）
            chat_state = gr.State([])

            # 事件绑定（必须在 Blocks 内，且函数已在上方定义）
            send_btn.click(on_send, [msg, chat_state], [msg, chat_state, chat])
            msg.submit(on_send, [msg, chat_state], [msg, chat_state, chat])
            retry_btn.click(on_retry, [chat_state], [chat_state, chat])
            undo_btn.click(on_undo, [chat_state], [chat_state, chat])
            clear_btn.click(on_clear, None, [chat_state, chat])

    # ===== 语言切换联动（顶栏选择器生效）=====
    lang_selector.change(
        update_lang,
        inputs=[lang_selector],
        outputs=[
            books_box, kb_box, status_box, update_btn,  # 左侧
            title_md, desc_md,                          # 右侧标题/描述
            chat, msg,                                  # 聊天区标签/占位
            send_btn, retry_btn, undo_btn, clear_btn,    # 四个按钮文案
        ]
    )

if __name__ == "__main__":
    demo.launch()
