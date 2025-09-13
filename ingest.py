import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
from dotenv import load_dotenv
from tqdm import tqdm
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from bs4 import BeautifulSoup
import ebooklib
from ebooklib import epub
import textract
from pathlib import Path

# ✅ 加载环境变量
load_dotenv()

# ✅ 嵌入模型配置
embedding_model_name = "BAAI/bge-m3"
embedding = HuggingFaceEmbeddings(model_name=embedding_model_name)

# ✅ 打印嵌入模型维度（可选）
if hasattr(embedding.client, 'get_sentence_embedding_dimension'):
    dim = embedding.client.get_sentence_embedding_dimension()
    print(f"📐 当前嵌入模型：{embedding_model_name}，向量维度：{dim}")
else:
    print(f"⚠️ 无法获取嵌入维度（模型：{embedding_model_name}）")

# ✅ 路径配置
DATA_PATH = "books"
DB_DIR = "chroma_db"
VALID_EXTENSIONS = [".pdf", ".txt", ".md", ".epub", ".html", ".doc"]
BOOKLIST_PATH = "booklist.json"

# ✅ 获取当前 books 文件夹中书籍列表
def get_current_book_list():
    return sorted([
        f for f in os.listdir(DATA_PATH)
        if any(f.lower().endswith(ext) for ext in VALID_EXTENSIONS)
    ])

# ✅ 加载缓存中的书单
def load_previous_book_list():
    if not os.path.exists(BOOKLIST_PATH):
        return []
    try:
        with open(BOOKLIST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

# ✅ 保存当前书单
def save_book_list(book_list):
    with open(BOOKLIST_PATH, "w", encoding="utf-8") as f:
        json.dump(book_list, f, ensure_ascii=False, indent=2)

# ✅ 加载 EPUB 文件
def load_epub(file_path):
    try:
        book = epub.read_epub(file_path)
        text = ""
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                text += soup.get_text(separator='\n')
        return Document(page_content=text.strip(), metadata={"source": file_path})
    except Exception as e:
        print(f"⚠️ EPUB 加载失败: {file_path} -> {e}")
        return None

# ✅ 加载 HTML 文件
def load_html(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
            text = soup.get_text(separator='\n')
            return Document(page_content=text.strip(), metadata={"source": file_path})
    except Exception as e:
        print(f"⚠️ HTML 加载失败: {file_path} -> {e}")
        return None

# ✅ 加载 DOC 文件
def load_doc(file_path):
    try:
        text = textract.process(file_path).decode('utf-8')
        return Document(page_content=text.strip(), metadata={"source": file_path})
    except Exception as e:
        print(f"⚠️ DOC 加载失败: {file_path} -> {e}")
        return None

# ✅ 加载所有支持的文档
def load_documents(book_files):
    documents = []
    print(f"🔍 共发现 {len(book_files)} 个新书文件，开始逐个加载...")

    for file_name in tqdm(book_files, desc="📄 加载文件中"):
        file_path = os.path.join(DATA_PATH, file_name)
        try:
            if file_path.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
            elif file_path.endswith(".txt") or file_path.endswith(".md"):
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read().replace('\x00', '').replace('\\', '')
                    documents.append(Document(page_content=text, metadata={"source": file_path}))
            elif file_path.endswith(".epub"):
                doc = load_epub(file_path)
                if doc:
                    documents.append(doc)
            elif file_path.endswith(".html"):
                doc = load_html(file_path)
                if doc:
                    documents.append(doc)
            elif file_path.endswith(".doc"):
                doc = load_doc(file_path)
                if doc:
                    documents.append(doc)
            else:
                print(f"⏩ 忽略暂不支持的格式: {file_path}")
        except Exception as e:
            print(f"⚠️ 加载失败: {file_path} -> {e}")

    return documents

# ✅ 主流程（增量更新）
def ingest():
    current_books = get_current_book_list()
    previous_books = load_previous_book_list()

    if not previous_books:
        print("🕐 第一次初始化，构建整个知识库可能需要较长时间，请耐心等待...")

    if current_books == previous_books:
        print("✅ 没有检测到新书籍文档加入，自动跳过重新构建知识库。")
        return

    new_books = list(set(current_books) - set(previous_books))
    if not new_books:
        print("📋 虽然文件变更，但没有新增有效书籍格式。")
        return

    print(f"📘 检测到新增书籍：{len(new_books)} 本：")
    for b in new_books:
        print(f" - {b}")

    print("📄 Step 1: 加载新增书籍文档中...")
    docs = load_documents(new_books)

    if not docs:
        print("❌ 未能成功加载任何文档。请检查格式或内容。")
        return

    print("✂️ Step 2: 识别新增书籍文档中...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "。", "！", "？", "!", "?", " "]
    )
    splits = splitter.split_documents(docs)
    splits = [doc for doc in splits if len(doc.page_content) <= 1500]

    print(f"🧠 Step 3: 共 {len(splits)} 个文本块，开始增量入库...")
    try:
        vectordb = Chroma(persist_directory=DB_DIR, embedding_function=embedding)
        vectordb.add_documents(splits)
        vectordb.persist()
        print("✅ 所有书籍文档已成功添加到您的知识库 🎉")
        save_book_list(current_books)
    except Exception as e:
        print(f"❌ 入库失败：{e}")

# ✅ 供 app.py 调用
def main():
    ingest()

if __name__ == "__main__":
    main()
