

# 口袋记忆 · MemoOrb

**口袋记忆 · MemoOrb** 是一个基于 **Python 3.11 + LangChain + Gradio** 的智能文档问答系统，将用户的电子书、PDF 或 TXT 文件转化为可检索的知识库/AI agent，通过网页聊天窗口与大语言模型交互，实现严格基于用户文档内容的 AI agent阅读助手。让每个认真阅读过的文档都变成你的“知识朋友”，在任何需要时随问随答，严谨而温柔。
## 示例应用场景

- **学生**：将教材/讲义放入库中，随时提问并查出处  
- **读者**：汇总 Kindle/微信读书笔记，像问老师一样提问  
- **律师/医生**：将法规/指南等私域文档转为问答系统  
- **写作者**：整理过往写作资料，边写边调取灵感与引用  

---

## 1️⃣项目背景 
在现代信息爆炸的时代，知识与文档以各种形式存在：电子书、PDF、Markdown 笔记、教学资料等。普通的读书或学习方式难以高效检索过去阅读的内容，也难以形成系统化的知识复用。口袋记忆 · MemoOrb应运而生：它可以将用户的读书笔记、PDF 文档、教材资料等自动转化为可检索的知识库

通过基于 RAG（Retrieval-Augmented Generation） 的对话机制，实现严格基于已有资料回答问题，用户无需掌握复杂工具，只需上传资料，即可获得类似私人助教的交互体验

这个项目不仅适合个人学习和知识管理，也能应用于学术研究、法律法规查询、医疗指南检索等专业场景。它体现了我在Python 编程、LLM 应用、向量数据库构建以及前端交互设计上的综合能力，同时展示了我对信息检索、知识管理和自然语言处理的深入理解。

---

## 2️⃣功能亮点

| 功能             | 说明 |
|-----------------|------------------------------------------------|
| 📥 文件导入即建库 | 用户只需将书籍/笔记放入 `books/` 文件夹即可自动处理 |
| 🧠 基于文档回答   | 限定 LLM 严格参考你的文档内容，不靠幻想回答 |
| 📌 中文引用支持   | 输出示例：“《康德哲学入门》- 第42页；《读书笔记.md》” |
| 💡 通俗解释能力   | 结合 LLM 通用能力，实现复杂概念的“读书式”通俗化 |
| 🔄 模块灵活替换   | 嵌入模型、向量库、UI 框架都可插拔替换 |
| 🚀 二次开发友好   | 结构清晰、逻辑独立，每个功能模块均可拆出独立使用 |

---

## 3️⃣技术栈
口袋记忆 · MemoOrb 采用了现代化的 Python 技术生态和 LLM 工程实践，支持模块化扩展和多场景应用。

| 模块 | 默认选择 | 可替换 / 可拓展 |
|------|---------|----------------|
| **Python 版本** | 3.11 | - |
| **核心框架** | LangChain | DeepSeek, Qwen 等 |
| **嵌入模型** | BAAI/bge-m3（本地） | OpenAI Embeddings, DeepSeek Embeddings |
| **大语言模型** | GPT-4o-mini via OpenAI | Qwen, Yi, Claude, Moonshot 等 |
| **向量数据库** | Chroma（本地持久化） | FAISS, PGVector, Weaviate |
| **文档加载器** | LangChain 内置文档处理模块 | Kindle 划线导出、微信读书笔记等 |
| **数据处理** | Sentence-Transformers, CrossEncoder | PyTorch, NumPy, Pandas |
| **Web UI** | Gradio | Streamlit, Flask, FastAPI |
| **异步与网络请求** | aiohttp, httpx, asyncio | requests, websockets |
| **环境管理** | Python venv | Conda, Docker（可选） |
| **辅助工具** | tqdm, python-dotenv, pathlib | - |
| **文件格式支持** | PDF, Markdown, TXT, EPUB | Word (.docx), PPT, 扫描图像（可拓展 OCR） |
| **可视化** | Matplotlib, Altair | Plotly, Seaborn |

> 💡 说明：该技术栈展示了 MemoOrb 在 LLM 应用、向量数据库、文档处理和前端交互上的全链路能力，结构清晰且模块可插拔，方便二次开发与扩展。


## 4️⃣项目结构

```

rag-books/
├── books/                 # 用户书籍/笔记目录
├── chroma_db/             # 向量库存储目录
├── ingest.py              # 文档处理与向量化
├── app.py                 # Gradio Web 界面
├── requirements.txt       # Python 依赖
├── .env.example           # 环境变量示例
├── start.sh               # 一键启动脚本 (Linux/Mac)
└── start.bat              # 一键启动脚本 (Windows)

````

---


## 5️⃣核心流程（RAG 机制）

1. **收集资料（`books/` 文件夹）**  
   - 支持格式：PDF（可复制文字）、Markdown、TXT  
   - 用户将读书笔记、电子书等放入该目录，无需额外操作  

2. **预处理与切分（`ingest.py`）**  
   - 调用 `PyPDFLoader` / `UnstructuredMarkdownLoader` 等读取文档  
   - 使用 `RecursiveCharacterTextSplitter` 进行中文友好分段  
   - 自动生成元数据：书名、页码、文件名等  

3. **嵌入与建库（Chroma 本地向量库）**  
   - 默认使用本地嵌入模型 `BAAI/bge-m3`（可切换 OpenAI Embedding）  
   - 向量库持久化存储于 `chroma_db/`  

4. **用户提问（Gradio Web UI ）**  
   - 用户通过网页输入问题  
   - 系统检索最相关的文档片段，传入 LLM  
   - 返回结构化答案 + 中文格式引用  

---

## 6️⃣安装与运行

```bash
# 打开terminal，克隆仓库
git clone https://github.com/your-username/MemoOrb.git
cd MemoOrb

# 创建 Python 3.11 虚拟环境（macOS/Linux）
python3.11 -m venv .venv
source .venv/bin/activate
# 创建 Python 3.11 虚拟环境（Windows）
.venv\Scripts\activate

# 安装依赖
pip install --upgrade pip
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env

# 编辑 .env 文件，添加 OpenAI/其他模型 Key
OPENAI_API_KEY=your_openai_api_key
HUGGINGFACE_API_TOKEN=your_huggingface_token
CHROMA_DB_DIR=./chroma_db

# 自动读取 books/ 中的文档，分段、生成嵌入向量，保存到 chroma_db/，默认有三本书
python ingest.py

# 启动 Gradi Web 界面，访问http://localhost:7860
python app.py

# 关闭项目
Ctrl+C
deactivate

> ⚠️ 如果是 Windows 用户，可以使用 `start.bat` 或直接运行 `python app.py`。
> ⚠️ 如果是 Linus/MacOS 用户，可以使用 `start.sh` 或直接运行 `python app.py`。
> ⚠️ 注意：必须使用 Python 3.11 环境，否则依赖可能不兼容。


```

### 项目使用注意事项
* `__pycache__/` 可以删除，不影响运行
* `chroma_db/` 会自动生成，无需手动修改
* 所有书籍文件放在 `books/` 目录，默认初始状态有三本书



---

## 7️⃣ 运行结果
![运行界面](images/run_example.png)


---

## 8️⃣ 后续优化建议

| 优化方向          | 内容 |
|-----------------|-----------------------------------------------|
| 自动监听          | 用 `watchdog` 自动监听 `books/` 文件夹更新，自动增量导入 |
| OCR 能力          | 引入 `paddleocr` 支持扫描版 PDF 与图像笔记 |
| 自定义提示词       | 允许用户控制助教风格：学术/通俗/诗意风/引经据典 |
| 用户会话存档       | 支持导出 PDF/Markdown 的聊天与引用 |
| 多模型适配        | 接入本地配置 Qwen、Moonshot、Claude 等 |

---

## 9️⃣License

MIT / Apache / GPL

---

## 0️⃣关于我

我是一名专注于 **量化分析、AI 应用与大模型集成开发 (AIIC)** 的开发者，热衷将算法、数据与智能应用结合，打造高性能、可扩展的技术产品。  

- 📧 **邮箱**: dannike19980521@163.com 
- 💻 **GitHub**: https://github.com/Danni-Ke
- 🍠 **小红书**: https://xhslink.com/m/1HTOwFfmMBA

### 技能与方向
- **量化与数据分析**：Python、Pandas、NumPy、scikit-learn、matplotlib  
- **AI/大模型应用**：LangChain、OpenAI API、文本/语音/图像生成与处理  
- **向量检索与知识管理**：Chroma、FAISS、PGVector  
- **全栈开发与部署**：FastAPI、Gradio、Docker、云端服务  
- **自动化与工具链**：脚本化数据处理、自动化文档处理、RAG 文档问答系统  

### 关于贡献/PR

欢迎提交 PR 或 Issues，如果你希望：

* 增加支持的文档类型
* 优化问答模型或界面
* 改进向量检索逻辑

---

