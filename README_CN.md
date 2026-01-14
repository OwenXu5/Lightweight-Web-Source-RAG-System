# 轻量级 Web 检索增强生成 (RAG) 系统 (Lightweight Web-Source RAG System)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![React](https://img.shields.io/badge/React-18.0+-61dafb.svg)](https://react.dev/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-009688.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

一个基于校园大语言模型平台 ChatECNU 构建的轻量级 RAG 系统，拥有现代化的 Web 界面。

本项目实现了一个完整的流程：
1.  **数据获取**：经过身份验证的网页抓取（修复了乱码问题）与智能分块。
2.  **索引构建**：基于 FAISS 的向量存储。
3.  **检索**：稠密向量检索 + 可选的神经重排序 (Rerank) + HyDE（假设性文档嵌入）。
4.  **界面交互**：支持实时流式输出和来源引用的现代化聊天 UI。

[English Documentation](README.md)

## ✨ 主要特性

- **Web 界面**：
    - **现代化 UI**：使用 React + Tailwind CSS + Framer Motion 构建。
    - **多轮对话历史**：侧边栏管理所有历史会话，支持创建新对话、删除对话和自动重命名。
    - **流式对话**：实时流式输出，打字机效果。
    - **来源查看器**：透明展示 AI 参考的具体文本片段，历史记录回溯时也会**同步加载**当时的参考来源。
    - **动态配置**：在 UI 中管理 URL、切换检索模式（HyDE/Hybrid）、调整参数。
- **健壮的后端**：
    - **FastAPI**：高性能异步 API。
    - **混合检索 (Hybrid Search)**：结合 **BM25 关键词检索** + **向量语义检索**，使用 **RRF (Reciprocal Rank Fusion)** 算法融合排序，大幅提升生僻词和专有名词的召回率。
    - **持久化存储**：使用 SQLite (`chat.db`) 自动保存所有对话记录和检索上下文。
    - **向量存储**：持久化的 FAISS 索引。
- **高级 RAG 技术**：
    - **HyDE**：生成假设性文档以增强语义匹配。
    - **Rerank**：神经重排序模型。

## 🚀 快速开始

### 前置条件
- Python 3.8+
- Node.js & npm (用于前端)
- ChatECNU 的 API Key (或兼容 OpenAI 格式的其他 API)

### 1. 后端设置

```bash
# 克隆仓库
git clone https://github.com/OwenXu5/Lightweight-Web-Source-RAG-System.git
cd Lightweight-Web-Source-RAG-System

# 安装 Python 依赖 (新增 rank_bm25, jieba)
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 重要：编辑 .env 文件，填入你的 OPENAI_API_KEY 和 OPENAI_API_BASE！

# 启动后端服务
python api.py
```
> 服务运行在：`http://localhost:8000`

### 2. 前端设置

打开一个新的终端窗口：
```bash
cd frontend

# 安装 Node 依赖
npm install

# 启动开发服务器
npm run dev
```
> 应用运行在：`http://localhost:5173`

---

## 🛠 配置说明

### 环境变量 (`.env`)

| 变量名 | 描述 |
|----------|-------------|
| `OPENAI_API_KEY` | **必需**。你的 API Key。 |
| `OPENAI_API_BASE` | **必需**。API 基础地址。 |
| `VECTOR_DIR` | 存储 FAISS 索引的目录 (默认: `wiki_vector_store`)。 |
| `REBUILD_FLAG` | 设置为 `True` 则在启动时强制重建索引。 |

### Web 界面设置

你可以在侧边栏中动态调整这些设置：

- **Knowledge Base URLs**：添加/删除 URL。修改后点击 **Rebuild Index** 生效。
- **Search Mode**：
    - **Hybrid Search (默认)**：同时使用 BM25 和向量检索，效果最好。
    - **HyDE**：假设性文档嵌入，适合短查询。
- **Rerank**：重排序开关。
- **Top-K**：召回数量。

## 📁 项目结构

```
.
├── api.py               # FastAPI 后端 (含 SQLite 会话管理)
├── main.py              # RAG 核心 (BM25, FAISS, RRF, Rerank)
├── chat.db              # SQLite 数据库 (存储聊天记录)
├── requirements.txt     # Python 依赖
├── .env                 # 配置文件
├── urls.txt             # URL 列表
└── frontend/            # React 前端
    ├── src/
    │   ├── components/  # 组件 (ChatInterface, Sidebar, SourceViewer)
    │   └── App.tsx      # 主应用逻辑
    └── ...
```

## 📝 故障排除

**1. "Connection Error" 或 SSL 问题**
- 确保 `.env` 中的 `OPENAI_API_BASE` 配置正确。
- 系统默认配置为 `verify=False` 以兼容内部网络证书。

**2. 中文乱码 (Mojibake)**
- 在 Web 界面点击 **Rebuild Index**。
- 更新后的抓取器会强制使用 UTF-8 解码来修复此问题。

**3. 前端无法连接后端**
- 确保 `python api.py` 正在运行。
- 检查端口 8000 是否被占用。

## 🤝 贡献

欢迎提交 Issue 或 PR！

## 📄 许可证

MIT License.
