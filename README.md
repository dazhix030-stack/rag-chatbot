# RAG Chatbot (Nexus-8)

ローカルLLM + RAGを使った労働基準法チャットボット。ChromaDB + BM25のハイブリッド検索で関連条文を取得し、Qwen2.5-7Bが回答を生成する。

## 機能

- ChromaDB（ベクトル検索）+ BM25（キーワード検索）のハイブリッド検索
- クエリ展開（キーワード自動抽出）+ リランキング
- Critic（自己評価）+ Fixer（自動修正）による回答品質向上
- Flask APIサーバー + シンプルなWebUI

## 使用技術

- Flask（APIサーバー）
- LangChain + ChromaDB（RAG）
- BM25（キーワード検索）
- Ollama + Qwen2.5-7B（ローカルLLM）
- bge-m3（埋め込みモデル）

## セットアップ

```bash
pip install flask flask-cors langchain-ollama langchain-chroma langchain-community rank_bm25
```

Ollamaのインストールと起動：

```bash
ollama pull qwen2.5:7b
ollama pull bge-m3
ollama serve
```

## 実行方法

```bash
# APIサーバーとして起動
python nexus_engine.py --api

# CLIモードで起動
python nexus_engine.py
```

ブラウザで `index.html` を開いてチャット開始。

## ファイル構成

```
nexus_engine.py   # Flask APIサーバー + RAGエンジン
index.html        # チャットUI
style.css         # スタイルシート
```

## 動作環境

- Python 3.12
- Ollama（ローカルLLM実行環境）
- GPU推奨（RTX 4050で動作確認済み）
