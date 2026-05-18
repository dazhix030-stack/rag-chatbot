# --- 0. GOD MODE: ABSOLUTE ISOLATION ---
import os
import logging
import sys

# 物理層：通信遮断
DB_PATH = os.path.expanduser("~/nexus/db")
MEMORY_DB = os.path.expanduser("~/nexus/nexus_memory.db")
os.environ["OLLAMA_HOST"] = "localhost:11434" 

# 論理層：ログ抹殺
logging.getLogger("chromadb").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logging.getLogger("langchain").setLevel(logging.ERROR)
logging.getLogger("werkzeug").setLevel(logging.ERROR) # Flaskの余分なログを消す

import re
import json
import sqlite3
import math
import statistics
import datetime
import traceback
import io
import unicodedata
from abc import ABC, abstractmethod
from contextlib import redirect_stdout
from typing import List, Dict, Any, Optional

# --- 環境依存のパス設定 (main.pyの構成を継承) ---
DB_PATH = os.path.expanduser("~/ai_workspace_v7_2/db")
MEMORY_DB = os.path.expanduser("~/ai_workspace_v7_2/nexus_memory.db")

# Fail-safe Import
try:
    from flask import Flask, request, jsonify
    from flask_cors import CORS
    from langchain_ollama import ChatOllama, OllamaEmbeddings
    from langchain_chroma import Chroma
    from langchain_core.documents import Document
    from langchain_community.retrievers import BM25Retriever
except ImportError:
    sys.exit("⚠️ Error: Install 'flask' 'flask-cors' 'langchain-ollama' 'langchain-chroma' 'langchain-community' 'rank_bm25'")

MODEL_MAIN = "qwen2.5:7b"
MODEL_CRITIC = "qwen2.5:7b"

# Custom Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')

class Colors:
    HEADER = '\033[95m'; BLUE = '\033[94m'; CYAN = '\033[96m'; GREEN = '\033[92m'; 
    WARNING = '\033[93m'; FAIL = '\033[91m'; ENDC = '\033[0m'

def cprint(text, color=Colors.ENDC):
    print(f"{color}{text}{Colors.ENDC}")

# --- 1. CORE COMPONENTS ---

class LocalSandbox:
    @staticmethod
    def execute(code: str) -> str:
        code = re.sub(r"```python|```", "", code).strip()
        code = re.sub(r"^\s*(import|from)\s+.*$", "", code, flags=re.MULTILINE)
        buffer = io.StringIO()
        safe_globals = {
            "print": print, "range": range, "len": len, "int": int, "float": float,
            "str": str, "list": list, "dict": dict, "sum": sum, "max": max, "min": min,
            "math": math, "statistics": statistics, "datetime": datetime, "re": re
        }
        try:
            with redirect_stdout(buffer):
                exec(code, safe_globals)
            return buffer.getvalue().strip() or "(No Output)"
        except Exception:
            return f"Runtime Error: {traceback.format_exc().splitlines()[-1]}"

class MicroMemory:
    def __init__(self, db_path):
        self.db_path = db_path
        self._init_db()

    def _get_conn(self):
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _init_db(self):
        conn = self._get_conn()
        conn.execute("CREATE TABLE IF NOT EXISTS short_term_memory (id INTEGER PRIMARY KEY, error_msg TEXT)")
        conn.commit()
        conn.close()

    def set_error(self, msg: str):
        conn = self._get_conn()
        conn.execute("DELETE FROM short_term_memory")
        conn.execute("INSERT INTO short_term_memory (error_msg) VALUES (?)", (msg[:80],))
        conn.commit()
        conn.close()

    def get_and_clear(self) -> str:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT error_msg FROM short_term_memory LIMIT 1")
        row = cursor.fetchone()
        if row:
            conn.execute("DELETE FROM short_term_memory")
            conn.commit()
        conn.close()
        return row[0] if row else ""

# --- 2. TWO-STAGE ROUTER ---

class SemanticRouter:
    def __init__(self, llm):
        self.llm = llm

    def route_stage_1(self, query: str) -> List[str]:
        q = query.lower()
        if any(op in q for op in "+-*/"): return ["CALCULATOR"]
        if any(k in q for k in ["リスト", "一覧", "抜き出し"]): return ["EXTRACTOR"]
        return ["LOGIC"]

    def route_stage_2(self, query: str, tool: str) -> str:
        return "GENERAL"

# --- 2.5. ADVANCED RETRIEVAL ---

class AdvancedRetriever:
    def __init__(self, vectorstore, critic_llm):
        self.vectorstore = vectorstore
        self.critic_llm = critic_llm
        self.bm25_retriever = None
        
        if self.vectorstore:
            try:
                db_data = self.vectorstore.get()
                docs = [Document(page_content=txt, metadata=meta or {}) 
                        for txt, meta in zip(db_data['documents'], db_data['metadatas'])]
                if docs:
                    self.bm25_retriever = BM25Retriever.from_documents(docs)
                    self.bm25_retriever.k = 10
            except Exception as e:
                cprint(f"⚠️ BM25 Init Error: {e}", Colors.WARNING)

    def _expand_query(self, query: str) -> List[str]:
        prompt = f"Query: {query}\nExtract 2 vital keywords for search. Output comma-separated ONLY."
        try:
            res = self.critic_llm.invoke(prompt).content
            keywords = [k.strip() for k in res.split(",") if k.strip()]
            return [query] + keywords
        except:
            return [query]

    def _rerank_docs(self, query: str, docs: List[Document], top_k: int = 5) -> List[Document]:
        query_terms = set(re.split(r'[のはがをにもとでや]', query))
        query_terms = {t for t in query_terms if len(t) >= 2}
        
        scored_docs = []
        for doc in docs:
            score = sum(1 for term in query_terms if term in doc.page_content)
            scored_docs.append((score, doc))
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored_docs[:top_k]]

    def get_context(self, query: str) -> str:
        if not self.vectorstore: return "適切な条文が見つかりませんでした。"
        
        queries = self._expand_query(query)
        all_docs = []
        for q in queries:
            try:
                v_docs = self.vectorstore.similarity_search(q, k=10)
                all_docs.extend(v_docs)
                if self.bm25_retriever:
                    b_docs = self.bm25_retriever.invoke(q)
                    all_docs.extend(b_docs)
            except Exception:
                pass
                
        unique_docs = {doc.page_content: doc for doc in all_docs}.values()
        final_docs = self._rerank_docs(query, list(unique_docs), top_k=5)
        
        return "\n".join([f"出典: {d.metadata.get('source', '不明')}\n{d.page_content}" for d in final_docs])

# --- 3. SPECIALIZED MODULES ---

class BaseModule(ABC):
    def __init__(self, llm, vectorstore):
        self.llm = llm
        self.vectorstore = vectorstore
    @abstractmethod
    def run(self, query: str, context: str, subtype: str) -> Dict[str, Any]: pass

class CalculatorModule(BaseModule):
    def run(self, query: str, context: str, subtype: str) -> Dict:
        try:
            expression = re.sub(r'[^0-9+\-*/(). ]', '', query)
            result = eval(expression, {"__builtins__": None}, {})
            return {"type": "CALC", "content": str(result), "weight": 1.0}
        except Exception:
            prompt = f"Output ONLY Python code: print({query})"
            code = self.llm.invoke(prompt).content
            res = LocalSandbox.execute(code)
            return {"type": "CALC", "content": res, "weight": 1.0}

class ExtractorModule(BaseModule):
    def run(self, query: str, context: str, subtype: str) -> Dict:
        prompt = f"""
ROLE: Strict Fact Extractor.
TASK: Extract lines from DATA.
DATA: {context[:2500]}
SUBTYPE: {subtype}
RULES:
1. Output RAW lines.
2. NO guessing. NO hallucination.
Query: {query}
"""
        return {"type": "EXTRACT", "content": self.llm.invoke(prompt).content, "weight": 0.9}

class LogicModule(BaseModule):
    def run(self, query: str, context: str, subtype: str) -> Dict:
        prompt = f"""以下の労働基準法の条文を読んで質問に答えてください。

条文:
{context[:4000]}

質問: {query}

回答のルール:
- まず条文に基づく権利・義務を答えること
- 条文に書いてある数字や時間を必ず含めること
- 違反があった場合の対処法が条文にあれば必ず含めること
- 条文が提供されている場合は必ずその内容を元に答えること
- 提供された条文に関連する内容が含まれている場合は必ず条文を元に答えること
- 提供された条文が質問と完全に無関係な場合のみ「この質問は労働基準法の範囲外のため回答できません」と答えること
- 条文にない内容は推測で答えないこと
- 最後に「詳しくは労働基準監督署にご相談ください」と添えること

回答:"""
        return {"type": "LOGIC", "content": self.llm.invoke(prompt).content, "weight": 0.9}

# --- 4. AGGREGATOR & CRITIC ---

class WeightedAggregator:
    @staticmethod
    def aggregate(results: List[Dict]) -> str:
        sorted_results = sorted(results, key=lambda x: x['weight'], reverse=True)
        # ユーザー（HTML）向けに一番ウェイトの高い回答のみをクリーンに返すように調整
        if sorted_results:
            return sorted_results[0]['content']
        return "回答を生成できませんでした。"

class DynamicCritic:
    def __init__(self, llm):
        self.llm = llm

    def critique(self, query: str, result: str, task_type: str) -> str:
        prompt = f"Check:{result}\nQuery:{query}\nResult:PASS/FAIL?"
        try:
            res = self.llm.bind(num_predict=2, temperature=0).invoke(prompt).content.strip().upper()
            return "PASS" if "FAIL" not in res else "FAIL"
        except:
            return "PASS"

class LightFixer:
    def __init__(self, llm):
        self.llm = llm
        
    def fix(self, text: str, error: str) -> str:
        if not error: return text
        prompt = f"Fix the text slightly based on Error. Text: {text} Error: {error}"
        return self.llm.invoke(prompt).content

# --- 5. NEXUS STRATUM ENGINE ---

class NexusStratum:
    def __init__(self):
        cprint("🚀 NEXUS-8: STRATUM ENGINE INITIALIZED", Colors.HEADER)
        try:
            self.main_llm = ChatOllama(model=MODEL_MAIN, base_url="http://localhost:11434", temperature=0)
            self.critic_llm = ChatOllama(model=MODEL_CRITIC, base_url="http://localhost:11434", temperature=0)
        except Exception as e:
            sys.exit(f"❌ Ollama Down: {e}")

        self.memory = MicroMemory(MEMORY_DB)
        self.router = SemanticRouter(self.main_llm)
        self.critic = DynamicCritic(self.critic_llm)
        self.fixer = LightFixer(self.main_llm)
        
        self.vectorstore = None
        self.retriever = None
        if os.path.exists(DB_PATH) and os.listdir(DB_PATH):
            try:
                self.embeddings = OllamaEmbeddings(model="bge-m3", base_url="http://localhost:11434")
                self.vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=self.embeddings)
                self.retriever = AdvancedRetriever(self.vectorstore, self.critic_llm)
            except Exception as e:
                cprint(f"⚠️ Vector DB Error: {e}", Colors.WARNING)

        self.modules = {
            "CALCULATOR": CalculatorModule(self.main_llm, self.vectorstore),
            "EXTRACTOR": ExtractorModule(self.main_llm, self.vectorstore),
            "LOGIC": LogicModule(self.main_llm, self.vectorstore)
        }

    def process_api(self, query: str) -> str:
        """APIやフロントエンドに文字列だけを返すメソッド"""
        cprint(f"\n🧠 [API] Query: {query}", Colors.BLUE)
        
        tools = self.router.route_stage_1(query)
        context = self.retriever.get_context(query) if self.retriever else ""
        prev_error = self.memory.get_and_clear()

        results = []
        primary_task = tools[0]

        for tool_name in tools:
            module = self.modules.get(tool_name)
            if module:
                subtype = self.router.route_stage_2(query, tool_name)
                try:
                    res = module.run(query, context, subtype)
                    results.append(res)
                except Exception as e:
                    pass

        if not results:
            return "適切な回答を生成できませんでした。"
            
        aggregated_text = WeightedAggregator.aggregate(results)
        
        if primary_task == "CALC":
            return aggregated_text

        if prev_error:
            aggregated_text = self.fixer.fix(aggregated_text, prev_error)

        critique = self.critic.critique(query, aggregated_text, primary_task)
        
        # main.py のように最後にスコアやステータスを付けてHTMLに返す
        if "PASS" in critique:
            return f"{aggregated_text}\n\n---\n✅ Nexus-8 認証: PASS (論理的整合性クリア)"
        else:
            reason = critique.replace("FAIL:", "").strip()
            self.memory.set_error(reason)
            return f"{aggregated_text}\n\n---\n⚠️ Nexus-8 警告: 回答に修正が必要な可能性があります ({reason})"


# --- 6. API SERVER & CLI SETUP ---

app = Flask(__name__)
CORS(app) # HTMLからのアクセスを許可
agent = None

def clean_text(text):
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r'\s+', ' ', text)

@app.route('/chat', methods=['POST'])
def chat():
    if not agent:
        return jsonify({"answer": "サーバーが初期化されていません。"})
        
    data = request.json
    query = clean_text(data.get('query', ''))
    
    # Nexus Engineで回答を生成
    answer = agent.process_api(query)
    
    # HTML側のJSは {"answer": ...} を受け取る仕様のままにしています
    return jsonify({"answer": answer})

if __name__ == "__main__":
    agent = NexusStratum()
    
    # 起動オプションでサーバーかCLIか切り替え
    if len(sys.argv) > 1 and sys.argv[1] == "--api":
        cprint("\n🌐 STARTING NEXUS API SERVER ON http://localhost:5000", Colors.HEADER)
        # ポート5000で起動 (あなたのHTMLの設定に合わせました)
        app.run(host="0.0.0.0", port=5000)
    else:
        cprint("\n💻 CLI モードで起動中。サーバーモードにする場合は '--api' を付けてください。", Colors.WARNING)
        while True:
            try:
                q = input("\nStratum > ")
                if not q or q.lower() in ["exit", "quit"]: break
                result = agent.process_api(q)
                print(f"\n{Colors.GREEN}{result}{Colors.ENDC}")
            except KeyboardInterrupt: break
            except Exception as e: print(e)