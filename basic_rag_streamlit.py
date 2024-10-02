import streamlit as st
import warnings
from urllib3.exceptions import NotOpenSSLWarning
from transformers import logging as transformers_logging
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_community.vectorstores import PGVector
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import time
import os
import requests
import psycopg2
import json

# 警告を無視
warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
transformers_logging.set_verbosity_error()

# 環境変数を読み込む
load_dotenv()

# 環境変数から接続情報を取得
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', "http://localhost:11434")

CONNECTION_STRING = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    return PGVector(
        connection_string=CONNECTION_STRING,
        embedding_function=embeddings,
        collection_name="your_collection_name"
    )

def get_no_rag_answer(llm, query, combined_handler, chat_history):
    no_rag_prompt = """以下のクエリに日本語で簡潔に答えてください。専門用語は説明を加えてください。
    回答は必ず日本語でお願いします。英語での回答は避け、日本語のみで答してください。
    
    これまでの会話履歴:
    {chat_history}

    クエリ: {query}
    """
    start_time = time.time()
    formatted_chat_history = "\n".join([f"Human: {h[0]}\nAI: {h[1]}" for h in chat_history])
    llm(no_rag_prompt.format(query=query, chat_history=formatted_chat_history), callbacks=[combined_handler])
    no_rag_time = time.time() - start_time
    return combined_handler.text, no_rag_time

def get_rag_answer(qa_chain, retriever, query, chat_history):
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    return qa_chain.llm_chain.llm(qa_chain.llm_chain.prompt.format(context=context, question=query, chat_history=chat_history))

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, prefix=""):
        self.container = container
        self.text = ""
        self.prefix = prefix

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(f"{self.prefix}\n\n{self.text}")

class CombinedStreamHandler(BaseCallbackHandler):
    def __init__(self, streamlit_handler, stdout_handler):
        self.streamlit_handler = streamlit_handler
        self.stdout_handler = stdout_handler
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.streamlit_handler.on_llm_new_token(token, **kwargs)
        self.stdout_handler.on_llm_new_token(token, **kwargs)
        self.text += token

def get_chunks(query, vectorstore):
    # ベクトル変換の時間計測
    start_time = time.time()
    query_vector = vectorstore.embedding_function.embed_query(query)
    vector_time = time.time() - start_time
    
    # ベクターデータベースからの関連情報検索の時間計測
    start_time = time.time()
    docs_and_scores = vectorstore.similarity_search_with_score(query, k=3)
    retrieval_time = time.time() - start_time
    
    # 取得されたチャンクの内容とスコアを保存（最大3件まで）
    retrieved_chunks = [(doc.page_content, score) for doc, score in docs_and_scores]
    print(f"Retrieved chunks: {retrieved_chunks}")  # デバッグ用
    print(f"Number of retrieved chunks: {len(retrieved_chunks)}")  # デバッグ用
    
    return retrieved_chunks, vector_time, retrieval_time, [doc for doc, _ in docs_and_scores]

def generate_response(query, chat_history, vectorstore, llm, combined_handler, retrieved_chunks):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    prompt_template = """以下の情報を参考にして、クエリに日本語で答えてください。回答は必ず日本語でお願いします。
    簡潔かつ分かりやすく説明してください。専門用語は必要に応じて説明を加えてください。
    英語での回答は避け、日本語のみで回答してください。

    これまでの会話履歴:
    {chat_history}

    参考情報：
    {context}

    クエリ: {question}
    """
    
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question", "chat_history"]
    )
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": PROMPT}
    )

    # LLMでの回答生成の時間計測
    start_time = time.time()
    result = chain({"question": query, "chat_history": chat_history, "context": "\n\n".join([chunk for chunk, _ in retrieved_chunks])}, callbacks=[combined_handler])
    llm_time = time.time() - start_time
    
    return combined_handler.text, result['source_documents'], llm_time

def get_available_models():
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags")
        if response.status_code == 200:
            models = response.json()["models"]
            return [model["name"] for model in models]
        else:
            st.error(f"Ollamaからモデルリストを取得できませんでした。ステータスコード: {response.status_code}")
            return []
    except Exception as e:
        st.error(f"Ollamaからモデルリストを取得できませんでした: {str(e)}")
        return []

# データベース接続関数
def get_db_connection():
    return psycopg2.connect(CONNECTION_STRING)

# 会話履歴をデータベースに保存する関数を修正
def save_conversation_history(rag_conversation_history, no_rag_conversation_history, processing_times):
    conn = get_db_connection()
    cur = conn.cursor()
    # 既存のデータを削除
    cur.execute("DELETE FROM conversation_history")
    # 新しいデータを挿入
    cur.execute("""
        INSERT INTO conversation_history (rag_conversation_history, no_rag_conversation_history, processing_times)
        VALUES (%s, %s, %s)
    """, (json.dumps(rag_conversation_history), json.dumps(no_rag_conversation_history), json.dumps(processing_times)))
    conn.commit()
    cur.close()
    conn.close()

# データベースから会話履歴を読み込む関数を修正
def load_conversation_history():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT rag_conversation_history, no_rag_conversation_history, processing_times FROM conversation_history ORDER BY id DESC LIMIT 1")
    result = cur.fetchone()
    cur.close()
    conn.close()
    if result:
        return (
            json.loads(result[0]) if isinstance(result[0], str) else result[0],
            json.loads(result[1]) if isinstance(result[1], str) else result[1],
            json.loads(result[2]) if isinstance(result[2], str) else result[2]
        )
    return [], [], []

# llm_historyを動的に生成する関数を修正
def generate_llm_history(conversation_history, use_rag=True):
    if use_rag:
        return [(query, answer) for query, answer in conversation_history]
    else:
        return [(query, answer) for query, answer in conversation_history]

def main():
    st.set_page_config(layout="wide")  # ページ全体を広く使用
    
    st.title("RAGシステム")

    # セッション状態の初期化を修正
    if 'rag_conversation_history' not in st.session_state or 'no_rag_conversation_history' not in st.session_state or 'processing_times' not in st.session_state:
        st.session_state.rag_conversation_history, st.session_state.no_rag_conversation_history, st.session_state.processing_times = load_conversation_history()
    if 'query' not in st.session_state:
        st.session_state.query = ""
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = None

    # 利用可能なモデルのリストを取得
    available_models = get_available_models()

    if not available_models:
        st.warning("利用可能なLLMモデルが見つかりません。Ollamaを使用してローカルにモデルをダウンロードしてください。")
        st.stop()  # これ以降の処理を停止

    # 初回実行時または選択されたモデルが利用可能でない場合、最初のモデルを選択
    if st.session_state.selected_model not in available_models:
        st.session_state.selected_model = available_models[0]

    # サイドバーにモデル選択ウィジェットとクリアボタンを追加
    st.sidebar.title("LLMモデル選択")
    selected_model = st.sidebar.selectbox(
        "使用するLLMモデルを選択してください：",
        options=available_models,
        index=available_models.index(st.session_state.selected_model)
    )

    # クリアボタンを追加
    if st.sidebar.button("会話履歴をクリア"):
        st.session_state.rag_conversation_history = []
        st.session_state.no_rag_conversation_history = []
        st.session_state.processing_times = []
        save_conversation_history([], [], [])
        st.rerun()

    vectorstore = load_vectorstore()
    
    llm = Ollama(model=st.session_state.selected_model, temperature=0.7, base_url=OLLAMA_BASE_URL)

    # 会話履歴の表示を修正
    for i in range(max(len(st.session_state.rag_conversation_history), len(st.session_state.no_rag_conversation_history))):
        st.subheader(f"質問 {i+1}")
        
        rag_query, rag_answer = st.session_state.rag_conversation_history[i] if i < len(st.session_state.rag_conversation_history) else ("", "")
        no_rag_query, no_rag_answer = st.session_state.no_rag_conversation_history[i] if i < len(st.session_state.no_rag_conversation_history) else ("", "")
        
        st.write(rag_query or no_rag_query)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**RAGを利用しない回答**")
            st.write(no_rag_answer)
            if i < len(st.session_state.processing_times):
                no_rag_time, _, _, _ = st.session_state.processing_times[i]
                st.markdown("**処理時間:**")
                st.write(f"回答生成: {no_rag_time:.4f}秒")
                st.write("ベクトル変換: 0.0000秒")
                st.write("関連情報検索: 0.0000秒")
        with col2:
            st.markdown("**RAGを利用した回答**")
            st.write(rag_answer)
            if i < len(st.session_state.processing_times):
                _, vector_time, retrieval_time, llm_time = st.session_state.processing_times[i]
                st.markdown("**処理時間:**")
                st.write(f"ベクトル変換: {vector_time:.4f}秒")
                st.write(f"関連情報検索: {retrieval_time:.4f}秒")
                st.write(f"回答生成: {llm_time:.4f}秒")
        
        st.markdown("---")

    # ストリーミング出力用のコンテナ
    stream_container = st.container()

    # 送信ボタンを入力欄の右に配置
    input_container = st.container()
    with input_container:
        col1, col2, col3 = st.columns([20, 4, 1])
        with col1:
            query = st.text_input("新しい質問を入力してください：", key="query_input", value=st.session_state.query, label_visibility="collapsed")
        with col2:
            submit_button = st.button("送信", key="submit", use_container_width=True)
        with col3:
            st.write("")  # 空のカラムでバランスを取る

    if submit_button and query:
        try:
            with st.spinner("回答を生成中..."):
                with stream_container:
                    # RAGを利用しない回答のストリーミング
                    st.markdown("**RAGを利用しない回答：**")
                    no_rag_container = st.empty()
                    no_rag_streamlit_handler = StreamHandler(no_rag_container)
                    no_rag_stdout_handler = StreamingStdOutCallbackHandler()
                    no_rag_combined_handler = CombinedStreamHandler(no_rag_streamlit_handler, no_rag_stdout_handler)
                    
                    no_rag_llm_history = generate_llm_history(st.session_state.no_rag_conversation_history, use_rag=False)
                    no_rag_answer, no_rag_time = get_no_rag_answer(llm, query, no_rag_combined_handler, no_rag_llm_history)
                    
                    st.markdown("**処理時間:**")
                    st.write(f"回答生成: {no_rag_time:.4f}秒")
                    st.write("ベクトル変換: 0.0000秒")
                    st.write("関連情報検索: 0.0000秒")
                    
                    st.markdown("---")
                    
                    # チャンクの取得
                    retrieved_chunks, vector_time, retrieval_time, source_docs = get_chunks(query, vectorstore)
                    
                    # RAGを利用する回答のストリーミング
                    st.markdown("**RAGを利用した回答：**")
                    rag_container = st.empty()
                    rag_streamlit_handler = StreamHandler(rag_container)
                    rag_stdout_handler = StreamingStdOutCallbackHandler()
                    rag_combined_handler = CombinedStreamHandler(rag_streamlit_handler, rag_stdout_handler)
                    
                    rag_llm_history = generate_llm_history(st.session_state.rag_conversation_history, use_rag=True)
                    rag_answer, _, llm_time = generate_response(query, rag_llm_history, vectorstore, llm, rag_combined_handler, retrieved_chunks)
                    
                    st.markdown("**処理時間:**")
                    st.write(f"ベクトル変換: {vector_time:.4f}秒")
                    st.write(f"関連情報検索: {retrieval_time:.4f}秒")
                    st.write(f"回答生成: {llm_time:.4f}秒")
                    
                    st.markdown("---")
                    
                    # チャンクの表示
                    st.subheader("取得されたチャンク:")
                    for i, (chunk, score) in enumerate(retrieved_chunks):
                        st.markdown(f"**チャンク {i+1}:**")
                        st.markdown(f"<div style='background-color: #f0f2f6; padding: 10px; border-radius: 5px;'>{chunk}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div style='background-color: #e6f3ff; padding: 5px; border-radius: 5px; display: inline-block; margin-top: 5px;'><strong>Similarity Score:</strong> <span style='color: #0066cc; font-size: 1.2em;'>{score:.4f}</span></div>", unsafe_allow_html=True)
                        st.markdown("---")

                # 参照元の表示
                st.subheader("参照元:")
                for doc in source_docs:
                    st.write(doc.metadata['source'])

            # 会話履歴と処理時間をセッション状態に追加
            st.session_state.rag_conversation_history.append((query, rag_answer))
            st.session_state.no_rag_conversation_history.append((query, no_rag_answer))
            st.session_state.processing_times.append((no_rag_time, vector_time, retrieval_time, llm_time))
            
            # データベースに最新の履歴のみを保存
            save_conversation_history(st.session_state.rag_conversation_history, st.session_state.no_rag_conversation_history, st.session_state.processing_times)
        
        except Exception as e:
            st.error(f"エラが発生しました: {str(e)}")
            print(f"詳細なエラー情報: {e}")  # デバッグ用にコンソールに詳細を出力
        
        finally:
            # 入力欄をクリア
            st.session_state.query = ""

if __name__ == "__main__":
    main()