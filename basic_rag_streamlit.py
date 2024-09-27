import streamlit as st
import warnings
from urllib3.exceptions import NotOpenSSLWarning
from transformers import logging as transformers_logging
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_community.vectorstores import PGVector
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import time
import os

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

CONNECTION_STRING = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return PGVector(
        connection_string=CONNECTION_STRING,
        embedding_function=embeddings,
        collection_name="your_collection_name"
    )

def get_no_rag_answer(llm, query, combined_handler, chat_history):
    no_rag_prompt = """以下のクエリに日本語で簡潔に答えてください。専門用語は説明を加えてください。
    回答は必ず日本語でお願いします。英語での回答は避け、日本語のみで回答してください。
    
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

def generate_response(query, chat_history, vectorstore, llm, combined_handler):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # ベクトル変換の時間計測
    start_time = time.time()
    query_vector = vectorstore.embedding_function.embed_query(query)
    vector_time = time.time() - start_time
    
    # ベクターデータベースからの関連情報検索の時間計測
    start_time = time.time()
    docs = retriever.get_relevant_documents(query)
    retrieval_time = time.time() - start_time
    
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
    result = chain({"question": query, "chat_history": chat_history}, callbacks=[combined_handler])
    llm_time = time.time() - start_time
    
    return combined_handler.text, result['source_documents'], vector_time, retrieval_time, llm_time

def main():
    st.set_page_config(layout="wide")  # ページ全体を広く使用
    
    st.title("RAGシステム")

    # セッション状態の初期化
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'llm_history' not in st.session_state:
        st.session_state.llm_history = []
    if 'query' not in st.session_state:
        st.session_state.query = ""
    if 'processing_times' not in st.session_state:
        st.session_state.processing_times = []

    vectorstore = load_vectorstore()
    llm = Ollama(model="mistral", temperature=0.7)

    # 会話履歴の表示
    for i, (query, no_rag_answer, rag_answer) in enumerate(st.session_state.conversation_history):
        st.subheader(f"質問 {i+1}")
        st.write(query)
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
                    
                    no_rag_answer, no_rag_time = get_no_rag_answer(llm, query, no_rag_combined_handler, st.session_state.llm_history)
                    
                    st.markdown("**処理時間:**")
                    st.write(f"回答生成: {no_rag_time:.4f}秒")
                    st.write("ベクトル変換: 0.0000秒")
                    st.write("関連情報検索: 0.0000秒")
                    
                    st.markdown("---")
                    
                    # RAGを利用する回答のストリーミング
                    st.markdown("**RAGを利用した回答：**")
                    rag_container = st.empty()
                    rag_streamlit_handler = StreamHandler(rag_container)
                    rag_stdout_handler = StreamingStdOutCallbackHandler()
                    rag_combined_handler = CombinedStreamHandler(rag_streamlit_handler, rag_stdout_handler)
                    
                    rag_answer, source_docs, vector_time, retrieval_time, llm_time = generate_response(query, st.session_state.llm_history, vectorstore, llm, rag_combined_handler)
                    
                    st.markdown("**処理時間:**")
                    st.write(f"ベクトル変換: {vector_time:.4f}秒")
                    st.write(f"関連情報検索: {retrieval_time:.4f}秒")
                    st.write(f"回答生成: {llm_time:.4f}秒")
                    
                    st.markdown("---")
                    
                    # 参照元の表示
                    st.subheader("参照元:")
                    for doc in source_docs:
                        st.write(doc.metadata['source'])

            # 会話履歴と処理時間をセッション状態に追加
            st.session_state.conversation_history.append((query, no_rag_answer, rag_answer))
            st.session_state.llm_history.append((query, rag_answer))
            st.session_state.processing_times.append((no_rag_time, vector_time, retrieval_time, llm_time))
        
        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")
            print(f"詳細なエラー情報: {e}")  # デバッグ用にコンソールに詳細を出力
        
        finally:
            # 入力欄をクリア
            st.session_state.query = ""
        
        # ページを再読み込みして履歴を更新
        st.rerun()

if __name__ == "__main__":
    main()