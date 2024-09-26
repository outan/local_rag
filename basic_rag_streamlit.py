import streamlit as st
import warnings
from urllib3.exceptions import NotOpenSSLWarning
from transformers import logging as transformers_logging
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_community.vectorstores import PGVector
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
from langchain.chains import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

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

def get_no_rag_answer(llm, query, combined_handler):
    no_rag_prompt = """以下のクエリに日本語で簡潔に答えてください。専門用語は説明を加えてください。
    回答は必ず日本語でお願いします。英語での回答は避け、日本語のみで回答してください。

    クエリ: {query}
    """
    llm(no_rag_prompt.format(query=query), callbacks=[combined_handler])
    return combined_handler.streamlit_handler.text

def get_rag_answer(qa_chain, retriever, query):
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    return qa_chain.llm_chain.llm(qa_chain.llm_chain.prompt.format(context=context, question=query))

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, prefix=""):
        self.container = container
        self.text = prefix
        self.prefix = prefix

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

class CombinedStreamHandler(BaseCallbackHandler):
    def __init__(self, streamlit_handler, stdout_handler):
        self.streamlit_handler = streamlit_handler
        self.stdout_handler = stdout_handler

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.streamlit_handler.on_llm_new_token(token, **kwargs)
        self.stdout_handler.on_llm_new_token(token, **kwargs)

def generate_response(query, chat_history, vectorstore, llm, combined_handler):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    prompt_template = """以下の情報を参考にして、クエリに日本語で答えてください。回答は必ず日本語でお願いします。
    簡潔かつ分かりやすく説明してください。専門用語は必要に応じて説明を加えてください。
    英語での回答は避け、日本語のみで回答してください。

    参考情報：
    {context}

    クエリ: {question}
    """
    
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": PROMPT}
    )

    result = chain({"question": query, "chat_history": chat_history}, callbacks=[combined_handler])
    
    return combined_handler.streamlit_handler.text, result['source_documents']

def main():
    st.title("RAGシステム")

    # セッション状態の初期化
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

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
        with col2:
            st.markdown("**RAGを利用した回答**")
            st.write(rag_answer)
        st.markdown("---")

    # 新しい質問の入力
    query = st.text_input("新しい質問を入力してください：")

    if st.button("回答を生成"):
        if query:
            with st.spinner("回答を生成中..."):
                # RAGを利用しない回答のストリーミング
                no_rag_container = st.empty()
                no_rag_streamlit_handler = StreamHandler(no_rag_container, prefix="RAGを利用しない回答：\n")
                no_rag_stdout_handler = StreamingStdOutCallbackHandler()
                no_rag_combined_handler = CombinedStreamHandler(no_rag_streamlit_handler, no_rag_stdout_handler)
                
                no_rag_answer = get_no_rag_answer(llm, query, no_rag_combined_handler)
                
                # RAGを利用する回答のストリーミング
                rag_container = st.empty()
                rag_streamlit_handler = StreamHandler(rag_container, prefix="RAGを利用した回答：\n")
                rag_stdout_handler = StreamingStdOutCallbackHandler()
                rag_combined_handler = CombinedStreamHandler(rag_streamlit_handler, rag_stdout_handler)
                
                rag_answer, source_docs = generate_response(query, [], vectorstore, llm, rag_combined_handler)
                
                # 会話履歴に追加
                st.session_state.conversation_history.append((query, no_rag_answer, rag_answer))
                
                # 参照元の表示
                st.subheader("参照元:")
                for doc in source_docs:
                    st.write(doc.metadata['source'])
            
            # ページを再読み込みして履歴を更新
            st.rerun()
        else:
            st.warning("質問を入力してください。")

if __name__ == "__main__":
    main()