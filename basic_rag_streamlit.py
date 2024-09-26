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

def get_no_rag_answer(llm, query):
    no_rag_prompt = f"以下のクエリに日本語で簡潔に答えてください。専門用語は説明を加えてください。\n\nクエリ: {query}\n\n"
    return llm(no_rag_prompt)

def get_rag_answer(qa_chain, retriever, query):
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    return qa_chain.llm_chain.llm(qa_chain.llm_chain.prompt.format(context=context, question=query))

def main():
    st.title("RAGシステム")

    # セッション状態の初期化
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    vectorstore = load_vectorstore()
    llm = Ollama(model="mistral", temperature=0.7)

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

    llm_chain = LLMChain(llm=llm, prompt=PROMPT)
    qa_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="context",
        verbose=True
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

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
                no_rag_answer = get_no_rag_answer(llm, query)
                rag_answer = get_rag_answer(qa_chain, retriever, query)
                
                # 会話履歴に追加
                st.session_state.conversation_history.append((query, no_rag_answer, rag_answer))
                
                # 最新の回答を表示
                st.subheader("最新の回答")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**RAGを利用しない回答**")
                    st.write(no_rag_answer)
                with col2:
                    st.markdown("**RAGを利用した回答**")
                    st.write(rag_answer)
            
            # ページを再読み込みして履歴を更新
            st.rerun()
        else:
            st.warning("質問を入力してください。")

if __name__ == "__main__":
    main()