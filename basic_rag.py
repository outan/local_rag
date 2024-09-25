import warnings
from urllib3.exceptions import NotOpenSSLWarning
from transformers import logging as transformers_logging
from langchain.chains import RetrievalQAWithSourcesChain, RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_community.vectorstores import PGVector
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
import sys

# 環境変数を読み込む
load_dotenv()

# 警告を無視
warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
transformers_logging.set_verbosity_error()

# 環境変数から接続情報を取得
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')

CONNECTION_STRING = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return PGVector(connection_string=CONNECTION_STRING, embedding_function=embeddings, collection_name="your_collection_name")

def print_bordered(title, content):
    border = "=" * 50
    print(f"\n{border}")
    print(f"{title}:")
    print(border)
    print(content)
    print(border)

def get_no_rag_answer(llm, query):
    print_bordered("RAGを利用しない処理", "プロセスの詳細")
    try:
        no_rag_prompt = f"以下のクエリに日本語で簡潔に答えてください。専門用語は説明を加えてください。\n\nクエリ: {query}\n\n"
        print("プロンプト:", no_rag_prompt)
        for chunk in llm.stream(no_rag_prompt):
            print(chunk, end='', flush=True)
        print()
    except Exception as e:
        error_message = f"エラーが発生しました: {str(e)}"
        print("エラー:", error_message)

def get_rag_answer(qa_chain, retriever, query):
    print_bordered("RAGを利用した処理", "プロセスの詳細")
    try:
        print("クエリ:", query)
        docs = retriever.invoke(query)
        print("\n検索された関連ドキュメント:")
        for i, doc in enumerate(docs):
            print(f"ドキュメント {i+1}:")
            print(f"  内容: {doc.page_content[:50]}...")
            print(f"  ...{doc.page_content[-50:]}")
            print(f"  メタデータ: {doc.metadata}")
            print("  " + "-" * 50)
        print("\n生成された回答:")
        context = "\n\n".join([doc.page_content for doc in docs])
        for chunk in qa_chain.llm_chain.llm.stream(qa_chain.llm_chain.prompt.format(context=context, question=query)):
            print(chunk, end='', flush=True)
        print()
    except Exception as e:
        error_message = f"エラーが発生しました: {str(e)}"
        print("エラー:", error_message)

def main():
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

    from langchain.chains import LLMChain
    from langchain.chains.combine_documents.stuff import StuffDocumentsChain

    llm_chain = LLMChain(llm=llm, prompt=PROMPT)
    qa_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="context",
        verbose=True
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    query = "RAGとファインチューニングの違いは何ですか？"

    get_no_rag_answer(llm, query)
    get_rag_answer(qa_chain, retriever, query)

if __name__ == "__main__":
    main()
