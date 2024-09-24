import warnings
from urllib3.exceptions import NotOpenSSLWarning
from transformers import logging as transformers_logging
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_community.vectorstores import PGVector
from langchain_huggingface import HuggingFaceEmbeddings

# 警告を無視
warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
transformers_logging.set_verbosity_error()

CONNECTION_STRING = "postgresql://dan.w@localhost:5432/rag_test"

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
        no_rag_answer = llm.invoke(no_rag_prompt)
        print("生成された回答:", no_rag_answer)
        return no_rag_answer
    except Exception as e:
        error_message = f"エラーが発生しました: {str(e)}"
        print("エラー:", error_message)
        return error_message

def get_rag_answer(qa_chain, query):
    print_bordered("RAGを利用した処理", "プロセスの詳細")
    try:
        print("クエリ:", query)
        rag_result = qa_chain.invoke({"query": query})
        print("\n検索された関連ドキュメント:")
        for i, doc in enumerate(rag_result['source_documents']):
            print(f"ドキュメント {i+1}:")
            print(f"  内容: {doc.page_content[:50]}...")
            print(f"  ...{doc.page_content[-50:]}")
            print(f"  メタデータ: {doc.metadata}")
            print("  " + "-" * 50)
        rag_answer = rag_result['result'].strip()
        print("\n生成された回答:", rag_answer)
        return rag_answer
    except Exception as e:
        error_message = f"エラーが発生しました: {str(e)}"
        print("エラー:", error_message)
        return error_message

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

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    query = "RAGとファインチューニングの違いは何ですか？"

    no_rag_answer = get_no_rag_answer(llm, query)
    rag_answer = get_rag_answer(qa_chain, query)

    # print_bordered("RAGを利用しない回答", no_rag_answer)
    # print_bordered("RAGを利用した回答", rag_answer)

if __name__ == "__main__":
    main()
