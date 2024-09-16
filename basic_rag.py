import warnings
from urllib3.exceptions import NotOpenSSLWarning
from transformers import logging as transformers_logging

# 警告を無視
warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
transformers_logging.set_verbosity_error()

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import warnings
from urllib3.exceptions import NotOpenSSLWarning

# urllib3の警告を無視
warnings.filterwarnings("ignore", category=NotOpenSSLWarning)

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)

def print_bordered(title, content):
    border = "=" * 50
    print(f"\n{border}")
    print(f"{title}:")
    print(border)
    print(content)
    print(border)

def main():
    # 保存されたベクトルストアを読み込む
    vectorstore = load_vectorstore()

    # Ollamaを使用してLLMを設定（Mistralモデルを使用）
    llm = Ollama(model="mistral", temperature=0.7)

    # RAGチェーンの作成
    prompt_template = """以下の情報を参考にして、質問に日本語で答えてください。回答は必ず日本語でお願いします。
    簡潔かつ分かりやすく説明してください。専門用語は必要に応じて説明を加えてください。
    英語での回答は避け、日本語のみで回答してください。

    参考情報：
    {context}

    質問: {question}

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

    # 評価用の質問
    question = "RAGとファインチューニングの違いは何ですか？"

    # RAGを利用しない回答
    print_bordered("RAGを利用しない処理", "プロセスの詳細")
    try:
        no_rag_prompt = f"以下の質問に日本語で簡潔に答えてください。専門用語は説明を加えてください。\n\n質問: {question}\n\n"
        print("プロンプト:", no_rag_prompt)
        no_rag_answer = llm.invoke(no_rag_prompt)
        print("生成された回答:", no_rag_answer)
    except Exception as e:
        no_rag_answer = f"エラーが発生しました: {str(e)}"
        print("エラー:", no_rag_answer)

    # RAGを利用した回答
    print_bordered("RAGを利用した処理", "プロセスの詳細")
    try:
        print("検索クエリ:", question)
        rag_result = qa_chain.invoke({"query": question})
        print("\n検索された関連ドキュメント:")
        for i, doc in enumerate(rag_result['source_documents']):
            print(f"ドキュメント {i+1}:")
            print(f"  内容: {doc.page_content[:50]}...")  # 最初の50文字のみ表示
            print(f"  ...{doc.page_content[-50:]}")  # 最後の50文字のみ表示
            print(f"  メタデータ: {doc.metadata}")
            print("  ----------------------------------")  # ドキュメントとドキュメントの間に線を表示
        
        print("\n生成されたプロンプト:")
        print(PROMPT.format(context=rag_result['source_documents'], question=question))
        
        rag_answer = rag_result['result'].strip()
        print("\n生成された回答:", rag_answer)
    except Exception as e:
        rag_answer = f"エラーが発生しました: {str(e)}"
        print("エラー:", rag_answer)

    # 結果の比較
    print_bordered("RAGを利用しない結果", no_rag_answer)
    print_bordered("RAGを利用した結果", rag_answer)

if __name__ == "__main__":
    main()
