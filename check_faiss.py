import warnings
from urllib3.exceptions import NotOpenSSLWarning
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import logging as transformers_logging

# 警告を無視
warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
transformers_logging.set_verbosity_error()

# 埋め込みモデルの設定
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# vectorstoreフォルダからベクトルストアを読み込む
try:
    vectorstore = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)
    print("ベクトルストアを正常に読み込みました。")
except FileNotFoundError:
    print("エラー: vectorstoreフォルダが見つかりません。")
    print("data_preparation.pyを実行してベクトルストアを作成してください。")
    exit(1)
except Exception as e:
    print(f"エラー: ベクトルストアの読み込み中に問題が発生しました: {str(e)}")
    exit(1)

# ベクトルストアの基本情報を表示
print(f"インデックスの次元数: {vectorstore.index.d}")
print(f"インデックスに含まれるベクトル数: {vectorstore.index.ntotal}")

# ドキュメントの内容を表示
print("\nドキュメントの内容:")
for i, doc_id in enumerate(list(vectorstore.index_to_docstore_id.values())[:5]):  # 最初の5つのみ表示
    doc = vectorstore.docstore.search(doc_id)
    print(f"\nドキュメント {i + 1}:")
    print(f"  メタデータ: {doc.metadata}")
    print("  内容:")
    
    # 内容を行ごとに分割して表示（最初の100文字のみ）
    content = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
    lines = content.split('\n')
    for line in lines:
        if line.strip():  # 空行を無視
            print(f"    {line.strip()}")
    
    print("-" * 50)  # ドキュメント間の区切り線