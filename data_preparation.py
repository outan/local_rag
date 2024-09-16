import os
import warnings
from urllib3.exceptions import NotOpenSSLWarning
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import logging as transformers_logging

# urllib3の警告を無視
warnings.filterwarnings("ignore", category=NotOpenSSLWarning)

# transformersの警告を無視
transformers_logging.set_verbosity_error()

def load_and_split_text(file_path, chunk_size=500, chunk_overlap=50):
    # データの読み込み
    loader = TextLoader(file_path)
    documents = loader.load()

    # テキストの分割
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)

    return texts

def prepare_vectorstore(folder_path):
    all_texts = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt') or filename.endswith('.md'):
            file_path = os.path.join(folder_path, filename)
            texts = load_and_split_text(file_path)
            all_texts.extend(texts)

    # 埋め込みモデルの設定
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # ベクトルをベクトルデータベースに保存
    vectorstore = FAISS.from_documents(all_texts, embeddings)

    return vectorstore

if __name__ == "__main__":
    import sys
    
    # デフォルトのフォルダパスを設定
    default_folder = "data"
    
    if len(sys.argv) < 2:
        input_folder = default_folder
        print(f"入力フォルダが指定されていません。デフォルトの{default_folder}を使用します。")
    else:
        input_folder = sys.argv[1]
    
    # フォルダの存在確認
    if not os.path.exists(input_folder):
        print(f"エラー: フォルダ '{input_folder}' が見つかりません。")
        sys.exit(1)
    
    print(f"{input_folder}内のファイルからベクトルストアを準備します...")
    vectorstore = prepare_vectorstore(input_folder)
    print("ベクトルストアの準備が完了しました。")

    # ベクトルストアを保存
    output_folder = "vectorstore"
    print(f"ベクトルストアを{output_folder}フォルダに保存します...")
    vectorstore.save_local(output_folder)
    print(f"ベクトルストアが'{output_folder}'フォルダに保存されました。")