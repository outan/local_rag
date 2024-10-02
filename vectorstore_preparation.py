import os
import warnings
from urllib3.exceptions import NotOpenSSLWarning
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import PGVector
from transformers import logging as transformers_logging
from dotenv import load_dotenv
import psycopg2
import hashlib

# 環境変数を読み込む
load_dotenv()

# urllib3の警告を無視
warnings.filterwarnings("ignore", category=NotOpenSSLWarning)

# transformersの警告を無視
transformers_logging.set_verbosity_error()

# 環境変数から接続情報を取得
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')

CONNECTION_STRING = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

def get_file_hash(file_path):
    """ファイルのMD5ハッシュを計算する"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def get_stored_file_hashes():
    """データベースに保存されているファイルハッシュを取得する"""
    conn = psycopg2.connect(CONNECTION_STRING)
    cur = conn.cursor()
    cur.execute("SELECT file_path, file_hash FROM file_hashes")
    stored_hashes = {row[0]: row[1] for row in cur.fetchall()}
    cur.close()
    conn.close()
    return stored_hashes

def update_file_hash(file_path, file_hash):
    """ファイルハッシュをデータベースに保存または更新する"""
    conn = psycopg2.connect(CONNECTION_STRING)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO file_hashes (file_path, file_hash)
        VALUES (%s, %s)
        ON CONFLICT (file_path) DO UPDATE
        SET file_hash = EXCLUDED.file_hash
    """, (file_path, file_hash))
    conn.commit()
    cur.close()
    conn.close()

def load_and_split_text(file_path, chunk_size=500, chunk_overlap=50):
    # データの読み込み
    loader = TextLoader(file_path)
    documents = loader.load()

    # テキストの分割
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)

    # ファイル名に基づいてアクセスレベルを設定
    access_level = "confidential" if "confidential" in os.path.basename(file_path).lower() else "general"
    for text in texts:
        text.metadata["access_level"] = access_level

    return texts

def prepare_vectorstore(folder_path):
    updated_texts = []
    stored_hashes = get_stored_file_hashes()
    updated_files = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.txt') or filename.endswith('.md'):
            file_path = os.path.join(folder_path, filename)
            current_hash = get_file_hash(file_path)
            
            if file_path not in stored_hashes or stored_hashes[file_path] != current_hash:
                texts = load_and_split_text(file_path)
                updated_texts.extend(texts)
                update_file_hash(file_path, current_hash)
                updated_files.append(file_path)

    if not updated_files:
        print("変更されたファイルはありません。ベクトルストアの更新は不要です。")
        return None

    print(f"更新されたファイル: {updated_files}")

    # 埋め込みモデルの設定
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

    # 既存のベクトルストアを取得
    vectorstore = PGVector(
        connection_string=CONNECTION_STRING,
        embedding_function=embeddings,
        collection_name="your_collection_name"
    )

    # 更新されたドキュメントのみを追加
    vectorstore.add_documents(updated_texts)

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

    # ベクトルストアの作成確認
    print("ベクトルストアが正常に作成されました。")
    print(f"接続文字列: {CONNECTION_STRING}")
    print(f"コレクション名: {vectorstore.collection_name}")