import psycopg2
from dotenv import load_dotenv
import os

# 環境変数を読み込む
load_dotenv()

# 環境変数から接続情報を取得
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')

CONNECTION_STRING = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

def create_tables():
    conn = psycopg2.connect(CONNECTION_STRING)
    cur = conn.cursor()
    
    tables_created = []

    # conversation_historyテーブルの作成
    cur.execute("""
    CREATE TABLE IF NOT EXISTS conversation_history (
        id SERIAL PRIMARY KEY,
        rag_conversation_history JSON,
        no_rag_conversation_history JSON,
        processing_times JSON,
        user_role VARCHAR(50),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    tables_created.append("conversation_history")
    
    # file_hashesテーブルの作成
    cur.execute("""
    CREATE TABLE IF NOT EXISTS file_hashes (
        file_path TEXT PRIMARY KEY,
        file_hash TEXT NOT NULL,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    tables_created.append("file_hashes")
    
    conn.commit()
    cur.close()
    conn.close()

    # 作成されたテーブルの一覧を表示
    for table in tables_created:
        print(f"{table}テーブルが正常に作成されました。")

if __name__ == "__main__":
    create_tables()