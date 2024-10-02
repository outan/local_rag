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
    
    # テーブルが存在しない場合のみ作成
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
    
    conn.commit()
    cur.close()
    conn.close()
    print("テーブルが正常に作成されました。")

if __name__ == "__main__":
    create_tables()