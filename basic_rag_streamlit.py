import streamlit as st
import warnings
from urllib3.exceptions import NotOpenSSLWarning
from transformers import logging as transformers_logging
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_community.vectorstores import PGVector
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import time
import os
import requests
import psycopg2
import json
import re
from sentence_transformers import CrossEncoder

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
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', "http://localhost:11434")

CONNECTION_STRING = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# デバッグモードの設定
DEBUG_MODE = os.getenv('DEBUG_MODE', 'False').lower() == 'true'

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    return PGVector(
        connection_string=CONNECTION_STRING,
        embedding_function=embeddings,
        collection_name="your_collection_name"
    )

# vectorstoreをグローバル変数として初期化
vectorstore = load_vectorstore()

def split_query(query):
    llm = Ollama(model=st.session_state.selected_model, temperature=0.7, base_url=OLLAMA_BASE_URL)
    split_prompt = PromptTemplate(
        input_variables=["query"],
        template="""
        以下のクエリを、必要最小限のサブクエリに分割してください。以下の点に注意してください：

        1. サブクエリは3つ以下にしてください。ただし、必要な場合のみ3つ使用してください。
        2. サブクエリはSQLクエリではなく、必ず自然言語の質問形式で作成してください。
        3. 他のサブクエリの結果から容易に計算や推論できる情報は、別のサブクエリとして含めないでください。
        4. 各サブクエリは、元のクエリに答えるために必要不可欠な情報を取得するものだけにしてください。

        例1：
        元のクエリ: "iphone 16とiphone 16 plusは重さの差分は？"
        適切なサブクエリ:
        1. "iphone 16の重さは？"
        2. "iphone 16 plusの重さは？"
        （注：重さの差分を計算するサブクエリは不要です。これは結果から容易に計算できるためです。）

        例2：
        元のクエリ: "2023年のiPhone、iPad、Macの売上高の合計は？"
        適切なサブクエリ:
        1. "2023年のiPhoneの売上高は？"
        2. "2023年のiPadの売上高は？"
        3. "2023年のMacの売上高は？"
        （注：この場合、3つのサブクエリが必要です。）

        重要: 回答は必ず以下の形式の有効なJSONのみで返してください。余分なテキストや説明は一切含めないでください。
        全ての文字列は必ず半角の二重引用符（"）で囲んでください。
        全角の括弧（「」）や他の全角文字は使用しないでください。

        正しい形式の例：
        {{"subqueries": ["サブクエリ1", "サブクエリ2", "サブクエリ3"]}}
        または
        {{"subqueries": ["サブクエリ1", "サブクエリ2"]}}

        クエリ: {query}
        """
    )
    chain = LLMChain(llm=llm, prompt=split_prompt)
    result = chain.run(query)
    
    # LLMの出力を前処理
    result = result.replace('」', '"').replace('「', '"')
    
    try:
        parsed_result = json.loads(result)
        if "subqueries" in parsed_result and isinstance(parsed_result["subqueries"], list):
            # SQLクエリでないことを確認
            if not any("SELECT" in subquery.upper() for subquery in parsed_result["subqueries"]):
                print(f"生成されたサブクエリ: {parsed_result['subqueries']}")
                return parsed_result["subqueries"]
            else:
                print("SQLクエリが含まれています。元のクエリを使用します。")
                return [query]
    except json.JSONDecodeError:
        print(f"JSONのパースに失敗しました。LLMの出力: {result}")
        # JSONの部分を抽出しようとする
        json_match = re.search(r'\{.*\}', result, re.DOTALL)
        if json_match:
            try:
                # 抽出されたJSONも前処理
                json_str = json_match.group().replace('」', '"').replace('「', '"')
                parsed_result = json.loads(json_str)
                if "subqueries" in parsed_result and isinstance(parsed_result["subqueries"], list):
                    # SQLクエリでないことを確認
                    if not any("SELECT" in subquery.upper() for subquery in parsed_result["subqueries"]):
                        return parsed_result["subqueries"]
                    else:
                        print("SQLクエリが含まれています。元のクエリを使用します。")
                        return [query]
            except json.JSONDecodeError:
                print(f"抽出されたJSONのパースにも失敗しました: {json_str}")
    
    print("有効なサブクエリを生成できませんでした。元のクエリを使用します。")
    return [query]  # 元のクエリをそのまま使用

def get_filtered_chunks(subquery, user_role, initial_k=10, final_k=3):
    if user_role == "manager":
        docs_and_scores = vectorstore.similarity_search_with_score(subquery, k=initial_k)
    else:
        docs_and_scores = vectorstore.similarity_search_with_score(
            subquery,
            k=initial_k,
            filter={"access_level": "general"}
        )
    
    # 初期のチャンクとスコアのリストを作成
    initial_chunks = [(doc.page_content, 1 - score) for doc, score in docs_and_scores]  # スコアを1 - scoreに変換
    
    # rerankモデルを使用してチャンクを並び替え
    reranked_chunks = rerank_chunks(subquery, initial_chunks)
    
    # 上位final_k個のチャンクを選択
    filtered_chunks = reranked_chunks[:final_k]
    
    if DEBUG_MODE:
        print(f"サブクエリ '{subquery}' に対する取得チャンク: {filtered_chunks}")
        print(f"取得チャンク数: {len(filtered_chunks)}")
    
    return filtered_chunks

def rerank_chunks(query, chunks, model_name='cross-encoder/ms-marco-MiniLM-L-12-v2'):
    reranker = CrossEncoder(model_name)
    pairs = [[query, chunk] for chunk, _ in chunks]
    scores = reranker.predict(pairs)
    
    # スコアを0-1の範囲に正規化
    min_score, max_score = min(scores), max(scores)
    normalized_scores = [(score - min_score) / (max_score - min_score) for score in scores]
    
    reranked_chunks = [(chunk, float(score)) for (chunk, _), score in zip(chunks, normalized_scores)]
    reranked_chunks.sort(key=lambda x: x[1], reverse=True)
    return reranked_chunks

def process_subquery(subquery, context, llm):
    response_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        以下の情報を参考にして、質問に日本語で答えてください。
        簡潔かつ分かりやすく説明してください。
        専門用語は必要に応じて説明を加えてください。

        参考情報：
        {context}

        質問: {question}
        """
    )
    chain = LLMChain(llm=llm, prompt=response_prompt)
    response = chain.run({"context": context, "question": subquery})
    return response

def combine_subquery_responses(responses, query):
    llm = Ollama(model=st.session_state.selected_model, temperature=0.7, base_url=OLLAMA_BASE_URL)
    combine_prompt = PromptTemplate(
        input_variables=["responses", "query"],
        template="""
        以下の情報を使用して、元の質問に対する包括的な回答を生成してください。
        ステップバイステップで考え、最終的な回答を導き出してください。

        元の質問: {query}

        サブクエリの回答:
        {responses}

        ステップ：
        1. 各サブクエリの回答を要約し、重要なポイントを抽出してください。
        2. 抽出したポイント間の関連性や矛盾点を分析してください。
        3. 元の質問に関連する情報を整理し、論理的な順序で並べてください。
        4. 不足している情報があれば、それを指摘してください。
        5. 上記の分析を基に、最終的な回答を作成してください。

        それでは、以下の形式で回答を作成してください：

        要約と重要ポイント：
        [ここに各サブクエリの回答の要約と重要ポイントを記述]

        分析：
        [ここにポイント間の関連性や矛盾点の分析を記述]

        情報の整理：
        [ここに関連情報を論理的順序で整理して記述]

        不足情報：
        [ここに不足している情報があれば記述]

        最終回答：
        [ここに最終的な包括的な回答を記述]
        """
    )
    chain = LLMChain(llm=llm, prompt=combine_prompt)
    combined_context = chain.run({"responses": "\n\n".join(responses), "query": query})
    return combined_context

# process_query_and_generate_final_answer関数を修正
def process_query_and_generate_final_answer(query, chat_history, llm, combined_handler, user_role, subqueries):
    start_time = time.time()
    print(f"process_query_and_generate_final_answer で使用されるサブクエリ: {subqueries}")  # デバッグ用
    subquery_responses = []
    all_retrieved_chunks = []
    total_retrieval_time = 0
    
    for subquery in subqueries:
        subquery_start_time = time.time()
        filtered_chunks = get_filtered_chunks(subquery, user_role)  # kを削除
        total_retrieval_time += time.time() - subquery_start_time
        
        all_retrieved_chunks.append((subquery, filtered_chunks))
        
        context = "\n".join([chunk for chunk, _ in filtered_chunks])
        response = process_subquery(subquery, context, llm)
        subquery_responses.append(response)
    
    combined_context = combine_subquery_responses(subquery_responses, query)
    
    prompt_template = f"""以下の情報を参考にして、クエリに日本語で答えてください。回答は必ず日本語でお願いします。
    簡潔かつ分かりやすく説明してください。専門用語は必要に応じて説明を加えてください。
    英語での回答は避け、日本語のみで回答してください。

    これまでの会話履歴:
    {{chat_history}}

    集約された情報:
    {combined_context}

    クエリ: {{question}}
    """
    
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["question", "chat_history"]
    )
    
    chain = LLMChain(llm=llm, prompt=PROMPT)

    llm_start_time = time.time()
    final_answer = chain.run({"question": query, "chat_history": chat_history}, callbacks=[combined_handler])
    llm_time = time.time() - llm_start_time
    
    total_time = time.time() - start_time
    
    return combined_handler.text, total_retrieval_time, llm_time, total_time, all_retrieved_chunks

def get_no_rag_answer(llm, query, combined_handler, chat_history):
    no_rag_prompt = """以下のクエリに日本語で答えてください。以下の点に注意してください：

    1. 回答は必ず日本語でお願いします。英語での回答は避けてください。
    2. 情報の正確性と忠実性を最優先してください。
    3. 確実に知っている情報のみを含めてください。
    4. 不確かな情報や推測は避け、「その情報は確認できません」と正直に述べてください。
    5. 質問の一部にしか答えられない場合は、答えられる部分のみ回答し、残りは分からないと明示してください。
    6. 専門用語を使用する場合は、簡潔な説明を加えてください。
    7. 回答に自信がない場合は、その旨を明確に伝えてください。

    これまでの会話履歴:
    {chat_history}

    クエリ: {query}

    回答:
    """
    start_time = time.time()
    formatted_chat_history = "\n".join([f"Human: {h[0]}\nAI: {h[1]}" for h in chat_history])
    llm(no_rag_prompt.format(query=query, chat_history=formatted_chat_history), callbacks=[combined_handler])
    no_rag_time = time.time() - start_time
    return combined_handler.text, no_rag_time

def get_rag_answer(qa_chain, retriever, query, chat_history):
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    return qa_chain.llm_chain.llm(qa_chain.llm_chain.prompt.format(context=context, question=query, chat_history=chat_history))

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, prefix=""):
        self.container = container
        self.text = ""
        self.prefix = prefix

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(f"{self.prefix}\n\n{self.text}")

class CombinedStreamHandler(BaseCallbackHandler):
    def __init__(self, streamlit_handler, stdout_handler):
        self.streamlit_handler = streamlit_handler
        self.stdout_handler = stdout_handler
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.streamlit_handler.on_llm_new_token(token, **kwargs)
        self.stdout_handler.on_llm_new_token(token, **kwargs)
        self.text += token

# データベース接続関数
def get_db_connection():
    return psycopg2.connect(CONNECTION_STRING)

# 会話履歴をデータベースに保存する関数を修正
def save_conversation_history(rag_conversation_history, no_rag_conversation_history, processing_times, user_role):
    conn = get_db_connection()
    cur = conn.cursor()
    # 既存のデータを削除（ユーザーロールに基づいて）
    cur.execute("DELETE FROM conversation_history WHERE user_role = %s", (user_role,))
    # 新しいデータを挿入
    cur.execute("""
        INSERT INTO conversation_history (rag_conversation_history, no_rag_conversation_history, processing_times, user_role)
        VALUES (%s, %s, %s, %s)
    """, (json.dumps(rag_conversation_history), json.dumps(no_rag_conversation_history), json.dumps(processing_times), user_role))
    conn.commit()
    cur.close()
    conn.close()

# データベースから会話履歴を読み込む関数を修正
def load_conversation_history(user_role):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT rag_conversation_history, no_rag_conversation_history, processing_times FROM conversation_history WHERE user_role = %s ORDER BY id DESC LIMIT 1", (user_role,))
    result = cur.fetchone()
    cur.close()
    conn.close()
    if result:
        return (
            json.loads(result[0]) if isinstance(result[0], str) else result[0],
            json.loads(result[1]) if isinstance(result[1], str) else result[1],
            json.loads(result[2]) if isinstance(result[2], str) else result[2]
        )
    return [], [], []

# llm_historyを動的に生成する関数を修正
def generate_llm_history(conversation_history, use_rag=True):
    if use_rag:
        return [(query, answer) for query, answer in conversation_history]
    else:
        return [(query, answer) for query, answer in conversation_history]

def get_available_models():
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags")
        if response.status_code == 200:
            models = response.json()["models"]
            return [model["name"] for model in models]
        else:
            print(f"モデルリストの取得に失敗しました。ステータスコード: {response.status_code}")
            return []
    except requests.RequestException as e:
        print(f"モデルリストの取得中にエラーが発生しました: {e}")
        return []

def main():
    st.set_page_config(layout="wide")  # ページ全体を広く使用
    
    st.title("RAGシステム")

    # 利用可能なモデルのリストを取得
    available_models = get_available_models()

    if not available_models:
        st.warning("利用可能なLLMモデルが見つかりません。Ollamaを使用してローカルにモデルをダウンロードしてください。")
        st.stop()  # これ以降の処理を停止

    # セッション状態の初期化を修正
    if 'selected_model' not in st.session_state or st.session_state.selected_model not in available_models:
        st.session_state.selected_model = available_models[0]

    # サイドバーにモデル選択ウィジェットを追加
    selected_model = st.sidebar.selectbox(
        "使用するLLMモデルを選択してください：",
        options=available_models,
        index=available_models.index(st.session_state.selected_model)
    )

    # 選択されたモデルをセッション状態に保存
    st.session_state.selected_model = selected_model

    # LLMの初期化を修正
    llm = Ollama(model=st.session_state.selected_model, temperature=0.7, base_url=OLLAMA_BASE_URL)

    # ユーザーロールの選択を追加（この行を上に移動）
    user_role = st.sidebar.selectbox("ユーザーロール", ["employee", "manager"])

    # セッション状態の初期化を修正
    if 'rag_conversation_history' not in st.session_state or 'no_rag_conversation_history' not in st.session_state or 'processing_times' not in st.session_state or 'user_role' not in st.session_state or st.session_state.user_role != user_role:
        st.session_state.rag_conversation_history, st.session_state.no_rag_conversation_history, st.session_state.processing_times = load_conversation_history(user_role)
        st.session_state.user_role = user_role
    if 'query' not in st.session_state:
        st.session_state.query = ""

    # クリアボタンを追加
    if st.sidebar.button("会話履歴をクリア"):
        st.session_state.rag_conversation_history = []
        st.session_state.no_rag_conversation_history = []
        st.session_state.processing_times = []
        save_conversation_history([], [], [], user_role)
        st.rerun()

    # 会話履歴の表示を修正
    for i in range(max(len(st.session_state.rag_conversation_history), len(st.session_state.no_rag_conversation_history))):
        st.subheader(f"質問 {i+1}")
        
        rag_query, rag_answer = st.session_state.rag_conversation_history[i] if i < len(st.session_state.rag_conversation_history) else ("", "")
        no_rag_query, no_rag_answer = st.session_state.no_rag_conversation_history[i] if i < len(st.session_state.no_rag_conversation_history) else ("", "")
        
        st.write(rag_query or no_rag_query)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**RAGを利用しない回答**")
            st.write(no_rag_answer)
            if i < len(st.session_state.processing_times):
                no_rag_time, _, _, _ = st.session_state.processing_times[i]
                st.markdown("**処理時間:**")
                st.write(f"回答生成: {no_rag_time:.4f}秒")
                st.write("ベクトル変換: 0.0000秒")
                st.write("関連情報検索: 0.0000秒")
        with col2:
            st.markdown("**RAGを利用した回答**")
            st.write(rag_answer)
            if i < len(st.session_state.processing_times):
                _, retrieval_time, llm_time, _ = st.session_state.processing_times[i]
                st.markdown("**処理時間:**")
                st.write(f"ベクトル変換: 0.0000秒")
                st.write(f"関連情報検索: {retrieval_time:.4f}秒")
                st.write(f"回答生成: {llm_time:.4f}秒")
        
        st.markdown("---")

    # ストリーミング出力用のコンテナ
    stream_container = st.container()

    # 送信ボタンを入力欄の右に配置
    input_container = st.container()
    with input_container:
        col1, col2, col3 = st.columns([20, 4, 1])
        with col1:
            query = st.text_input("新しい質問を入力してください：", key="query_input", value=st.session_state.query, label_visibility="collapsed")
        with col2:
            submit_button = st.button("送信", key="submit", use_container_width=True)
        with col3:
            st.write("")  # 空のカラムでバランスを取る

    if submit_button and query:
        try:
            with st.spinner("回答を生成中..."):
                with stream_container:
                    # RAGを利用しない回答のストリーミング
                    st.markdown("**RAGを利用しない回答：**")
                    no_rag_container = st.empty()
                    no_rag_streamlit_handler = StreamHandler(no_rag_container)
                    no_rag_stdout_handler = StreamingStdOutCallbackHandler()
                    no_rag_combined_handler = CombinedStreamHandler(no_rag_streamlit_handler, no_rag_stdout_handler)
                    
                    no_rag_llm_history = generate_llm_history(st.session_state.no_rag_conversation_history, use_rag=False)
                    no_rag_answer, no_rag_time = get_no_rag_answer(llm, query, no_rag_combined_handler, no_rag_llm_history)
                    
                    st.markdown("**処理時間:**")
                    st.write(f"回答生成: {no_rag_time:.4f}秒")
                    st.write("ベクトル変換: 0.0000秒")
                    st.write("関連情報検索: 0.0000秒")
                    
                    st.markdown("---")
                    
                    # split_queryの結果を表示（ここで一度だけ呼び出す）
                    subqueries = split_query(query)
                    st.subheader("生成されたサブクエリ:")
                    for i, subquery in enumerate(subqueries):
                        st.write(f"{i+1}. {subquery}")
                    st.markdown("---")
                    
                    # RAGを利用する回答のストリーミング
                    st.markdown("**RAGを利用した回答：**")
                    rag_container = st.empty()
                    rag_streamlit_handler = StreamHandler(rag_container)
                    rag_stdout_handler = StreamingStdOutCallbackHandler()
                    rag_combined_handler = CombinedStreamHandler(rag_streamlit_handler, rag_stdout_handler)
                    
                    rag_llm_history = generate_llm_history(st.session_state.rag_conversation_history, use_rag=True)
                    rag_answer, total_retrieval_time, llm_time, total_time, all_retrieved_chunks = process_query_and_generate_final_answer(query, rag_llm_history, llm, rag_combined_handler, user_role, subqueries)
                    
                    st.markdown("**処理時間:**")
                    st.write(f"ベクトル変換: 0.0000秒")
                    st.write(f"関連情報検索: {total_retrieval_time:.4f}秒")
                    st.write(f"回答生成: {llm_time:.4f}秒")
                    
                    st.markdown("---")
                    
                    # チャンクの表示
                    st.subheader("取得されたチャンク:")
                    for subquery, chunks in all_retrieved_chunks:
                        st.markdown(f"**サブクエリ: {subquery}**")
                        for i, (chunk, score) in enumerate(chunks):
                            st.markdown(f"チャンク {i+1}:")
                            st.markdown(f"<div style='background-color: #f0f2f6; padding: 10px; border-radius: 5px;'>{chunk}</div>", unsafe_allow_html=True)
                            st.markdown(f"<div style='background-color: #e6f3ff; padding: 5px; border-radius: 5px; display: inline-block; margin-top: 5px;'><strong>Similarity Score:</strong> <span style='color: #0066cc; font-size: 1.2em;'>{score:.4f}</span></div>", unsafe_allow_html=True)
                        st.markdown("---")
                    
                    # 参照元の表示
                    st.subheader("参照元:")
                    for subquery, chunks in all_retrieved_chunks:
                        for chunk, _ in chunks:
                            if isinstance(chunk, str):
                                st.write("テキストチャンク（メタデータなし）")
                            else:
                                st.write(chunk.metadata.get('source', '不明な参照元'))

                # 会話履歴と処理時間をセッション状態に追加
                st.session_state.rag_conversation_history.append((query, rag_answer))
                st.session_state.no_rag_conversation_history.append((query, no_rag_answer))
                st.session_state.processing_times.append((no_rag_time, 0, total_retrieval_time, llm_time))
                
                # データベースに最新の履歴のみを保存（ユーザーロールを含む）
                save_conversation_history(st.session_state.rag_conversation_history, st.session_state.no_rag_conversation_history, st.session_state.processing_times, user_role)
        
        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")
            print(f"詳細なエラー情報: {e}")  # デバッグ用コンソールに詳細を出力
        
        finally:
            # 入力欄をクリア
            st.session_state.query = ""

if __name__ == "__main__":
    main()