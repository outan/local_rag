# ローカルRAGプロジェクト

## 概要
このプロジェクトは、ローカル環境でRetrieval-Augmented Generation (RAG) システムを実現することを目的としています。外部APIに依存せずに、効率的な情報検索と生成を可能にし、カスタマイズされた質問応答システムを構築します。

## セットアップ

1. リポジトリのクローン：
   ```
   git clone https://github.com/outan/local_rag
   cd local_rag
   ```

1. 仮想環境の作成とアクティベーション：
   ```
   python -m venv rag_env
   source rag_env/bin/activate  # Linux/macOS
   # または
   rag_env\Scripts\activate  # Windows
   ```

1. 依存関係のインストール：
   ```
   pip install -r requirements.txt
   ```

## プロジェクト構造

- `basic_rag.py`: RAGシステムの主要ロジック
- `vectorstore_preparation.py`: チャンクの分割とベクトルストア作成
- `check_faiss.py`: FAISSベクトルストア確認用
- `requirements.txt`: プロジェクト依存関係リスト
- `.gitignore`: バージョン管理除外ファイル指定
- `data/`: サンプルデータ格納ディレクトリ
- `vectorstore/`: FAISSベクトルストアデータ保存ディレクトリ（gitignore対象）

## 使用方法

1. データ準備：
   - `data/`フォルダにテキストファイル（.txtまたは.md）を配置
   - これらのファイルがRAGシステムの知識ベースとなります

1. ベクトルストア作成：
   ```
   python vectorstore_preparation.py
   ```
   - 対象：`data/`フォルダ内のテキストファイル（.txtまたは.md）
   - 結果：`vectorstore/`フォルダにFAISSベクトルストアが作成されます
        - index.faiss: FAISSインデックスファイル（高速な類似度検索のためのベクトルデータを格納）
        - index.pkl: メタデータと設定情報を含むPickleファイル
    - 各ファイルの役割：
        - index.faiss: テキストの埋め込みベクトルを効率的に検索できる形式で保存
        - index.pkl: 以下の情報を保存
            - ドキュメントメタデータ
                - 各ドキュメントの元のファイル名
                - ドキュメントのID
                - テキストチャンクの位置情報（開始位置、終了位置）
            - チャンクの原文（page_contentとして保存）
            - インデックス構造情報
                - FAISSインデックスの構造に関する情報
                - ベクトルIDとドキュメントIDのマッピング
            - 埋め込み設定（モデル名、次元数）
            - テキスト分割設定（チャンクサイズ、オーバーラップ）
            - 検索設定：デフォルトの類似度検索パラメータ（例：k値）
            - その他の設定（バージョン、作成日時など）

1. ベクトルストア確認：
   ```
   python check_faiss.py
   ```

1. RAGシステム実行：
   ```
   python basic_rag.py
   ```

## 注意事項

- このプロジェクトはローカル環境で動作するように設計されています。
- 大規模なデータセットを扱う場合は、メモリ使用量に注意してください。
- モデルやベクトルデータベースの選択は、パフォーマンスと精度のバランスを考慮して行ってください。


## ライセンス
