# ローカルRAGプロジェクト

このプロジェクトは、ローカル環境でRetrieval-Augmented Generation (RAG) システムを実現することを目的としています。

## 目的

ローカル環境でRAGを実装し、効率的な情報検索と生成を可能にします。これにより、外部APIに依存せずに、カスタマイズされた質問応答システムを構築できます。

## セットアップ

1. リポジトリをクローンします：
   ```
   git clone [リポジトリのURL]
   ```

1. プロジェクトディレクトリに移動します：
   ```
   cd [プロジェクトディレクトリ名]
   ```

1. 仮想環境を作成し、アクティブ化します：
   ```
   python -m venv rag_env
   source rag_env/bin/activate  # Linuxの場合
   # または
   rag_env\Scripts\activate  # Windowsの場合
   ```

1. 依存関係をインストールします：
   ```
   pip install -r requirements.txt
   ```

## プロジェクト構造

- `basic_rag.py`: RAGシステムの主要なロジックを含むスクリプト
- `data_preparation.py`: データの準備とベクトルストアの作成を行うスクリプト
- `check_faiss.py`: FAISSベクトルストアの確認用スクリプト
- `requirements.txt`: プロジェクトの依存関係リスト
- `.gitignore`: バージョン管理から除外するファイルやディレクトリを指定
- `data/`: サンプルデータを含むディレクトリ
- `vectorstore/`: FAISSベクトルストアのデータを保存するディレクトリ（gitignoreされています）

## 使用方法

1. データの準備：
   ```
   python data_preparation.py
   ```

1. ベクトルストアの確認：
   ```
   python check_faiss.py
   ```

1. RAGシステムの実行：
   ```
   python basic_rag.py
   ```

## 注意事項

- このプロジェクトはローカル環境で動作するように設計されています。
- 大規模なデータセットを扱う場合は、メモリ使用量に注意してください。
- モデルやベクトルデータベースの選択は、パフォーマンスと精度のバランスを考慮して行ってください。


## ライセンス
