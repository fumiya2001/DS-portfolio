# DS Portfolio

## Projects

###  LLM Classification (Kaggle)
- 内容：テキスト分類タスクに対してLLMを活用
- 手法：microsoft/deberta-v3-base を用いたfine-tuningおよびアンサンブル
- 結果：Accuracy 84.4%

詳細: ./LLM_Classification_kaggle

---

###  Titanic
- 内容：生存予測モデル
- 手法：複数のアルゴリズム比較、ハイパーパラメータチューニング
- 結果：Accuracy 78.4%

詳細: ./titanic

### LASSO
- 内容:LASSO回帰(L1正則化)のアルゴリズムをRでスクラッチ実装
  
詳細: ./LASSO_from_scratch

### SQL
- 内容: SQLを用いた売上分析・カテゴリ分析・遅延分析  
- 手法：JOIN、GROUP BY、CTE、ウィンドウ関数を使用したデータ分析  


詳細: ./SQL

### Simple RAG System
- 内容：PDFをもとに質問応答を行うRAG（Retrieval-Augmented Generation）システムの実装  
- 手法：Sentence Transformersによるembedding、pgvectorによるベクトル検索、Cross-Encoderによるreranking、FastAPIによるAPI化、ローカルLLMによる回答生成  
  

詳細: ./simple_RAG

## Skills
- Python：pandas, scikit-learn, PyTorch, transformers
- R：統計・アルゴリズム実装
- SQL：データ抽出・前処理
- Docker：環境構築・再現性確保
- NLP / LLM：テキスト分類, fine-tuning (LoRA）、RAG
