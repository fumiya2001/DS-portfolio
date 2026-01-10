# NLP Disaster Tweets Classification from Kaggle

This project explores various Natural Language Processing (NLP) techniques to classify whether a tweet is about a real disaster or not, using the [Kaggle Disaster Tweets dataset](https://www.kaggle.com/c/nlp-getting-started).

##  Project Overview
The goal is to demonstrate a full machine learning pipeline, including feature extraction from text, dimensionality reduction, and advanced ensemble modeling with Transformer architectures.

##  Execution Environment

> **Note**  
> Only the final notebook (`04_Model_Ensemble.ipynb`) requires **Google Colab with GPU support**  
> due to the computational cost of training multiple Transformer models  
> using 5-fold cross-validation ensemble averaging.
>
> The other notebooks can be executed on a local machine.

For reproducibility, the final model training notebook can be opened directly in Google Colab.

### Run on Google Colab
 
- **Ensemble Model (GPU required)**  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
  https://colab.research.google.com/github/fumiya2001/DS-portfolio/blob/main/LLM_Classification_kaggle/04_Model_Ensemble.ipynb
  )


## Repository Structure

| File | Description | Key Tech Stack |
|:---|:---|:---|
| `01_Embeddings_Generation.ipynb` | Feature extraction from raw text using pre-trained models. | Transformers, PyTorch |
| `02_PCA_Logistic.ipynb` | Dimensionality reduction and baseline classification. | Scikit-learn, PCA, Logistic |
| `03_Deep-learning_transformers.ipynb` | Fine-tuning BERT-based models for sequence classification. | Transformers |
| `04_Model_Ensemble.ipynb` | Combining multiple models to optimize predictive performance. | Ensemble Learning, Transformers |

---

## Technical Deep Dive

### 1. Feature Engineering & PCA
Since raw embeddings often result in high-dimensional feature spaces (768 dimensions), I applied **Principal Component Analysis (PCA)** to:
- Reduce computational overhead for downstream models.
- Mitigate the risk of the "curse of dimensionality."
- Visualize the information density using cumulative explained variance.

### 2. Model Evolution
- **Baseline:** SVM on PCA-reduced embeddings offered a fast and interpretable starting point.
- **Deep Learning:** Fine-tuned Transformer models (deberta-v3-base) to capture nuanced linguistic contexts that linear models miss.
- **Ensemble:** **Ensemble (Cross-Validation Averaging):** Instead of relying on a single model, I implemented a **5-fold Cross-Validation** strategy. I trained 5 separate Transformer models on different folds of the data and averaged their predictions for the test set. This "Ensemble Averaging" significantly improved model robustness and reduced variance, leading to a more stable and higher leaderboard score.

##  Results & Insights
| Model | Accuracy / F1-Score | Note |
|:---|:---|:---|
| PCA + SVM |  [0.76708] | Lightweight & Fast |
| Transformer |  [0.83297] | Strong contextual understanding |
| **Ensemble (Best)** |  **[0.84400]** | Optimized for competition |


---

##  Requirements
- numpy 
- pandas
- Scikit-learn 
- Transformers / Datasets (HuggingFace)
- Matplotlib 