# 🛡️ Toxic Comment Classification

A machine learning project to detect and classify toxic comments with high accuracy and fairness.  
Built using advanced NLP methods, bias mitigation strategies, and adversarial training to support scalable and ethical content moderation.

---

## 📌 Overview
Online platforms face a growing challenge in moderating toxic content such as hate speech, harassment, and identity-based attacks.  
This project uses the **Jigsaw Unintended Bias in Toxicity Classification** dataset to develop robust models capable of detecting toxicity even under challenging conditions like class imbalance and adversarial input.

**Key highlights:**
- Explored **transformer-based embeddings** (BERT, RoBERTa, DeBERTa) vs. traditional NLP methods.
- Applied techniques like **oversampling, weighted loss, and focal loss** to handle class imbalance.
- Enhanced **robustness** using adversarial training against lexical manipulation.
- Evaluated fairness and accuracy across multiple demographic and identity-based attributes.

---

## ✨ Features
✅ Toxicity classification with bias mitigation  
✅ Advanced embeddings (BERT, RoBERTa, DeBERTa)  
✅ Adversarial training to handle obfuscated toxic language  
✅ Exploratory Data Analysis on gender, race, and identity-related bias  
✅ Metrics: **F1-score, Precision, Recall, AUC** for fair evaluation

---

## 🛠️ Tech Stack
| Category | Tools / Frameworks |
|----------|-------------------|
| Programming | Python |
| NLP / ML | BERT, RoBERTa, DeBERTa, scikit-learn, PyTorch |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Cloud | AWS SageMaker (training, preprocessing, pipeline design) |

---


---

## 🔬 Methodology
1. **Data Preparation**  
   - Jigsaw dataset uploaded to S3 and explored in AWS SageMaker.
   - Applied two preprocessing strategies (minimal vs. traditional cleaning).
   - Addressed missing values and retained key linguistic features.

2. **Model Training**  
   - Transformer-based embeddings (DeBERTa-v3) with adversarial training.
   - Class imbalance handled via **class weighting**, **oversampling**, and **data augmentation**.

3. **Evaluation**  
   - F1-score, Precision, Recall, and AUC to ensure balanced performance.
   - Compared results across preprocessing strategies and model architectures.

---

## 📊 Key Insights from EDA
- **Insult** is the most common toxicity type (≈59% of toxic comments).
- **Obscene language** and **identity attacks** highly correlate with toxicity.
- Transgender-related comments show a disproportionately higher toxicity ratio.
- Homosexual mentions have the highest rate of identity attacks.

---

## 📈 Results
- Successfully built a **robust toxicity detection model** with improved bias handling.
- Model shows strong potential for real-world moderation systems.
- Contributions to research on **ethical AI** and **adversarial robustness**.

---

## 🚀 Future Work
- Extend multilingual support for low-resource languages.
- Improve model explainability for better deployment.
- Experiment with multimodal features (e.g., images or metadata).


⭐ If you find this project helpful or inspiring, please give it a star!  
Feel free to fork the repo and submit pull requests with improvements.


