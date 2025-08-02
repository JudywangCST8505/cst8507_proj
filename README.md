## Problem Definition
This project aims to develop an interpretable classification system capable of identifying food hazards and associated products from recall reports. The primary research question is: Can we build a food safety classification model that is both accurate and explainable? To explore this, we compare traditional text classification methods with transformer-based models such as RoBERTa, using datasets from the “SemEval 2025 Task 9: The Food Hazard Detection Challenge.

## Dataset Used

This project uses the official dataset released for SemEval-2025 Task 9: The Food Hazard Detection Challenge https://github.com/food-hazard-detection-semeval-2025/food-hazard-detection-semeval-2025.github.io/tree/main/data. 

The dataset comprises 5,082 labeled training samples, 565 labeled validation samples, and 997 labeled test samples. Each sample contains metadata fields (year, month, day, country), title, text, and four labels (hazard-category, product-category, hazard, and product). While the title is very short, the text gives more detailed description of the recall.

## Evaluation Metrics

Two evaluation scores, ST1_score and ST2_score, are used to measure the classification performance:

- ST1_score: Measures model performance on higher-level hazard-category and product-category labels
- ST2_score: Measures performance on more fine-grained hazard and product labels

The ST1_score is calculated as the average of two macro F1 scores: One for classifying the hazard category, and the other for classifying the product category. The ST2_score is calculated as the average of two macro F1 scores: One for classifying the hazard, and the other for classifying the product. 

## Model Explanation
Two models were developed for this food hazard detection: TF-IDF vectorization with logistic regression (TF-IDF-LR) model and transformer-based RoBERTa model. The overall workflow consists of several key stages: data preprocessing, benchmark modelling, advanced transformer-based modelling, training, and performance evaluation.

As the benchmark, we implemented a pipeline using TF-IDF vectorization with character n-gram of (2, 5), combined with logistic regression. Four TF-IDF based models were trained for hazard-category, product-category, hazard and product. 

As the transformer-based model, we fine-tuned a RoBERTa-based transformer pre-trained model for each label. Four RoBERTa based models were trained for hazard-category, product-category, hazard and product. 

To handle class imbalance, two techniques were experimented:

(1) WeightedRandomSampler in PyTorch’s DataLoader, to sample data points based on specified probabilities

(2) CrossEntropyLoss with class weights to give more importance to minority classes

## Results Achieved

(1) Applying imbalance-handling techniques can enhance classification accuracy for minority classes. However, this improvement often comes at the expense of reduced performance on majority classes.

(2) The RoBERTa models without imbalnce-handling techniques achieve a macro F1-score of 0.67 for subtask 1 (ST1) to classify hazard category and product category, and 0.43 for subtask 2 (ST2) to detect the specific hazard and product. 

## Comparison with Baseline Results

(1) The performances of various models are summarized in the Table below. The RoBERTa-based models significantly outperform the benchmark on both ST1 and ST2, demonstrating the strong advantage of transformer-based models for this classification task. 

(2) The RoBERTa-based model without imbalance-handling techniques achieves the highest overall performance, suggesting that the imbalance-handling techniques is unnecessary and could even slightly reduce the fine-grained classification accuracy. 

(3) Notably, the RoBERTa model without imbalance-handling techniques also surpasses the challenge leaderboard results, especially on ST2 [https://docs.google.com/spreadsheets/d/e/2PACX-1vSHGZQIN_As8etpiZdpIFOl0tCFArK3FA1N7d4yiScAt7hfoj8LEZzYh2jk3XmZjp_hoCajOJddaj0b/pubhtml?widget=true&headers=false#gid=666707274].

| Model           | ST1 Score | ST2 Score |
|-----------------|-----------|-----------|
| Benchmark (TF-IDF + LR) | 0.49      | 0.14      |
| RoBERTa (w/ imbalance)  | 0.66      | 0.22      |
| RoBERTa (no imbalance)  | 0.67      | 0.43      |
| Challenge leaderboard   | 0.60      | 0.32      |
