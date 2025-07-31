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
The methods developed for this food hazard detection include both conventional ML models and transformer-based models, aiming to achieve high accuracy and interpretability in an imbalanced multi-class text classification task. The overall workflow consists of several key stages: data preprocessing, baseline modeling, and advanced transformer-based modelling.

As a benchmark, we implemented a pipeline using TF-IDF vectorization with character n-gram of (2, 5), combined with logistic regression. Four TF-IDF based models were trained for hazard-category, product-category, hazard and product. 

As the transformer-based model, we fine-tuned a RoBERTa-based transformer pre-trained model for each label, leveraging its strong contextual language modeling. Same as the benchmark,  four RoBERTa based models were trained for hazard-category, product-category, hazard and product. 

To handle class imbalance, we experimented with several techniques: 

(1) WeightedRandomSampler in PyTorch’s DataLoader, to sample data points based on  specified probabilities

(2) CrossEntropyLoss with class weights to give more importance to minority classes.

## Results Achieved

(1) Applying imbalance-handling techniques can enhance classification accuracy for minority classes. However, this improvement often comes at the expense of reduced performance on majority classes, likely due to the model encountering a less balanced and potentially distorted representation of the overall data distribution.

(2) The resulting models without imbalnce-handling techniques achieve a macro F1-score of 0.67 for subtask 1 (ST1) to classify hazard categories and product categories, and 0.43 for subtask 2 (ST2) to detect the specific hazard and product. 

## Comparison with Baseline Results (Please include references for the baselines used)

(1) Compared to the Benchmark model, the RoBERTa-based models – with or without imbalance-handling techniques – demonstrates improved performance in identifying the minority classes, as reflected in the confusion matrices. 

(2) The evaluation results for ST1_score and ST2_score are summarized in the Table below. The RoBERTa-based models significantly outperform the benchmark on both ST1 and ST2, demonstrating the strong advantage of transformer-based models for this classification task. 
The model without imbalance-handling techniques achieves the highest overall performance, suggesting that the imbalance-handling techniques may be unnecessary and could even slightly reduce the fine-grained classification accuracy.  ST2 is clearly more challenging, as all models perform worse on it compared to ST1. 

(3) Notably, the RoBERTa model without  imbalance-handling techniques also surpasses the challenge leaderboard results, especially on ST2.

| Model           | ST1 Score | ST2 Score |
|-----------------|-----------|-----------|
| Benchmark (TF-IDF + LR) | 0.49      | 0.14      |
| RoBERTa (w/ imbalance)  | 0.66      | 0.22      |
| RoBERTa (no imbalance)  | 0.67      | 0.43      |
| Challenge leaderboard   | 0.60      | 0.32      |
