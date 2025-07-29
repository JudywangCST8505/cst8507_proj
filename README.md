## Problem Definition

## Dataset Used

## Evaluation Metrics

One metric, defined below, is used to evaluate the detection accuracy for the subtask 2 (ST2). For the ST1, the hazards_pred, products_pred, hazards_pred, and products_pred are replaced by hazards_category_pred, products_category_pred, hazards_category_pred, and products_category_pred

def compute_score(hazards_true, products_true, hazards_pred, products_pred): \
  f1_hazards = f1_score( \
    hazards_true, \
    hazards_pred, \
    average='macro' \
  ) \
  f1_products = f1_score( \
    products_true[hazards_pred == hazards_true], \
    products_pred[hazards_pred == hazards_true], \
    average='macro' \
  ) \
  return (f1_hazards + f1_products) / 2

## Model Explanation

## Results Achieved

## Comparison with Baseline Results (Please include references for the baselines used)
