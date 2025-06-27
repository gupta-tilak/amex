# American Express Offer Click Prediction

This project builds a modular machine learning pipeline to predict the probability of a customer clicking on an offer, based on provided datasets.

## Project Structure

```
amex/
  data/
    data_loader.py           # Data loading functions
    data_cleaning.py         # Data cleaning/preprocessing
    feature_engineering.py   # Feature engineering/selection
  eda/
    exploratory_analysis.py  # EDA and visualization
  models/
    model_training.py        # Model training and evaluation
  utils/
    helpers.py               # (Optional) Utility functions
  main.py                    # Pipeline runner
  requirements.txt           # Dependencies
  README.md                  # Project overview
  dataset/                   # Place all data files here
```

## Setup

1. Clone the repository and navigate to the project directory.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place all provided data files in the `dataset/` directory.

## Usage

To run the full pipeline (data loading, cleaning, feature engineering, model training):

```bash
python -m amex.main
```

- The pipeline will print progress at each step and show evaluation metrics for baseline models (Logistic Regression, LightGBM).
- EDA plots are available in `eda/exploratory_analysis.py` but are commented out by default in `main.py`. Uncomment to visualize data distributions and correlations.

## Assumptions
- Data files are in Parquet/CSV format as described in the problem statement.
- Feature engineering and cleaning steps are modular and can be easily modified.
- All code is commented with assumptions for easy review and adjustment.

## Next Steps
- Tune models, add advanced feature engineering, and improve evaluation as needed.
