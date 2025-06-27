import pandas as pd

def generate_submission(df, id_col='id2', offer_col='id3', pred_col='pred', k=7, output_path='submission.csv'):
    """
    Generate a submission file for MAP@7 competitions.
    - df: DataFrame with columns [id_col, offer_col, pred_col]
    - id_col: user/customer ID
    - offer_col: offer ID
    - pred_col: predicted probability or score
    - k: number of top offers to select per user
    - output_path: path to save the submission CSV
    Output CSV format:
        id2, id3
        12345, 111 222 333 444 555 666 777
        ...
    """
    topk = (
        df.sort_values([id_col, pred_col], ascending=[True, False])
          .groupby(id_col)[offer_col]
          .apply(lambda x: ' '.join(map(str, x.head(k))))
          .reset_index()
    )
    topk.columns = [id_col, offer_col]
    topk.to_csv(output_path, index=False)
    print(f"Submission file saved to {output_path}") 