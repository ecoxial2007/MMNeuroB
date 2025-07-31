import glob
import json
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
from collections import Counter
import os  # For dummy file creation
import numpy as np # For nanmean

# --- 1. Define diagnostic string to hierarchical feature mapping ---

LABEL_TO_FEATURES_MAP = {
    "Ganglioneuroblastoma, nodular type": {"CN": 1, "SS": 1, "N": 1},
    "节细胞性神经母细胞瘤结节型": {"CN": 1, "SS": 1, "N": 1},
    "Ganglioneuroblastoma, mixed type": {"CN": 1, "SS": 1, "N": 0},
    "Ganglioneuroblastoma, intermixed type": {"CN": 1, "SS": 1, "N": 0},
    "节细胞性神经母细胞瘤混合型": {"CN": 1, "SS": 1, "N": 0},
    "Neuroblastoma, poorly differentiated": {"CN": 1, "SS": 0, "N": -1},
    "神经母细胞瘤，分化差": {"CN": 1, "SS": 0, "N": -1},
    "Neuroblastoma, differentiating": {"CN": 1, "SS": 0, "N": -1},
    "神经母细胞瘤，分化": {"CN": 1, "SS": 0, "N": -1},
    "Neuroblastoma": {"CN": 1, "SS": 0, "N": -1},
    "Ganglioneuroma, mature": {"CN": 0, "SS": -1, "N": -1},
    "节细胞性神经瘤，成熟型": {"CN": 0, "SS": -1, "N": -1},
    "Ganglioneuroma, maturing": {"CN": 0, "SS": -1, "N": -1},
    "节细胞性神经瘤，即将成熟型": {"CN": 0, "SS": -1, "N": -1},
    "Ganglioneuroma": {"CN": 0, "SS": -1, "N": -1},
    "节细胞性神经瘤": {"CN": 0, "SS": -1, "N": -1},
}
DEFAULT_FEATURES = {"CN": -2, "SS": -2, "N": -2}

def get_hierarchical_features(label_str):
    return LABEL_TO_FEATURES_MAP.get(label_str, DEFAULT_FEATURES)


# --- 2. Load and process JSON data (similar to before) ---
def load_and_process_data(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)['per_file_predictions']

    all_y_gt_str, all_y_pred_str = [], []
    cn_gt_list, cn_pred_list = [], []
    ss_gt_list, ss_pred_list = [], []
    n_gt_list, n_pred_list = [], []
    unmapped_gt_labels, unmapped_pred_labels = Counter(), Counter()

    for case_id, values in data.items():
        y_gt_str, y_pred_str = values["y_gt"], values["y_pred"]
        all_y_gt_str.append(y_gt_str)
        all_y_pred_str.append(y_pred_str)

        gt_features = get_hierarchical_features(y_gt_str)
        pred_features = get_hierarchical_features(y_pred_str)

        if gt_features == DEFAULT_FEATURES: unmapped_gt_labels[y_gt_str] += 1
        if pred_features == DEFAULT_FEATURES: unmapped_pred_labels[y_pred_str] += 1

        if gt_features["CN"] != -2:
            cn_gt_list.append(gt_features["CN"])
            cn_pred_list.append(pred_features["CN"])

        if gt_features["CN"] == 1 and gt_features["SS"] in [0, 1]:
            ss_gt_list.append(gt_features["SS"])
            ss_pred_list.append(pred_features["SS"])

        if gt_features["CN"] == 1 and gt_features["SS"] == 1 and gt_features["N"] in [0, 1]:
            n_gt_list.append(gt_features["N"])
            n_pred_list.append(pred_features["N"])

    if unmapped_gt_labels:
        print("警告: 以下 GT 标签未能映射到层级特征 (出现次数):")
        for label, count in unmapped_gt_labels.items(): print(f"  - '{label}': {count}")
    if unmapped_pred_labels:
        print("警告: 以下 Pred 标签未能映射到层级特征 (出现次数):")
        for label, count in unmapped_pred_labels.items(): print(f"  - '{label}': {count}")
    print("-" * 30)

    return {
        "all_y_gt_str": all_y_gt_str, "all_y_pred_str": all_y_pred_str,
        "cn_gt": cn_gt_list, "cn_pred": cn_pred_list,
        "ss_gt": ss_gt_list, "ss_pred": ss_pred_list,
        "n_gt": n_gt_list, "n_pred": n_pred_list,
    }

# --- 3. Calculate metrics and return DataFrame ---
def get_metrics_df(y_true, y_pred, title, labels_for_report, target_names_for_report):
    """Calculates metrics and returns a pandas DataFrame"""
    if not y_true:
        print(f"\n{title} Metrics: No enough data to evaluate.")
        return pd.DataFrame()  # Return empty DataFrame

    try:
        report_dict = classification_report(
            y_true, y_pred,
            labels=labels_for_report,
            target_names=target_names_for_report,
            output_dict=True,
            zero_division=0
        )
        # Convert dictionary to DataFrame, using categories/averages as row index
        df_report = pd.DataFrame(report_dict).transpose()
        # Convert support column to integer if exists
        if 'support' in df_report.columns:
            df_report['support'] = df_report['support'].astype(int)
        return df_report
    except Exception as e:
        print(f"  Error calculating {title} metrics: {e}")
        return pd.DataFrame()

def process_single_json(json_file_path):
    """
    Processes a single JSON file, calculates all metrics, and returns a dictionary of DataFrames.
    """
    print(f"Processing {os.path.basename(json_file_path)}...")
    processed_data = load_and_process_data(json_file_path)
    all_metrics_dfs = {}

    # --- 4a. Calculate and store metrics for each final class ---
    unique_final_classes = sorted(list(set(processed_data["all_y_gt_str"]) | set(processed_data["all_y_pred_str"])))

    df_final_metrics = get_metrics_df(
        processed_data["all_y_gt_str"],
        processed_data["all_y_pred_str"],
        title="Final Diagnosis Classes",
        labels_for_report=unique_final_classes,
        target_names_for_report=unique_final_classes
    )
    if not df_final_metrics.empty:
        all_metrics_dfs['Final_Class_Metrics'] = df_final_metrics
    else:
        print("Failed to generate DataFrame for final class metrics.")

    # --- 4b. Calculate and store metrics for each hierarchical level ---
    hierarchical_levels_config = [
        {
            "key_gt": "cn_gt", "key_pred": "cn_pred", "title": "Cell Nests (CN)",
            "labels": [0, 1], "target_names": ["CN Absent (0)", "CN Present (1)"],
            "sheet_name": "Cell_Nests_Metrics"
        },
        {
            "key_gt": "ss_gt", "key_pred": "ss_pred", "title": "Schwannian Stroma (SS)",
            "labels": [0, 1], "target_names": ["SS <50% (0)", "SS >=50% (1)"],
            "sheet_name": "Schwannian_Stroma_Metrics"
        },
        {
            "key_gt": "n_gt", "key_pred": "n_pred", "title": "Nodules (N)",
            "labels": [0, 1], "target_names": ["Nodules Absent (0)", "Nodules Present (1)"],
            "sheet_name": "Nodules_Metrics"
        }
    ]

    for config in hierarchical_levels_config:
        df_level_metrics = get_metrics_df(
            processed_data[config["key_gt"]],
            processed_data[config["key_pred"]],
            title=config["title"],
            labels_for_report=config["labels"],
            target_names_for_report=config["target_names"]
        )
        if not df_level_metrics.empty:
            all_metrics_dfs[config["sheet_name"]] = df_level_metrics
        else:
            print(f"Failed to generate DataFrame for {config['title']} metrics.")

    return all_metrics_dfs

def aggregate_and_export_to_excel(json_file_paths, excel_output_path="classification_metrics_averaged.xlsx"):
    """
    Processes multiple JSON files, aggregates their metrics by averaging,
    and exports the results to an Excel file.
    """
    all_runs_metrics = {} # {sheet_name: [df1, df2, ...]}

    for json_file_path in json_file_paths:
        metrics_for_this_run = process_single_json(json_file_path)
        for sheet_name, df in metrics_for_this_run.items():
            if sheet_name not in all_runs_metrics:
                all_runs_metrics[sheet_name] = []
            all_runs_metrics[sheet_name].append(df)

    with pd.ExcelWriter(excel_output_path, engine='openpyxl') as writer:
        for sheet_name, list_of_dfs in all_runs_metrics.items():
            if not list_of_dfs:
                print(f"No data to aggregate for sheet: {sheet_name}")
                continue

            # Concatenate all DataFrames for the current sheet
            combined_df = pd.concat(list_of_dfs)

            # Group by index (class names/metric types) and calculate the mean
            # Exclude 'support' from mean calculation, as it should be summed
            cols_to_average = ['precision', 'recall', 'f1-score']
            averaged_df = combined_df[cols_to_average].groupby(combined_df.index).mean()

            # Sum the 'support' column separately
            if 'support' in combined_df.columns:
                summed_support = combined_df['support'].groupby(combined_df.index).sum()
                # Merge summed support back into the averaged DataFrame
                averaged_df['support'] = summed_support.astype(int)

            # Reorder columns if needed to match original order
            original_cols_order = [col for col in list_of_dfs[0].columns if col in averaged_df.columns]
            averaged_df = averaged_df[original_cols_order]

            averaged_df.to_excel(writer, sheet_name=sheet_name)
            print(f"Averaged metrics for '{sheet_name}' written to '{excel_output_path}'.")

    print(f"\nAll averaged metrics successfully saved to: {excel_output_path}")



if __name__ == "__main__":
    # Get all JSON file paths matching the pattern
    json_file_paths = [f'./results/checkpoint_anno4_head8_test_results.json']

    if not json_file_paths:
        print("No JSON files found matching the pattern. Please ensure the directory and file structure are correct.")
    else:
        print(f"Found {len(json_file_paths)} JSON files to process:")
        for p in json_file_paths:
            print(f"- {p}")

        # Run the aggregation and export process
        aggregate_and_export_to_excel(json_file_paths, f"./results/hiera_accuracy.xlsx")