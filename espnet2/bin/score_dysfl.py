#!/usr/bin/env python3

import argparse
import logging
import numpy as np
import sys
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def get_parser():
    parser = argparse.ArgumentParser(
        description="非流暢性検出結果の評価（4クラス分類）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pred", type=str, required=True, help="予測ファイル")
    parser.add_argument("--ref", type=str, required=True, help="参照ファイル")
    parser.add_argument("--output", type=str, required=True, help="結果出力ファイル")
    return parser

def main(args):
    # クラス名の定義
    class_names = ["Others", "Interjection", "Repair", "Filler"]
    
    # 予測ファイルと参照ファイルを読み込む
    pred_dict = {}
    with open(args.pred, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(None, 1)
            if len(parts) == 2:
                utt_id, pred_str = parts
                pred_dict[utt_id] = np.array([int(float(p)) for p in pred_str.split()])
    
    ref_dict = {}
    with open(args.ref, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(None, 1)
            if len(parts) == 2:
                utt_id, ref_str = parts
                ref_dict[utt_id] = np.array([int(float(r)) for r in ref_str.split()])
    
    # 共通のutt_idだけを評価
    common_ids = sorted(set(pred_dict.keys()) & set(ref_dict.keys()))
    
    if not common_ids:
        logging.error("予測と参照の間に共通の発話IDが見つかりません")
        return 1
    
    # 評価用の配列を作成
    all_preds = []
    all_refs = []
    
    for utt_id in common_ids:
        pred = pred_dict[utt_id]
        ref = ref_dict[utt_id]
        
        # 長さが異なる場合は短い方に合わせる
        min_len = min(len(pred), len(ref))
        pred = pred[:min_len]
        ref = ref[:min_len]
        
        all_preds.extend(pred)
        all_refs.extend(ref)
    
    # NumPy配列に変換
    all_preds = np.array(all_preds)
    all_refs = np.array(all_refs)
    
    # 評価指標を計算
    accuracy = accuracy_score(all_refs, all_preds)
    
    # クラスごとの詳細な評価指標
    report = classification_report(
        all_refs, all_preds, 
        labels=[0, 1, 2, 3],
        target_names=class_names,
        output_dict=True
    )
    
    # 混同行列
    cm = confusion_matrix(all_refs, all_preds, labels=[0, 1, 2, 3])
    
    # 結果をファイルに書き込む
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("非流暢性検出結果（4クラス分類）\n")
        f.write("==========================\n")
        f.write(f"全体正解率: {accuracy:.4f}\n\n")
        
        f.write("クラスごとの評価指標:\n")
        f.write("-" * 50 + "\n")
        for i, class_name in enumerate(class_names):
            if str(i) in report:
                metrics = report[str(i)]
                f.write(f"\n{class_name} (Class {i}):\n")
                f.write(f"  適合率: {metrics['precision']:.4f}\n")
                f.write(f"  再現率: {metrics['recall']:.4f}\n")
                f.write(f"  F1スコア: {metrics['f1-score']:.4f}\n")
                f.write(f"  サポート: {int(metrics['support'])}\n")
        
        f.write("\n混同行列:\n")
        f.write("-" * 50 + "\n")
        f.write("    予測→  ")
        for name in class_names:
            f.write(f"{name[:6]:>8}")
        f.write("\n")
        f.write("真値↓\n")
        for i, row in enumerate(cm):
            f.write(f"{class_names[i]:<10}")
            for val in row:
                f.write(f"{val:>8}")
            f.write("\n")
        
        # マクロ平均とマイクロ平均
        f.write("\n全体的な評価指標:\n")
        f.write("-" * 50 + "\n")
        if 'macro avg' in report:
            macro = report['macro avg']
            f.write(f"マクロ平均 - 適合率: {macro['precision']:.4f}, "
                   f"再現率: {macro['recall']:.4f}, "
                   f"F1スコア: {macro['f1-score']:.4f}\n")
        if 'weighted avg' in report:
            weighted = report['weighted avg']
            f.write(f"重み付き平均 - 適合率: {weighted['precision']:.4f}, "
                   f"再現率: {weighted['recall']:.4f}, "
                   f"F1スコア: {weighted['f1-score']:.4f}\n")
    
    # 標準出力にも結果を表示
    logging.info(f"全体正解率: {accuracy:.4f}")
    for i, class_name in enumerate(class_names):
        if str(i) in report:
            metrics = report[str(i)]
            logging.info(f"{class_name}: F1={metrics['f1-score']:.4f}, "
                        f"P={metrics['precision']:.4f}, "
                        f"R={metrics['recall']:.4f}")
    
    return 0

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    )
    sys.exit(main(args))
