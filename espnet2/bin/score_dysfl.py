#!/usr/bin/env python3

import argparse
import logging
import numpy as np
import sys
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix

def get_parser():
    parser = argparse.ArgumentParser(
        description="非流暢性検出結果の評価",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pred", type=str, required=True, help="予測ファイル")
    parser.add_argument("--ref", type=str, required=True, help="参照ファイル")
    parser.add_argument("--output", type=str, required=True, help="結果出力ファイル")
    return parser

def main(args):
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
    
    # 評価指標を計算
    accuracy = accuracy_score(all_refs, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_refs, all_preds, average='binary', zero_division=0
    )
    
    # 混同行列
    tn, fp, fn, tp = confusion_matrix(all_refs, all_preds, labels=[0, 1]).ravel()
    
    # 結果をファイルに書き込む
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("非流暢性検出結果\n")
        f.write("==========================\n")
        f.write(f"正解率: {accuracy:.4f}\n")
        f.write(f"適合率: {precision:.4f}\n")
        f.write(f"再現率: {recall:.4f}\n")
        f.write(f"F1スコア: {f1:.4f}\n")
        f.write("\n混同行列:\n")
        f.write(f"真陰性: {tn}\n")
        f.write(f"偽陽性: {fp}\n")
        f.write(f"偽陰性: {fn}\n")
        f.write(f"真陽性: {tp}\n")
    
    # 標準出力にも結果を表示
    logging.info(f"正解率: {accuracy:.4f}")
    logging.info(f"適合率: {precision:.4f}")
    logging.info(f"再現率: {recall:.4f}")
    logging.info(f"F1スコア: {f1:.4f}")
    
    return 0

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    )
    sys.exit(main(args))
