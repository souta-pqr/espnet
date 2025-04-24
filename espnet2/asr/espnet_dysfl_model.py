# espnet2/asr/espnet_dysfl_model.py
import torch
import logging
from typing import List, Union, Optional

from espnet2.asr.espnet_model import ESPnetASRModel


class ESPnetASRDysflModel(ESPnetASRModel):
    """ESPnet ASR Model with character-level disfluency classification."""

    def __init__(
        self,
        vocab_size: int,
        token_list: list,
        frontend: torch.nn.Module,
        specaug: torch.nn.Module,
        normalize: torch.nn.Module,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        ctc: torch.nn.Module,
        dysfl_weight: float = 0.1,
        **kwargs,
    ):
        print("ESPnetASRDysflModel is being initialized for character-level disfluency detection")
        super().__init__(
            vocab_size=vocab_size,
            token_list=token_list,
            frontend=frontend,
            specaug=specaug,
            normalize=normalize,
            encoder=encoder,
            decoder=decoder,
            ctc=ctc,
            **kwargs,
        )
        # 文字レベルの分類のための分類器
        encoder_output_size = encoder.output_size()
        self.dysfl_classifier = torch.nn.Linear(encoder_output_size, 1)
        self.dysfl_weight = dysfl_weight
        self.sigmoid = torch.nn.Sigmoid()
        self.bce_loss = torch.nn.BCEWithLogitsLoss(reduction="none")

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        isdysfl: torch.Tensor = None,
        **kwargs,
    ):
        """Forward pass with character-level disfluency classification."""
        # 通常のASR処理を実行
        loss_asr, stats, weight = super().forward(
            speech=speech,
            speech_lengths=speech_lengths,
            text=text,
            text_lengths=text_lengths,
            **kwargs,
        )

        # isdysflが提供されている場合のみ、dysfl分類を行う
        if isdysfl is not None:
            batch_size = speech.size(0)
            max_length = text_lengths.max().item()
            
            # print(f"isdysfl入力データ: {isdysfl.shape}, {isdysfl.dtype}")
            
            try:
                # isdysflをテンソルに変換
                if not isinstance(isdysfl, torch.Tensor):
                    isdysfl = torch.tensor(isdysfl, device=speech.device)
                
                # 形状の処理
                if len(isdysfl.shape) == 1:
                    # 1次元の場合（サンプル数が1）
                    dysfl_label = isdysfl.view(1, -1).float()
                elif len(isdysfl.shape) == 2:
                    # 2次元の場合（バッチ × 文字数）
                    dysfl_label = isdysfl.float()
                else:
                    # その他の形状はエラー
                    raise ValueError(f"Unexpected isdysfl shape: {isdysfl.shape}")
                
                # バッチサイズの調整
                if dysfl_label.size(0) != batch_size:
                    if dysfl_label.size(0) > batch_size:
                        dysfl_label = dysfl_label[:batch_size]
                    else:
                        # バッチサイズが足りない場合、追加のゼロパディング
                        padding = torch.zeros(
                            batch_size - dysfl_label.size(0), dysfl_label.size(1),
                            dtype=torch.float, device=speech.device
                        )
                        dysfl_label = torch.cat([dysfl_label, padding], dim=0)
                
                # シーケンス長の調整
                if dysfl_label.size(1) != max_length:
                    if dysfl_label.size(1) > max_length:
                        # 長すぎる場合は切り詰め
                        dysfl_label = dysfl_label[:, :max_length]
                    else:
                        # 短すぎる場合はパディング
                        padding = torch.zeros(
                            batch_size, max_length - dysfl_label.size(1),
                            dtype=torch.float, device=speech.device
                        )
                        dysfl_label = torch.cat([dysfl_label, padding], dim=1)
                
                # 値を0と1に制限
                dysfl_label = torch.clamp(dysfl_label, 0.0, 1.0)
                
                # print(f"処理後のdysfl_label: {dysfl_label.shape}, 値範囲: {dysfl_label.min()}-{dysfl_label.max()}")
                
                # 有効な文字のマスクを作成（パディング部分を無視）
                mask = torch.arange(max_length, device=text_lengths.device)[None, :] < text_lengths[:, None]
                mask = mask.float()
                
            except Exception as e:
                # print(f"isdysfl処理エラー: {e}")
                # エラーが発生した場合、ダミーのラベルとマスクを作成
                dysfl_label = torch.zeros(batch_size, max_length, dtype=torch.float, device=speech.device)
                mask = torch.arange(max_length, device=text_lengths.device)[None, :] < text_lengths[:, None]
                mask = mask.float()
            
            # エンコーダ出力を取得
            encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
            if isinstance(encoder_out, tuple):
                encoder_out = encoder_out[0]
            
            # エンコーダ出力の長さを文字長に合わせる
            batch_size, enc_length, enc_dim = encoder_out.size()
            
            if enc_length != max_length:
                # print(f"エンコーダ出力長とテキスト長が一致しません: {enc_length} vs {max_length}")
                # エンコーダ出力を文字長に合わせる（線形補間による簡易的な方法）
                resampled_encoder_out = torch.nn.functional.interpolate(
                    encoder_out.transpose(1, 2),  # [batch, dim, frames]
                    size=max_length,
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)  # [batch, max_length, dim]
            else:
                resampled_encoder_out = encoder_out
            
            # 文字ごとに分類
            dysfl_logits = self.dysfl_classifier(resampled_encoder_out).squeeze(-1)  # [batch, max_length]
            
            # マスクを適用して有効な文字のみで損失を計算
            masked_loss = self.bce_loss(dysfl_logits, dysfl_label) * mask
            loss_dysfl = masked_loss.sum() / (mask.sum() + 1e-8)  # パディング部分を除外
            
            # 予測精度を計算
            dysfl_probs = self.sigmoid(dysfl_logits)
            predictions = (dysfl_probs > 0.5).float()
            accuracy = ((predictions == dysfl_label) * mask).sum() / (mask.sum() + 1e-8)
            
            # 合計損失の計算
            loss = loss_asr + self.dysfl_weight * loss_dysfl
            
            # 統計情報の更新
            stats["loss_dysfl"] = loss_dysfl.detach()
            stats["acc_dysfl"] = accuracy
        else:
            loss = loss_asr
        
        return loss, stats, weight
    
    def predict_dysfl(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ):
        """推論時に非流暢性予測を行うメソッド"""
        try:
            # エンコーダ出力を取得
            encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
            if isinstance(encoder_out, tuple):
                encoder_out = encoder_out[0]
            
            # テキストの最大長を取得
            batch_size, max_length = text.size()
            
            # エンコーダ出力の長さを文字長に合わせる
            _, enc_length, enc_dim = encoder_out.size()
            
            logging.info(f"エンコーダ出力サイズ: {encoder_out.size()}, テキストサイズ: {text.size()}")
            
            # エンコーダ出力を文字長に合わせる
            try:
                if enc_length != max_length:
                    logging.info(f"長さが異なるため補間します: {enc_length} → {max_length}")
                    resampled_encoder_out = torch.nn.functional.interpolate(
                        encoder_out.transpose(1, 2),  # [batch, dim, frames]
                        size=max_length,
                        mode='linear',
                        align_corners=False
                    ).transpose(1, 2)  # [batch, max_length, dim]
                else:
                    resampled_encoder_out = encoder_out
                    
            except Exception as e:
                logging.warning(f"補間中にエラーが発生しました: {e}、代替手段を使用します")
                # 代替手段：単純に先頭から取得または0埋め
                if enc_length > max_length:
                    resampled_encoder_out = encoder_out[:, :max_length, :]
                else:
                    padding = torch.zeros(
                        batch_size, max_length - enc_length, enc_dim,
                        device=encoder_out.device, dtype=encoder_out.dtype
                    )
                    resampled_encoder_out = torch.cat([encoder_out, padding], dim=1)
            
            # 文字ごとに分類
            dysfl_logits = self.dysfl_classifier(resampled_encoder_out).squeeze(-1)  # [batch, max_length]
            
            # シグモイド関数で確率に変換
            dysfl_probs = self.sigmoid(dysfl_logits)
            
            # 閾値0.5で二値分類
            dysfl_preds = (dysfl_probs > 0.5).float()
            
            # 無効なパディング部分をマスク（ignore_idの位置）
            mask = text != self.ignore_id
            
            # マスクを適用
            dysfl_probs = dysfl_probs * mask.float()
            dysfl_preds = dysfl_preds * mask.float()
            
            logging.info(f"非流暢性予測完了: probs={dysfl_probs.size()}, preds={dysfl_preds.size()}")
            
            return dysfl_probs, dysfl_preds
            
        except Exception as e:
            logging.error(f"非流暢性予測中にエラーが発生: {e}")
            import traceback
            logging.error(traceback.format_exc())
            
            # エラー時は空の結果を返す
            dummy_output = torch.zeros(batch_size, max_length, device=speech.device)
            return dummy_output, dummy_output
