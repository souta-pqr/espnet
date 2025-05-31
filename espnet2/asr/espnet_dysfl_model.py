# espnet2/asr/espnet_dysfl_model.py
import torch
import logging
from typing import List, Union, Optional

from espnet2.asr.espnet_model import ESPnetASRModel


class ESPnetASRDysflModel(ESPnetASRModel):
    """ESPnet ASR Model with character-level disfluency classification using token-dependency mechanism.
    
    Disfluency types:
    - 0: Others (fluent)
    - 1: Interjection
    - 2: Repair
    - 3: Filler
    """

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
        num_dysfl_classes: int = 4,  # 四値分類
        **kwargs,
    ):
        print("ESPnetASRDysflModel is being initialized with token-dependency mechanism (4-class)")
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
        # エンコーダの出力サイズ
        encoder_output_size = encoder.output_size()
        
        # トークン埋め込みへのアクセスを取得
        if hasattr(decoder, "embed"):
            self.token_embedding = decoder.embed
            token_embed_size = self.token_embedding.weight.shape[1]
        else:
            print("Warning: Decoder does not have 'embed' attribute. Using a new embedding layer.")
            token_embed_size = 320
            self.token_embedding = torch.nn.Embedding(vocab_size, token_embed_size)
        
        # トークン依存メカニズム用の分類器（4クラス分類）
        self.num_dysfl_classes = num_dysfl_classes
        self.dysfl_classifier = torch.nn.Linear(
            encoder_output_size + token_embed_size, 
            num_dysfl_classes
        )
        
        self.dysfl_weight = dysfl_weight
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-1)
        
        print(f"Token-dependency mechanism initialized: encoder_dim={encoder_output_size}, "
              f"token_embed_dim={token_embed_size}, num_classes={num_dysfl_classes}")

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        isdysfl: torch.Tensor = None,
        **kwargs,
    ):
        """Forward pass with character-level disfluency classification (4-class)."""
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
            
            try:
                # isdysflをテンソルに変換
                if not isinstance(isdysfl, torch.Tensor):
                    isdysfl = torch.tensor(isdysfl, device=speech.device)
                
                # 形状の処理
                if len(isdysfl.shape) == 1:
                    dysfl_label = isdysfl.view(1, -1).long()
                elif len(isdysfl.shape) == 2:
                    dysfl_label = isdysfl.long()
                else:
                    raise ValueError(f"Unexpected isdysfl shape: {isdysfl.shape}")
                
                # バッチサイズの調整
                if dysfl_label.size(0) != batch_size:
                    if dysfl_label.size(0) > batch_size:
                        dysfl_label = dysfl_label[:batch_size]
                    else:
                        padding = torch.full(
                            (batch_size - dysfl_label.size(0), dysfl_label.size(1)),
                            -1,  # ignore_indexを使用
                            dtype=torch.long, device=speech.device
                        )
                        dysfl_label = torch.cat([dysfl_label, padding], dim=0)
                
                # シーケンス長の調整
                if dysfl_label.size(1) != max_length:
                    if dysfl_label.size(1) > max_length:
                        dysfl_label = dysfl_label[:, :max_length]
                    else:
                        padding = torch.full(
                            (batch_size, max_length - dysfl_label.size(1)),
                            -1,  # ignore_indexを使用
                            dtype=torch.long, device=speech.device
                        )
                        dysfl_label = torch.cat([dysfl_label, padding], dim=1)
                
                # 値を0-3に制限（-1はignore_indexとして保持）
                valid_mask = dysfl_label != -1
                dysfl_label = torch.where(
                    valid_mask,
                    torch.clamp(dysfl_label, 0, self.num_dysfl_classes - 1),
                    dysfl_label
                )
                
                # 有効な文字のマスクを作成
                mask = torch.arange(max_length, device=text_lengths.device)[None, :] < text_lengths[:, None]
                mask = mask.float()
                
            except Exception as e:
                # エラーが発生した場合、ダミーのラベルとマスクを作成
                dysfl_label = torch.full(
                    (batch_size, max_length), -1, dtype=torch.long, device=speech.device
                )
                mask = torch.arange(max_length, device=text_lengths.device)[None, :] < text_lengths[:, None]
                mask = mask.float()
            
            # エンコーダ出力を取得
            encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
            if isinstance(encoder_out, tuple):
                encoder_out = encoder_out[0]
            
            # エンコーダ出力の長さを文字長に合わせる
            batch_size, enc_length, enc_dim = encoder_out.size()
            
            if enc_length != max_length:
                resampled_encoder_out = torch.nn.functional.interpolate(
                    encoder_out.transpose(1, 2),
                    size=max_length,
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)
            else:
                resampled_encoder_out = encoder_out
            
            # トークン依存メカニズムの実装
            vocab_size = self.token_embedding.weight.size(0)
            clipped_text = torch.clamp(text, 0, vocab_size - 1)
            token_embeddings = self.token_embedding(clipped_text)
            
            # エンコーダ出力とトークン埋め込みを連結
            combined_features = torch.cat([resampled_encoder_out, token_embeddings], dim=-1)
            
            # 非流暢性ロジットを計算（4クラス分類）
            dysfl_logits = self.dysfl_classifier(combined_features)  # [batch, max_length, 4]
            
            # CrossEntropyLossの計算
            # dysfl_logitsを [batch*max_length, 4] に reshape
            dysfl_logits_flat = dysfl_logits.view(-1, self.num_dysfl_classes)
            dysfl_label_flat = dysfl_label.view(-1)
            
            loss_dysfl = self.ce_loss(dysfl_logits_flat, dysfl_label_flat)
            loss_dysfl = loss_dysfl.view(batch_size, max_length)
            
            # マスクを適用して有効な文字のみで損失を計算
            masked_loss = loss_dysfl * mask
            loss_dysfl = masked_loss.sum() / (mask.sum() + 1e-8)
            
            # 予測精度を計算（クラスごと）
            predictions = torch.argmax(dysfl_logits, dim=-1)  # [batch, max_length]
            
            # 全体の精度
            correct = ((predictions == dysfl_label) & (dysfl_label != -1)).float()
            accuracy = (correct * mask).sum() / (mask.sum() + 1e-8)
            
            # クラスごとの精度を計算
            class_accuracies = {}
            for cls in range(self.num_dysfl_classes):
                cls_mask = (dysfl_label == cls) & (mask.bool())
                if cls_mask.sum() > 0:
                    cls_correct = (predictions == cls) & cls_mask
                    class_accuracies[f"acc_dysfl_cls{cls}"] = cls_correct.float().sum() / cls_mask.float().sum()
            
            # 合計損失の計算
            loss = loss_asr + self.dysfl_weight * loss_dysfl
            
            # 統計情報の更新
            stats["loss_dysfl"] = loss_dysfl.detach()
            stats["acc_dysfl"] = accuracy
            for k, v in class_accuracies.items():
                stats[k] = v
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
        """推論時に非流暢性予測を行うメソッド（四値分類）"""
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
            
            if enc_length != max_length:
                logging.info(f"長さが異なるため補間します: {enc_length} → {max_length}")
                resampled_encoder_out = torch.nn.functional.interpolate(
                    encoder_out.transpose(1, 2),
                    size=max_length,
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)
            else:
                resampled_encoder_out = encoder_out
            
            # トークン依存メカニズム
            vocab_size = self.token_embedding.weight.size(0)
            clipped_text = torch.clamp(text, 0, vocab_size - 1)
            token_embeddings = self.token_embedding(clipped_text)
            
            # エンコーダ出力とトークン埋め込みを連結
            combined_features = torch.cat([resampled_encoder_out, token_embeddings], dim=-1)
            
            # 非流暢性ロジットを計算
            dysfl_logits = self.dysfl_classifier(combined_features)  # [batch, max_length, 4]
            
            # ソフトマックスで確率に変換
            dysfl_probs = torch.softmax(dysfl_logits, dim=-1)  # [batch, max_length, 4]
            
            # 最大確率のクラスを予測
            dysfl_preds = torch.argmax(dysfl_probs, dim=-1)  # [batch, max_length]
            
            # 無効なパディング部分をマスク
            mask = text != self.ignore_id
            dysfl_preds = dysfl_preds * mask.long()
            
            # 各クラスの確率も返す（必要に応じて）
            logging.info(f"非流暢性予測完了: probs={dysfl_probs.size()}, preds={dysfl_preds.size()}")
            
            return dysfl_probs, dysfl_preds
            
        except Exception as e:
            logging.error(f"非流暢性予測中にエラーが発生: {e}")
            import traceback
            logging.error(traceback.format_exc())
            
            # エラー時は空の結果を返す
            dummy_probs = torch.zeros(batch_size, max_length, self.num_dysfl_classes, device=speech.device)
            dummy_preds = torch.zeros(batch_size, max_length, device=speech.device, dtype=torch.long)
            return dummy_probs, dummy_preds
