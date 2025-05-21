# espnet2/asr/espnet_dysfl_model.py
import torch
import logging
from typing import List, Union, Optional, Dict, Tuple

from espnet2.asr.espnet_model import ESPnetASRModel


class ESPnetASRDysflModel(ESPnetASRModel):
    """ESPnet ASR Model with multi-category disfluency classification using token-dependency mechanism."""

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
        print("ESPnetASRDysflModel is being initialized with multi-category token-dependency mechanism")
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
            # 埋め込み層がない場合のフォールバック
            print("Warning: Decoder does not have 'embed' attribute. Using a new embedding layer.")
            token_embed_size = 320  # 一般的な埋め込みサイズ
            self.token_embedding = torch.nn.Embedding(vocab_size, token_embed_size)
        
        # 3種類の非流暢性のそれぞれにトークン依存メカニズム用の分類器を作成
        self.dysfl_classifier_filler = torch.nn.Linear(encoder_output_size + token_embed_size, 1)
        self.dysfl_classifier_disfluency = torch.nn.Linear(encoder_output_size + token_embed_size, 1)
        self.dysfl_classifier_interjection = torch.nn.Linear(encoder_output_size + token_embed_size, 1)
        
        self.dysfl_weight = dysfl_weight
        self.sigmoid = torch.nn.Sigmoid()
        self.bce_loss = torch.nn.BCEWithLogitsLoss(reduction="none")
        
        print(f"Multi-category token-dependency mechanism initialized: encoder_dim={encoder_output_size}, token_embed_dim={token_embed_size}")

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        isdysfl_filler: torch.Tensor = None,
        isdysfl_disfluency: torch.Tensor = None,
        isdysfl_interjection: torch.Tensor = None,
        **kwargs,
    ):
        """Forward pass with multi-category disfluency classification with token-dependency mechanism."""
        # 通常のASR処理を実行
        loss_asr, stats, weight = super().forward(
            speech=speech,
            speech_lengths=speech_lengths,
            text=text,
            text_lengths=text_lengths,
            **kwargs,
        )

        # 各カテゴリの非流暢性分類の損失を計算
        loss_dysfl_total = 0.0
        batch_size = speech.size(0)
        max_length = text_lengths.max().item()
        
        # 有効な文字のマスクを作成（パディング部分を無視）
        mask = torch.arange(max_length, device=text_lengths.device)[None, :] < text_lengths[:, None]
        mask = mask.float()
        
        # エンコーダ出力を取得
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]
        
        # エンコーダ出力の長さを文字長に合わせる
        batch_size, enc_length, enc_dim = encoder_out.size()
        if enc_length != max_length:
            # エンコーダ出力を文字長に合わせる（線形補間による簡易的な方法）
            resampled_encoder_out = torch.nn.functional.interpolate(
                encoder_out.transpose(1, 2),  # [batch, dim, frames]
                size=max_length,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)  # [batch, max_length, dim]
        else:
            resampled_encoder_out = encoder_out
        
        # トークン依存メカニズムの実装
        # 埋め込み層の最大インデックスを取得
        vocab_size = self.token_embedding.weight.size(0)
        # インデックスをクリッピング（0以上vocab_size未満に制限）
        clipped_text = torch.clamp(text, 0, vocab_size - 1)
        # 埋め込みを取得
        token_embeddings = self.token_embedding(clipped_text)  # [batch, max_length, embed_dim]
        
        # エンコーダ出力とトークン埋め込みを連結
        combined_features = torch.cat([resampled_encoder_out, token_embeddings], dim=-1)
        
        dysfl_stats = {}
        
        # フィラーの非流暢性予測と損失計算
        if isdysfl_filler is not None:
            try:
                # isdysflをテンソルに変換して形状を調整
                dysfl_label = self._prepare_dysfl_tensor(isdysfl_filler, batch_size, max_length, speech.device)
                
                # フィラー分類器で非流暢性を予測
                dysfl_logits = self.dysfl_classifier_filler(combined_features).squeeze(-1)  # [batch, max_length]
                
                # マスクを適用して有効な文字のみで損失を計算
                masked_loss = self.bce_loss(dysfl_logits, dysfl_label) * mask
                loss_filler = masked_loss.sum() / (mask.sum() + 1e-8)  # パディング部分を除外
                
                # 予測精度を計算
                dysfl_probs = self.sigmoid(dysfl_logits)
                predictions = (dysfl_probs > 0.5).float()
                accuracy = ((predictions == dysfl_label) * mask).sum() / (mask.sum() + 1e-8)
                
                # 損失を加算
                loss_dysfl_total += loss_filler
                
                # 統計情報の更新
                dysfl_stats["loss_dysfl_filler"] = loss_filler.detach()
                dysfl_stats["acc_dysfl_filler"] = accuracy
            except Exception as e:
                logging.warning(f"フィラー予測中にエラーが発生: {e}")
        
        # 言い直しの非流暢性予測と損失計算
        if isdysfl_disfluency is not None:
            try:
                # isdysflをテンソルに変換して形状を調整
                dysfl_label = self._prepare_dysfl_tensor(isdysfl_disfluency, batch_size, max_length, speech.device)
                
                # 言い直し分類器で非流暢性を予測
                dysfl_logits = self.dysfl_classifier_disfluency(combined_features).squeeze(-1)  # [batch, max_length]
                
                # マスクを適用して有効な文字のみで損失を計算
                masked_loss = self.bce_loss(dysfl_logits, dysfl_label) * mask
                loss_disfluency = masked_loss.sum() / (mask.sum() + 1e-8)  # パディング部分を除外
                
                # 予測精度を計算
                dysfl_probs = self.sigmoid(dysfl_logits)
                predictions = (dysfl_probs > 0.5).float()
                accuracy = ((predictions == dysfl_label) * mask).sum() / (mask.sum() + 1e-8)
                
                # 損失を加算
                loss_dysfl_total += loss_disfluency
                
                # 統計情報の更新
                dysfl_stats["loss_dysfl_disfluency"] = loss_disfluency.detach()
                dysfl_stats["acc_dysfl_disfluency"] = accuracy
            except Exception as e:
                logging.warning(f"言い直し予測中にエラーが発生: {e}")
        
        # 感動詞の非流暢性予測と損失計算
        if isdysfl_interjection is not None:
            try:
                # isdysflをテンソルに変換して形状を調整
                dysfl_label = self._prepare_dysfl_tensor(isdysfl_interjection, batch_size, max_length, speech.device)
                
                # 感動詞分類器で非流暢性を予測
                dysfl_logits = self.dysfl_classifier_interjection(combined_features).squeeze(-1)  # [batch, max_length]
                
                # マスクを適用して有効な文字のみで損失を計算
                masked_loss = self.bce_loss(dysfl_logits, dysfl_label) * mask
                loss_interjection = masked_loss.sum() / (mask.sum() + 1e-8)  # パディング部分を除外
                
                # 予測精度を計算
                dysfl_probs = self.sigmoid(dysfl_logits)
                predictions = (dysfl_probs > 0.5).float()
                accuracy = ((predictions == dysfl_label) * mask).sum() / (mask.sum() + 1e-8)
                
                # 損失を加算
                loss_dysfl_total += loss_interjection
                
                # 統計情報の更新
                dysfl_stats["loss_dysfl_interjection"] = loss_interjection.detach()
                dysfl_stats["acc_dysfl_interjection"] = accuracy
            except Exception as e:
                logging.warning(f"感動詞予測中にエラーが発生: {e}")
        
        # 非流暢性の合計損失があれば、総合損失に加算
        if loss_dysfl_total > 0:
            # 種類数で割って正規化された非流暢性損失に重みを掛ける
            num_dysfl_types = sum([
                1 if isdysfl_filler is not None else 0,
                1 if isdysfl_disfluency is not None else 0,
                1 if isdysfl_interjection is not None else 0
            ])
            
            if num_dysfl_types > 0:
                loss_dysfl = loss_dysfl_total / num_dysfl_types
                loss = loss_asr + self.dysfl_weight * loss_dysfl
                
                # 統計情報の更新
                dysfl_stats["loss_dysfl"] = loss_dysfl.detach()
                # 各種類の統計情報をstatsに追加
                for key, value in dysfl_stats.items():
                    stats[key] = value
            else:
                loss = loss_asr
        else:
            loss = loss_asr
        
        return loss, stats, weight
    
    def _prepare_dysfl_tensor(self, dysfl_data, batch_size, max_length, device):
        """非流暢性データを適切なテンソル形式に変換する"""
        # テンソルに変換
        if not isinstance(dysfl_data, torch.Tensor):
            dysfl_data = torch.tensor(dysfl_data, device=device)
        
        # 形状の処理
        if len(dysfl_data.shape) == 1:
            # 1次元の場合（サンプル数が1）
            dysfl_label = dysfl_data.view(1, -1).float()
        elif len(dysfl_data.shape) == 2:
            # 2次元の場合（バッチ × 文字数）
            dysfl_label = dysfl_data.float()
        else:
            # その他の形状はエラー
            raise ValueError(f"Unexpected dysfl shape: {dysfl_data.shape}")
        
        # バッチサイズの調整
        if dysfl_label.size(0) != batch_size:
            if dysfl_label.size(0) > batch_size:
                dysfl_label = dysfl_label[:batch_size]
            else:
                # バッチサイズが足りない場合、追加のゼロパディング
                padding = torch.zeros(
                    batch_size - dysfl_label.size(0), dysfl_label.size(1),
                    dtype=torch.float, device=device
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
                    dtype=torch.float, device=device
                )
                dysfl_label = torch.cat([dysfl_label, padding], dim=1)
        
        # 値を0と1に制限
        dysfl_label = torch.clamp(dysfl_label, 0.0, 1.0)
        
        return dysfl_label
    
    def predict_dysfl(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """推論時に複数カテゴリの非流暢性予測を行うメソッド（トークン依存メカニズム対応）"""
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
            
            # トークン依存メカニズム - 予測されたトークンの埋め込みを取得
            # 埋め込み層の最大インデックスを取得
            vocab_size = self.token_embedding.weight.size(0)
            # インデックスをクリッピング（0以上vocab_size未満に制限）
            clipped_text = torch.clamp(text, 0, vocab_size - 1)
            # 埋め込みを取得
            token_embeddings = self.token_embedding(clipped_text)  # [batch, max_length, embed_dim]
            
            # エンコーダ出力とトークン埋め込みを連結
            combined_features = torch.cat([resampled_encoder_out, token_embeddings], dim=-1)
            
            # 無効なパディング部分をマスク（ignore_idの位置）
            mask = text != self.ignore_id
            mask_float = mask.float()
            
            # 3種類の分類器で非流暢性予測
            results = {}
            
            # フィラー予測
            dysfl_logits = self.dysfl_classifier_filler(combined_features).squeeze(-1)
            dysfl_probs = self.sigmoid(dysfl_logits) * mask_float
            dysfl_preds = (dysfl_probs > 0.5).float() * mask_float
            results["filler"] = (dysfl_probs, dysfl_preds)
            
            # 言い直し予測
            dysfl_logits = self.dysfl_classifier_disfluency(combined_features).squeeze(-1)
            dysfl_probs = self.sigmoid(dysfl_logits) * mask_float
            dysfl_preds = (dysfl_probs > 0.5).float() * mask_float
            results["disfluency"] = (dysfl_probs, dysfl_preds)
            
            # 感動詞予測
            dysfl_logits = self.dysfl_classifier_interjection(combined_features).squeeze(-1)
            dysfl_probs = self.sigmoid(dysfl_logits) * mask_float
            dysfl_preds = (dysfl_probs > 0.5).float() * mask_float
            results["interjection"] = (dysfl_probs, dysfl_preds)
            
            logging.info(f"非流暢性予測完了: 3種類のカテゴリ全てについて予測しました")
            
            return results
            
        except Exception as e:
            logging.error(f"非流暢性予測中にエラーが発生: {e}")
            import traceback
            logging.error(traceback.format_exc())
            
            # エラー時は空の結果を返す
            dummy_output = torch.zeros(batch_size, max_length, device=speech.device)
            empty_results = {
                "filler": (dummy_output, dummy_output),
                "disfluency": (dummy_output, dummy_output),
                "interjection": (dummy_output, dummy_output)
            }
            return empty_results
