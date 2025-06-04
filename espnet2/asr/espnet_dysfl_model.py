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
        
        # エンコーダの出力サイズ（フォールバック用）
        encoder_output_size = encoder.output_size()
        
        # デコーダの隠れ状態サイズを安全に取得
        decoder_hidden_size = None
        if hasattr(decoder, "dunits"):
            # Transducerデコーダの場合
            decoder_hidden_size = decoder.dunits
            print(f"Transducer decoder detected: hidden_size={decoder_hidden_size}")
        elif hasattr(decoder, "output_size"):
            # Transformer Decoderの場合
            try:
                decoder_hidden_size = decoder.output_size
                if callable(decoder_hidden_size):
                    decoder_hidden_size = decoder_hidden_size()
                print(f"Transformer decoder detected: hidden_size={decoder_hidden_size}")
            except:
                decoder_hidden_size = None
        
        # フォールバック：エンコーダのサイズを使用
        if decoder_hidden_size is None:
            decoder_hidden_size = encoder_output_size
            print(f"Warning: Could not determine decoder hidden size. Using encoder size: {decoder_hidden_size}")
        
        # トークン埋め込みへのアクセスを取得
        if hasattr(decoder, "embed"):
            self.token_embedding = decoder.embed
            token_embed_size = self.token_embedding.weight.shape[1]
            print(f"Using decoder embedding: embed_size={token_embed_size}")
        else:
            print("Warning: Decoder does not have 'embed' attribute. Using a new embedding layer.")
            token_embed_size = min(320, encoder_output_size // 2)  # より安全なサイズ
            self.token_embedding = torch.nn.Embedding(vocab_size, token_embed_size)
        
        # トークン依存メカニズム用の分類器（元論文のW[E(yi); si]に対応）
        self.num_dysfl_classes = num_dysfl_classes
        
        # 2つの設定を試す：理想的な設定とフォールバック設定
        try:
            # 理想的な設定：トークン埋め込み + デコーダ隠れ状態
            classifier_input_size = token_embed_size + decoder_hidden_size
            self.dysfl_classifier = torch.nn.Linear(classifier_input_size, num_dysfl_classes)
            print(f"Token-dependency classifier initialized: token_embed={token_embed_size} + decoder_hidden={decoder_hidden_size} = {classifier_input_size}")
        except Exception as e:
            print(f"Failed to create ideal classifier: {e}")
            # フォールバック設定：トークン埋め込み + エンコーダ出力
            classifier_input_size = token_embed_size + encoder_output_size
            self.dysfl_classifier = torch.nn.Linear(classifier_input_size, num_dysfl_classes)
            print(f"Fallback classifier initialized: token_embed={token_embed_size} + encoder_output={encoder_output_size} = {classifier_input_size}")
        
        self.dysfl_weight = dysfl_weight
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-1)
        
        # デバッグ情報を保存
        self.token_embed_size = token_embed_size
        self.decoder_hidden_size = decoder_hidden_size
        self.encoder_output_size = encoder_output_size
        
        print(f"Initialization complete: dysfl_weight={dysfl_weight}, num_classes={num_dysfl_classes}")

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
                # デバッグ情報
                if batch_size <= 2:  # 最初の数バッチのみ詳細ログ
                    print(f"Debug: batch_size={batch_size}, max_length={max_length}")
                    print(f"Debug: text.shape={text.shape}, text_lengths={text_lengths}")
                    print(f"Debug: text min/max: {text.min().item()}/{text.max().item()}")
                
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
                print(f"Error in dysfl preprocessing: {e}")
                # エラーが発生した場合、ダミーのラベルとマスクを作成
                dysfl_label = torch.full(
                    (batch_size, max_length), -1, dtype=torch.long, device=speech.device
                )
                mask = torch.arange(max_length, device=text_lengths.device)[None, :] < text_lengths[:, None]
                mask = mask.float()
            
            try:
                # エンコーダ出力を取得
                encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
                if isinstance(encoder_out, tuple):
                    encoder_out = encoder_out[0]
                
                if batch_size <= 2:  # 最初の数バッチのみ詳細ログ
                    print(f"Debug: encoder_out.shape={encoder_out.shape}")
                
                # 安全なトークン埋め込みの取得（元論文のE(yi)）
                vocab_size = self.token_embedding.weight.size(0)
                # パディングトークンや無効なインデックスを安全に処理
                safe_text = torch.where(
                    (text >= 0) & (text < vocab_size),
                    text,
                    torch.zeros_like(text)  # 無効なインデックスは0（通常は<blank>）に
                )
                
                if batch_size <= 2:  # 最初の数バッチのみ詳細ログ
                    print(f"Debug: safe_text min/max: {safe_text.min().item()}/{safe_text.max().item()}")
                    print(f"Debug: vocab_size={vocab_size}")
                
                token_embeddings = self.token_embedding(safe_text)  # [batch, max_length, embed_dim]
                if batch_size <= 2:
                    print(f"Debug: token_embeddings.shape={token_embeddings.shape}")
                
                # デコーダの隠れ状態を取得（元論文のsi）
                # Transducerかどうかを確認
                if hasattr(self, 'use_transducer_decoder') and self.use_transducer_decoder:
                    if batch_size <= 2:
                        print("Debug: Using Transducer decoder")
                    decoder_hidden_states = self._get_transducer_decoder_states_safe(
                        encoder_out, safe_text, text_lengths
                    )
                else:
                    if batch_size <= 2:
                        print("Debug: Using Transformer decoder")
                    decoder_hidden_states = self._get_transformer_decoder_states_safe(
                        encoder_out, encoder_out_lens, safe_text, text_lengths
                    )
                
                if batch_size <= 2:
                    print(f"Debug: decoder_hidden_states.shape={decoder_hidden_states.shape}")
                
                # 次元チェック
                if token_embeddings.size(-1) + decoder_hidden_states.size(-1) != self.dysfl_classifier.in_features:
                    if batch_size <= 2:
                        print(f"Warning: Feature dimension mismatch!")
                        print(f"Expected: {self.dysfl_classifier.in_features}")
                        print(f"Got: {token_embeddings.size(-1)} + {decoder_hidden_states.size(-1)} = {token_embeddings.size(-1) + decoder_hidden_states.size(-1)}")
                        print("Falling back to simplified approach using encoder output")
                    
                    # 簡略化されたフォールバック：エンコーダ出力を使用
                    resampled_encoder_out = self._resample_encoder_output(encoder_out, max_length)
                    combined_features = torch.cat([token_embeddings, resampled_encoder_out], dim=-1)
                else:
                    # トークン埋め込み + デコーダ隠れ状態を連結（元論文のW[E(yi); si]）
                    combined_features = torch.cat([token_embeddings, decoder_hidden_states], dim=-1)
                
                if batch_size <= 2:
                    print(f"Debug: combined_features.shape={combined_features.shape}")
                    print(f"Debug: dysfl_classifier.in_features={self.dysfl_classifier.in_features}")
                
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
                    
            except Exception as e:
                print(f"Error in dysfl forward pass: {e}")
                import traceback
                traceback.print_exc()
                # エラーが発生した場合は非流暢性損失をスキップ
                loss = loss_asr
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
        """推論時に非流暢性予測を行うメソッド（安全な実装）"""
        try:
            # エンコーダ出力を取得
            encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
            if isinstance(encoder_out, tuple):
                encoder_out = encoder_out[0]
            
            batch_size, max_length = text.size()
            
            # 安全なトークン埋め込みを取得
            vocab_size = self.token_embedding.weight.size(0)
            safe_text = torch.where(
                (text >= 0) & (text < vocab_size),
                text,
                torch.zeros_like(text)  # 無効なインデックスは0に
            )
            
            token_embeddings = self.token_embedding(safe_text)  # [batch, max_length, embed_dim]
            
            # デコーダの隠れ状態を取得（元論文のsi）
            if hasattr(self, 'use_transducer_decoder') and self.use_transducer_decoder:
                decoder_hidden_states = self._get_transducer_decoder_states_safe(
                    encoder_out, safe_text, text_lengths
                )
            else:
                decoder_hidden_states = self._get_transformer_decoder_states_safe(
                    encoder_out, encoder_out_lens, safe_text, text_lengths
                )
            
            # 次元チェックとフォールバック
            expected_dim = self.dysfl_classifier.in_features
            actual_dim = token_embeddings.size(-1) + decoder_hidden_states.size(-1)
            
            if actual_dim != expected_dim:
                logging.warning(f"次元不整合: 期待={expected_dim}, 実際={actual_dim}")
                logging.warning("エンコーダ出力を使用したフォールバックを適用")
                resampled_encoder_out = self._resample_encoder_output(encoder_out, max_length)
                combined_features = torch.cat([token_embeddings, resampled_encoder_out], dim=-1)
            else:
                # トークン埋め込み + デコーダ隠れ状態を連結（元論文のW[E(yi); si]）
                combined_features = torch.cat([token_embeddings, decoder_hidden_states], dim=-1)
            
            # 非流暢性ロジットを計算
            dysfl_logits = self.dysfl_classifier(combined_features)  # [batch, max_length, 4]
            
            # ソフトマックスで確率に変換
            dysfl_probs = torch.softmax(dysfl_logits, dim=-1)  # [batch, max_length, 4]
            
            # 最大確率のクラスを予測
            dysfl_preds = torch.argmax(dysfl_probs, dim=-1)  # [batch, max_length]
            
            # 無効なパディング部分をマスク
            mask = text != self.ignore_id
            dysfl_preds = dysfl_preds * mask.long()
            
            return dysfl_probs, dysfl_preds
            
        except Exception as e:
            logging.error(f"非流暢性予測中にエラーが発生: {e}")
            import traceback
            logging.error(traceback.format_exc())
            
            # エラー時は空の結果を返す
            batch_size, max_length = text.size()
            dummy_probs = torch.zeros(batch_size, max_length, self.num_dysfl_classes, device=speech.device)
            dummy_preds = torch.zeros(batch_size, max_length, device=speech.device, dtype=torch.long)
            return dummy_probs, dummy_preds
    
    def _get_transducer_decoder_states_safe(
        self, encoder_out: torch.Tensor, text: torch.Tensor, text_lengths: torch.Tensor
    ):
        """Transducerデコーダから各トークンの隠れ状態を安全に取得"""
        batch_size, max_length = text.size()
        
        try:
            # Transducerデコーダの場合、各トークンを順次処理
            # より現実的なアプローチ：単純にデコーダに入力を渡す
            
            # 最後のトークン（通常</s>）を除いた入力を作成
            if max_length > 1:
                decoder_input = text[:, :-1]  # [batch, max_length-1]
            else:
                decoder_input = text
            
            # デコーダの隠れ状態を取得
            decoder_out = self.decoder(decoder_input)  # [batch, seq_len, hidden_dim]
            
            # 元の長さに合わせるためのパディング処理
            if decoder_out.size(1) < max_length:
                # 最後の隠れ状態を複製してパディング
                last_state = decoder_out[:, -1:, :]  # [batch, 1, hidden_dim]
                padding_length = max_length - decoder_out.size(1)
                padding = last_state.repeat(1, padding_length, 1)
                decoder_out = torch.cat([decoder_out, padding], dim=1)
            elif decoder_out.size(1) > max_length:
                decoder_out = decoder_out[:, :max_length, :]
            
            return decoder_out
            
        except Exception as e:
            print(f"Warning: Transducer decoder states failed: {e}")
            # フォールバック：エンコーダ出力をリサンプリング
            return self._resample_encoder_output(encoder_out, max_length)
    
    def _get_transformer_decoder_states_safe(
        self, encoder_out: torch.Tensor, encoder_out_lens: torch.Tensor,
        text: torch.Tensor, text_lengths: torch.Tensor
    ):
        """Transformer Decoderから各トークンの隠れ状態を安全に取得"""
        batch_size, max_length = text.size()
        
        try:
            # SOSトークンを追加してデコーダ入力を作成
            if text.size(1) > 1:
                ys_in_pad = text[:, :-1]  # 最後の</s>を除く
                ys_in_lens = torch.clamp(text_lengths - 1, min=1)
            else:
                ys_in_pad = text
                ys_in_lens = text_lengths
            
            # デバッグ情報（最初のバッチのみ）
            if batch_size <= 2:
                print(f"Debug: ys_in_pad.shape={ys_in_pad.shape}, ys_in_lens={ys_in_lens}")
            
            # デコーダの隠れ状態を計算
            decoder_out, _ = self.decoder(
                encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
            )  # [batch, seq_len, hidden_dim]
            
            if batch_size <= 2:
                print(f"Debug: decoder_out.shape={decoder_out.shape}")
            
            # パディングして元の長さに合わせる
            if decoder_out.size(1) < max_length:
                padding = torch.zeros(
                    batch_size, 
                    max_length - decoder_out.size(1), 
                    decoder_out.size(2),
                    device=decoder_out.device,
                    dtype=decoder_out.dtype
                )
                decoder_out = torch.cat([decoder_out, padding], dim=1)
            elif decoder_out.size(1) > max_length:
                decoder_out = decoder_out[:, :max_length, :]
                
            return decoder_out
            
        except Exception as e:
            print(f"Warning: Transformer decoder states failed: {e}")
            if batch_size <= 2:
                import traceback
                traceback.print_exc()
            # フォールバック：エンコーダ出力をリサンプリング
            return self._resample_encoder_output(encoder_out, max_length)
    
    def _resample_encoder_output(self, encoder_out: torch.Tensor, target_length: int):
        """エンコーダ出力を目標長にリサンプリング（フォールバック用）"""
        batch_size, enc_length, enc_dim = encoder_out.size()
        
        if enc_length == target_length:
            return encoder_out
        
        # 線形補間を使ってリサンプリング
        resampled = torch.nn.functional.interpolate(
            encoder_out.transpose(1, 2),  # [batch, enc_dim, enc_length]
            size=target_length,
            mode='linear',
            align_corners=False
        ).transpose(1, 2)  # [batch, target_length, enc_dim]
        
        return resampled
