# espnet2/asr/espnet_dysfl_model.py
import torch
import logging

from espnet2.asr.espnet_model import ESPnetASRModel


class ESPnetASRDysflModel(ESPnetASRModel):
    """ESPnet ASR Model with dysfl classification."""

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
        print("ESPnetASRDysflModel is being initialized")  # デバッグ用
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
        # dysfl分類のための線形層を追加
        encoder_output_size = encoder.output_size()
        self.dysfl_classifier = torch.nn.Linear(encoder_output_size, 1)
        self.dysfl_weight = dysfl_weight
        self.sigmoid = torch.nn.Sigmoid()
        self.bce_loss = torch.nn.BCELoss(reduction="none")

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        isdysfl: torch.Tensor = None,
        **kwargs,
    ):
        """Forward pass with dysfl classification."""
        # 通常のASR処理を実行
        loss_asr, stats, weight = super().forward(
            speech=speech,
            speech_lengths=speech_lengths,
            text=text,
            text_lengths=text_lengths,
            **kwargs,
        )

        # isdysflが提供されている場合、dysfl分類を行う
        if isdysfl is not None:
            print(f"Speech batch size: {speech.size(0)}, isdysfl size: {isdysfl.size(0)}")
            
            batch_size = speech.size(0)
            # バッチサイズを一致させる
            if isdysfl.size(0) != batch_size:
                print(f"Resizing isdysfl from {isdysfl.size(0)} to {batch_size}")
                if isdysfl.size(0) > batch_size:
                    # 切り詰める
                    isdysfl = isdysfl[:batch_size]
                else:
                    # 足りない場合はゼロで埋める (または最後の値で埋める)
                    padding = torch.zeros(batch_size - isdysfl.size(0), 1, dtype=torch.long, device=isdysfl.device)
                    if isdysfl.size(0) > 0:
                        # 最後の値で埋める場合
                        if len(isdysfl.shape) == 1:
                            last_val = isdysfl[-1].item()
                        else:
                            last_val = isdysfl[-1, 0].item()
                        padding.fill_(last_val)
                    isdysfl = torch.cat([isdysfl, padding], dim=0)
                print(f"New isdysfl size: {isdysfl.size()}")
            
            # isdysflの形状を整える
            if len(isdysfl.shape) == 1:
                isdysfl = isdysfl.view(-1, 1)
                print(f"Reshaped isdysfl to: {isdysfl.size()}")
            
            # エンコーダの出力を取得
            encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
            if isinstance(encoder_out, tuple):
                encoder_out = encoder_out[0]
            
            # 最初のフレームを使用して分類
            first_frame = encoder_out[:, 0, :]
            dysfl_logits = self.dysfl_classifier(first_frame)
            dysfl_probs = self.sigmoid(dysfl_logits)
            
            print(f"dysfl_probs size: {dysfl_probs.size()}, isdysfl size: {isdysfl.size()}")
            
            # 適切な型に変換
            isdysfl = isdysfl.to(dtype=torch.float32, device=dysfl_probs.device)
            
            # BCELossの計算
            loss_dysfl = self.bce_loss(dysfl_probs, isdysfl)
            loss_dysfl = loss_dysfl.mean()
            
            # 予測精度を計算
            predictions = (dysfl_probs > 0.5).float()
            acc_dysfl = (predictions == isdysfl).float().mean()
            
            # 合計損失の計算
            loss = loss_asr + self.dysfl_weight * loss_dysfl
            
            # 統計情報の更新
            stats["loss_dysfl"] = loss_dysfl.detach()
            stats["acc_dysfl"] = acc_dysfl
        else:
            loss = loss_asr
        
        return loss, stats, weight
