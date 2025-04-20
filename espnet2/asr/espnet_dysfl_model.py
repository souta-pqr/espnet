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
        print("ESPnetASRDysflModel is being initialized")
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
        # 単一の二値分類タスクを行うための分類器
        encoder_output_size = encoder.output_size()
        self.dysfl_classifier = torch.nn.Linear(encoder_output_size, 1)
        self.dysfl_weight = dysfl_weight
        self.sigmoid = torch.nn.Sigmoid()
        # BCEWithLogitsLossを使用するとよりロバストに処理できる
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
            batch_size = speech.size(0)
            
            # デバッグ情報
            print(f"isdysfl原型: {isdysfl.shape}, {isdysfl.dtype}")
            
            # isdysflから各発話に対する単一ラベルを生成
            # 各発話のラベルを1つにまとめる（1が一つでも含まれていれば1とする）
            try:
                # 最初にテンソルに変換
                if not isinstance(isdysfl, torch.Tensor):
                    isdysfl = torch.tensor(isdysfl, device=speech.device)
                
                # 形状を調べる
                if len(isdysfl.shape) == 1:
                    # 単一の値なら、バッチサイズに合わせて拡張
                    dysfl_label = isdysfl.float().view(-1, 1)
                elif len(isdysfl.shape) == 2:
                    # すでに[バッチ, 可変長]の形式なので、各行の最大値を取る（1があれば1、なければ0）
                    dysfl_label = torch.max(isdysfl, dim=1, keepdim=True)[0].float()
                else:
                    # 想定外の形状の場合、単純に1次元目の最初の値を使用
                    dysfl_label = isdysfl[:, 0].float().view(-1, 1)
                
                # バッチサイズが一致しない場合は調整
                if dysfl_label.size(0) != batch_size:
                    if dysfl_label.size(0) > batch_size:
                        dysfl_label = dysfl_label[:batch_size]
                    else:
                        padding = torch.zeros(
                            batch_size - dysfl_label.size(0), 1,
                            dtype=torch.float, device=dysfl_label.device
                        )
                        dysfl_label = torch.cat([dysfl_label, padding], dim=0)
                
                # 値を0と1に制限
                dysfl_label = torch.clamp(dysfl_label, 0.0, 1.0)
                
                print(f"処理後dysfl_label: {dysfl_label.shape}, {dysfl_label.dtype}, 値範囲: [{dysfl_label.min()}, {dysfl_label.max()}]")
            except Exception as e:
                print(f"isdysfl処理エラー: {e}")
                # エラーが発生した場合はダミーラベルを作成
                dysfl_label = torch.zeros(batch_size, 1, dtype=torch.float, device=speech.device)
            
            # エンコーダの出力を取得
            encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
            if isinstance(encoder_out, tuple):
                encoder_out = encoder_out[0]
            
            # 最初のフレームを使用して分類
            first_frame = encoder_out[:, 0, :]
            dysfl_logits = self.dysfl_classifier(first_frame)
            
            # 分類器の出力をそのままBCEWithLogitsLossに渡す
            loss_dysfl = self.bce_loss(dysfl_logits, dysfl_label).mean()
            
            # 予測精度を計算用にシグモイド変換
            dysfl_probs = self.sigmoid(dysfl_logits)
            predictions = (dysfl_probs > 0.5).float()
            acc_dysfl = (predictions == dysfl_label).float().mean()
            
            # 合計損失の計算
            loss = loss_asr + self.dysfl_weight * loss_dysfl
            
            # 統計情報の更新
            stats["loss_dysfl"] = loss_dysfl.detach()
            stats["acc_dysfl"] = acc_dysfl
        else:
            loss = loss_asr
        
        return loss, stats, weight
