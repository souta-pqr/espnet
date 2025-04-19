import logging

from espnet2.tasks.asr import ASRTask
from espnet2.train.preprocessor import CommonPreprocessor


class ASRDysflTask(ASRTask):
    """ASR task with dysfl classification."""

    @classmethod
    def required_data_names(cls, train: bool = True, inference: bool = False):
        if not inference:
            retval = ("speech", "text")
        else:
            # 推論モード
            retval = ("speech",)
        return retval

    @classmethod
    def optional_data_names(cls, train: bool = True, inference: bool = False):
        retval = list(super().optional_data_names(train, inference))
        # isdysflをオプションデータとして追加
        retval.append("isdysfl")
        return tuple(retval)

    @classmethod
    def build_model(cls, args):
        # 親クラスを使用してモデルをビルド
        model = super().build_model(args)
        # モデルのクラスを明示的に変更（model_classの設定で上書きされる場合も）
        return model
