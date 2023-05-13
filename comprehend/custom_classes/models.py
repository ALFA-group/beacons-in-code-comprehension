import os
import pickle as pkl
import typing
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

class CodeModel(ABC):
    def __init__(self, base_path: Path) -> None:
        self._base_path = base_path

    @abstractmethod
    def fit_transform(self, programs: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

class DNN(CodeModel):
    def __init__(self, base_path: Path) -> None:
        super().__init__(base_path)

    @staticmethod
    def _get_rep(forward_output) -> np.ndarray:
        rep = forward_output[0].mean(axis=1)
        if rep.device != "cpu":
            rep = rep.cpu()
        return rep.detach().numpy().squeeze()

    @abstractmethod
    def _forward_pipeline(self, program: str) -> torch.Tensor:
        raise NotImplementedError()

    def fit_transform(self, programs: np.ndarray) -> np.ndarray:
        outputs = []
        for program in programs:
            outputs.append(self._get_rep(self._forward_pipeline(program)))
        return np.array(outputs)

class HFModel(DNN):
    def __init__(self, base_path: Path) -> None:
        super().__init__(base_path)
        cache_dir = self._base_path.joinpath(
            ".cache",
            "models",
            self._spec.split(os.sep)[0],
            self._spec.split(os.sep)[-1],
        )
        if not cache_dir.exists():
            cache_dir.mkdir(parents=True, exist_ok=True)
        self._tokenizer = AutoTokenizer.from_pretrained(self._spec, cache_dir=cache_dir)
        self._model = AutoModel.from_pretrained(self._spec, cache_dir=cache_dir)

    @property
    @abstractmethod
    def _spec(self) -> str:
        raise NotImplementedError()

    def _forward_pipeline(self, program: str) -> torch.Tensor:
        return self._model.forward(self._tokenizer.encode(program, return_tensors="pt"))

class CodeBERT(HFModel):
    def __init__(self, base_path: Path) -> None:
        super().__init__(base_path)

    @property
    def _spec(self) -> str:
        return "microsoft/codebert-base-mlm"


class CodeGPT2(HFModel):
    def __init__(self, base_path: Path) -> None:
        super().__init__(base_path)

    @property
    def _spec(self) -> str:
        return "microsoft/CodeGPT-small-py"


class CodeBERTa(HFModel):
    def __init__(self, base_path: Path) -> None:
        super().__init__(base_path)

    @property
    def _spec(self) -> str:
        return "huggingface/CodeBERTa-small-v1"