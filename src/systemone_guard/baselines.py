"""Fair TF-IDF baselines receive BOTH natural language and serialized state."""

from collections.abc import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .data import CLASSES
from .types import DecisionInput, FloatArray, LabeledInput


def serialize(r: DecisionInput) -> str:
    return (
        "request: "
        + r["request"]
        + " context: "
        + r["context"]
        + " "
        + " ".join(
            f"{k}={r[k]}"
            for k in ("action", "sensitivity", "destination", "authorization", "source")
        )
    )


def fit(train: Sequence[LabeledInput], ngram: str = "word") -> Pipeline:
    v = (
        TfidfVectorizer(
            analyzer="char", ngram_range=(2, 5), max_features=25000, sublinear_tf=True
        )
        if ngram == "char"
        else TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=10000)
    )
    pipe = Pipeline(
        [("vec", v), ("clf", LogisticRegression(max_iter=700, random_state=42))]
    )
    pipe.fit([serialize(x) for x in train], [x["decision"] for x in train])

    return pipe


def probs(model: Pipeline, rows: Sequence[DecisionInput]) -> FloatArray:
    p = model.predict_proba([serialize(x) for x in rows])

    return np.asarray(p[:, [list(model.classes_).index(c) for c in CLASSES]])
