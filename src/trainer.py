from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from models import build_model
from preprocessing import build_preprocessor


def train_classifier(features, labels, model_type, test_size, random_state):
    stratify = labels if labels.nunique() > 1 else None
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    pipeline = make_pipeline(
        build_preprocessor(features),
        build_model(model_type),
    )
    pipeline.fit(x_train, y_train)
    return pipeline, x_test, y_test
