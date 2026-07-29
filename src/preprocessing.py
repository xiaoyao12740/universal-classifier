from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(features):
    numeric_columns = features.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [column for column in features.columns if column not in numeric_columns]

    transformers = []
    if numeric_columns:
        transformers.append(("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric_columns))
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")),
                categorical_columns,
            )
        )

    return ColumnTransformer(transformers=transformers)
