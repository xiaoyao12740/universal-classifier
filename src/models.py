from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def build_model(model_type: str):
    model_type = model_type.lower()

    if model_type == "svm":
        return SVC(kernel="rbf", C=2.0, gamma="scale")
    if model_type == "random_forest":
        return RandomForestClassifier(n_estimators=120, random_state=42)
    if model_type == "knn":
        return KNeighborsClassifier(n_neighbors=5)

    supported = "svm, random_forest, knn"
    raise ValueError(f"Unsupported model type '{model_type}'. Supported: {supported}")
