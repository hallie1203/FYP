import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb
from torch_geometric.datasets import EllipticBitcoinDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from lightgbm import LGBMClassifier


def compute_roc_auc(y_true, y_prob):
    unique_classes = np.unique(y_true)
    if len(unique_classes) > 2:
        # For multi-class, use the full probability matrix with one-vs-rest.
        return roc_auc_score(y_true, y_prob, multi_class='ovr')
    else:
        return roc_auc_score(y_true, y_prob)


def run_classifiers(X_train, X_val, X_test, y_train, y_val, y_test):
    # Determine if the task is binary or multi-class.
    num_classes = len(np.unique(y_test))

    def visualize(name, y, y_pred):
        print("\n")
        plt.figure(figsize=(6, 6))
        sns.heatmap(confusion_matrix(y, y_pred), annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title(name + " Test Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

    # ----- LDA -----
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)
    y_pred_lda = lda.predict(X_test)
    print("LDA Classification Report:")
    print(classification_report(y_test, y_pred_lda))
    print("LDA Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_lda))
    if hasattr(lda, "predict_proba"):
        if num_classes == 2:
            y_prob_lda = lda.predict_proba(X_test)[:, 1]
        else:
            y_prob_lda = lda.predict_proba(X_test)
        print("LDA ROC AUC Score:", compute_roc_auc(y_test, y_prob_lda))
    visualize("LDA", y_test, y_pred_lda)

    # ----- QDA -----
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(X_train, y_train)
    y_pred_qda = qda.predict(X_test)
    print("QDA Classification Report:")
    print(classification_report(y_test, y_pred_qda))
    print("QDA Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_qda))
    if hasattr(qda, "predict_proba"):
        if num_classes == 2:
            y_prob_qda = qda.predict_proba(X_test)[:, 1]
        else:
            y_prob_qda = qda.predict_proba(X_test)
        print("QDA ROC AUC Score:", compute_roc_auc(y_test, y_prob_qda))
    visualize("QDA", y_test, y_pred_qda)

    # ----- AdaBoost -----
    ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=50, random_state=42)
    ada.fit(X_train, y_train)
    y_pred_ada = ada.predict(X_test)
    print("AdaBoost Classification Report:")
    print(classification_report(y_test, y_pred_ada))
    print("AdaBoost Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_ada))
    if hasattr(ada, "predict_proba"):
        if num_classes == 2:
            y_prob_ada = ada.predict_proba(X_test)[:, 1]
        else:
            y_prob_ada = ada.predict_proba(X_test)
        print("AdaBoost ROC AUC Score:", compute_roc_auc(y_test, y_prob_ada))
    visualize("AdaBoost", y_test, y_pred_ada)

    # ----- LightGBM -----
    lgbm = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=10, class_weight='balanced', random_state=42)
    lgbm.fit(X_train, y_train)
    y_pred_lgbm = lgbm.predict(X_test)
    y_prob = lgbm.predict_proba(X_test)[:, 1]
    print("\nLightGBM Classification Report:\n", classification_report(y_test, y_pred_lgbm, zero_division=0))
    print("LightGBM Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lgbm))
    print("LightGBM ROC AUC Score:", roc_auc_score(y_test, y_prob))
    visualize("LightGBM", y_test, y_pred_lgbm)


if __name__ == '__main__':
    # ------------------------
    # 1. Load the Elliptic Bitcoin Dataset
    # ------------------------
    dataset = EllipticBitcoinDataset(root='data/EllipticBitcoin')
    data = dataset[0]  # The dataset contains a single graph.
    print("Dataset Summary:")
    print(data)

    # ------------------------
    # 2. Extract Node Features and Labels
    # ------------------------
    # data.x: Node feature matrix; data.y: Node labels.
    X = data.x.cpu().numpy()  # shape: (num_nodes, num_features)
    y = data.y.cpu().numpy()  # shape: (num_nodes,)
    mask = (y != 2)  # Filter out 'unknown' class (label 2)
    X = X[mask]
    y = y[mask]
    print("Unique labels in dataset:", np.unique(y))

    # ------------------------
    # 3. Train/Validation/Test Split
    # ------------------------
    # Stratify the split to preserve the original class distribution.
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=0.01, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    print("Training set shape:", X_train.shape)
    print("Validation set shape:", X_val.shape)
    print("Test set shape:", X_test.shape)

    # ------------------------
    # 4. Run Classifiers and Evaluate
    # ------------------------
    run_classifiers(X_train, X_val, X_test, y_train, y_val, y_test)