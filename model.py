"""
MNIST Classifier Pipeline with Scikit-Learn

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_mnist
import os
import tempfile
import urllib.request

import joblib
import numpy as np

from sklearn.base import clone
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. 加载 MNIST
# ============================================================

def load_mnist(n_train=10000, n_test=2000):
    """
    下载并缓存 MNIST，返回训练集和测试集的前若干行。

    图片保留 0~255 像素范围，摊平为 784 维。
    """
    url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
    path = os.path.join(tempfile.gettempdir(), "mnist.npz")

    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)

    with np.load(path) as data:
        X_train = data["x_train"][:n_train].reshape(-1, 784).astype(np.float32)
        y_train = data["y_train"][:n_train].astype(np.int64)

        X_test = data["x_test"][:n_test].reshape(-1, 784).astype(np.float32)
        y_test = data["y_test"][:n_test].astype(np.int64)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
    }


# ============================================================
# 2-4. “是不是 5”的二分类和交叉验证预测
# ============================================================

def binary_target(y, digit=5):
    """标签等于 digit 的位置为 True，其余为 False。"""
    return np.asarray(y) == digit


def train_sgd(X, y, random_state=42):
    """训练并返回 SGDClassifier。"""
    clf = SGDClassifier(random_state=random_state)
    clf.fit(X, y)
    return clf


def cross_val_predictions(clf, X, y, cv=3, method="predict"):
    """
    生成交叉验证的 out-of-fold 预测或分数。

    每个样本的预测都来自一个没有用该样本训练过的模型。
    """
    return cross_val_predict(
        clone(clf),
        X,
        y,
        cv=cv,
        method=method,
    )


# ============================================================
# 5-6. 二分类混淆矩阵和指标
# ============================================================

def confusion_counts(y_true, y_pred):
    """
    返回固定类别顺序 [False, True] 下的混淆矩阵计数：
    TN、FP、FN、TP。
    """
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[False, True],
    )

    tn, fp, fn, tp = matrix.ravel()

    return {
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def precision_recall_f1(y_true, y_pred):
    """返回 precision、recall、F1。"""
    precision = precision_score(
        y_true, y_pred, zero_division=0
    )
    recall = recall_score(
        y_true, y_pred, zero_division=0
    )
    f1 = f1_score(
        y_true, y_pred, zero_division=0
    )

    return float(precision), float(recall), float(f1)


# ============================================================
# 7-9. 调阈值、按阈值评估和 ROC-AUC
# ============================================================

def threshold_for_precision(y_true, scores, target=0.90):
    """
    找到第一个能达到目标 precision 的 decision score 阈值。
    """
    precisions, thresholds = precision_recall_curve(y_true, scores)

    # precision_recall_curve 返回的 precisions 比 thresholds 多一个值；
    # 最后一个 precision 没有对应阈值，所以只看 precisions[:-1]。
    eligible = np.flatnonzero(precisions[:-1] >= target)

    if eligible.size == 0:
        return float("inf")

    return float(thresholds[eligible[0]])


def evaluate_at_threshold(y_true, scores, threshold):
    """按给定分数阈值分类，并返回分类指标和预测为正类的数量。"""
    y_pred = np.asarray(scores) >= threshold
    precision, recall, f1 = precision_recall_f1(y_true, y_pred)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "positives": int(np.sum(y_pred)),
    }


def roc_auc(y_true, scores):
    """返回 ROC-AUC 和达到至少 90% 召回率时的 FPR。"""
    auc = roc_auc_score(y_true, scores)
    fpr, tpr, _ = roc_curve(y_true, scores)

    indices = np.flatnonzero(tpr >= 0.90)
    fpr_at_recall_90 = float(fpr[indices[0]]) if indices.size else 1.0

    return {
        "auc": float(auc),
        "fpr_at_recall_90": fpr_at_recall_90,
    }


# ============================================================
# 10-13. 多分类 pipeline、交叉验证和混淆分析
# ============================================================

def multiclass_pipeline(random_state=42):
    """
    返回尚未训练的流水线：
    StandardScaler -> SGDClassifier。
    """
    return make_pipeline(
        StandardScaler(),
        SGDClassifier(random_state=random_state),
    )


def multiclass_cv_accuracy(model, X, y, cv=3):
    """返回交叉验证平均 accuracy。"""
    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="accuracy",
    )

    return float(np.mean(scores))


def normalized_confusion(y_true, y_pred):
    """返回固定为 10×10、按真实类别行归一化的混淆矩阵。"""
    return confusion_matrix(
        y_true,
        y_pred,
        labels=range(10),
        normalize="true",
    )


def most_confused_pairs(cm, k=3):
    """
    返回最大的 k 个非对角线混淆项：
    (真实类别, 预测类别, 混淆比例)。
    """
    cm = np.asarray(cm)
    candidates = []

    for true_class in range(cm.shape[0]):
        for predicted_class in range(cm.shape[1]):
            if true_class != predicted_class:
                candidates.append(
                    (
                        true_class,
                        predicted_class,
                        float(cm[true_class, predicted_class]),
                    )
                )

    # 比例从大到小；相同则按类别编号从小到大，保持确定性
    candidates.sort(key=lambda item: (-item[2], item[0], item[1]))

    return [
        (int(true_class), int(predicted_class), round(rate, 3))
        for true_class, predicted_class, rate in candidates[:max(0, k)]
    ]


# ============================================================
# 14-15. 多标签分类
# ============================================================

def multilabel_targets(y):
    """
    为每个数字生成两个布尔标签：
    第 0 列：数字 >= 7
    第 1 列：数字是奇数
    """
    y = np.asarray(y)

    return np.column_stack([
        y >= 7,
        y % 2 == 1,
    ]).astype(bool)


def multilabel_knn(X, Y, n_neighbors=5):
    """训练并返回 KNN 多标签分类器。"""
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X, Y)
    return clf


def multilabel_f1(Y_true, Y_pred):
    """计算各标签 F1 的宏平均。"""
    return float(
        f1_score(
            Y_true,
            Y_pred,
            average="macro",
            zero_division=0,
        )
    )


# ============================================================
# 16-18. 最终测试、保存/加载、预测原始图片
# ============================================================

def final_test_accuracy(model, X_test, y_test):
    """在已经拟合好的模型上计算测试准确率，不重新训练。"""
    predictions = model.predict(X_test)
    return float(accuracy_score(y_test, predictions))


def save_and_reload_classifier(model, path):
    """用 joblib 保存模型，再加载并返回加载后的模型。"""
    joblib.dump(model, path)
    return joblib.load(path)


def predict_digits(model, images):
    """
    输入原始图片 (n, 28, 28)，摊平后送入已拟合的模型。
    返回由 Python int 组成的列表。
    """
    images = np.asarray(images)
    X = images.reshape(len(images), 784).astype(np.float32)

    predictions = model.predict(X)
    return [int(label) for label in predictions]

# Step 2 - binary_target
import os
import tempfile
import urllib.request

import joblib
import numpy as np

from sklearn.base import clone
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. 加载 MNIST
# ============================================================

def load_mnist(n_train=10000, n_test=2000):
    """
    下载并缓存 MNIST，返回训练集和测试集的前若干行。

    图片保留 0~255 像素范围，摊平为 784 维。
    """
    url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
    path = os.path.join(tempfile.gettempdir(), "mnist.npz")

    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)

    with np.load(path) as data:
        X_train = data["x_train"][:n_train].reshape(-1, 784).astype(np.float32)
        y_train = data["y_train"][:n_train].astype(np.int64)

        X_test = data["x_test"][:n_test].reshape(-1, 784).astype(np.float32)
        y_test = data["y_test"][:n_test].astype(np.int64)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
    }


# ============================================================
# 2-4. “是不是 5”的二分类和交叉验证预测
# ============================================================

def binary_target(y, digit=5):
    """标签等于 digit 的位置为 True，其余为 False。"""
    return np.asarray(y) == digit


def train_sgd(X, y, random_state=42):
    """训练并返回 SGDClassifier。"""
    clf = SGDClassifier(random_state=random_state)
    clf.fit(X, y)
    return clf


def cross_val_predictions(clf, X, y, cv=3, method="predict"):
    """
    生成交叉验证的 out-of-fold 预测或分数。

    每个样本的预测都来自一个没有用该样本训练过的模型。
    """
    return cross_val_predict(
        clone(clf),
        X,
        y,
        cv=cv,
        method=method,
    )


# ============================================================
# 5-6. 二分类混淆矩阵和指标
# ============================================================

def confusion_counts(y_true, y_pred):
    """
    返回固定类别顺序 [False, True] 下的混淆矩阵计数：
    TN、FP、FN、TP。
    """
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[False, True],
    )

    tn, fp, fn, tp = matrix.ravel()

    return {
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def precision_recall_f1(y_true, y_pred):
    """返回 precision、recall、F1。"""
    precision = precision_score(
        y_true, y_pred, zero_division=0
    )
    recall = recall_score(
        y_true, y_pred, zero_division=0
    )
    f1 = f1_score(
        y_true, y_pred, zero_division=0
    )

    return float(precision), float(recall), float(f1)


# ============================================================
# 7-9. 调阈值、按阈值评估和 ROC-AUC
# ============================================================

def threshold_for_precision(y_true, scores, target=0.90):
    """
    找到第一个能达到目标 precision 的 decision score 阈值。
    """
    precisions, thresholds = precision_recall_curve(y_true, scores)

    # precision_recall_curve 返回的 precisions 比 thresholds 多一个值；
    # 最后一个 precision 没有对应阈值，所以只看 precisions[:-1]。
    eligible = np.flatnonzero(precisions[:-1] >= target)

    if eligible.size == 0:
        return float("inf")

    return float(thresholds[eligible[0]])


def evaluate_at_threshold(y_true, scores, threshold):
    """按给定分数阈值分类，并返回分类指标和预测为正类的数量。"""
    y_pred = np.asarray(scores) >= threshold
    precision, recall, f1 = precision_recall_f1(y_true, y_pred)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "positives": int(np.sum(y_pred)),
    }


def roc_auc(y_true, scores):
    """返回 ROC-AUC 和达到至少 90% 召回率时的 FPR。"""
    auc = roc_auc_score(y_true, scores)
    fpr, tpr, _ = roc_curve(y_true, scores)

    indices = np.flatnonzero(tpr >= 0.90)
    fpr_at_recall_90 = float(fpr[indices[0]]) if indices.size else 1.0

    return {
        "auc": float(auc),
        "fpr_at_recall_90": fpr_at_recall_90,
    }


# ============================================================
# 10-13. 多分类 pipeline、交叉验证和混淆分析
# ============================================================

def multiclass_pipeline(random_state=42):
    """
    返回尚未训练的流水线：
    StandardScaler -> SGDClassifier。
    """
    return make_pipeline(
        StandardScaler(),
        SGDClassifier(random_state=random_state),
    )


def multiclass_cv_accuracy(model, X, y, cv=3):
    """返回交叉验证平均 accuracy。"""
    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="accuracy",
    )

    return float(np.mean(scores))


def normalized_confusion(y_true, y_pred):
    """返回固定为 10×10、按真实类别行归一化的混淆矩阵。"""
    return confusion_matrix(
        y_true,
        y_pred,
        labels=range(10),
        normalize="true",
    )


def most_confused_pairs(cm, k=3):
    """
    返回最大的 k 个非对角线混淆项：
    (真实类别, 预测类别, 混淆比例)。
    """
    cm = np.asarray(cm)
    candidates = []

    for true_class in range(cm.shape[0]):
        for predicted_class in range(cm.shape[1]):
            if true_class != predicted_class:
                candidates.append(
                    (
                        true_class,
                        predicted_class,
                        float(cm[true_class, predicted_class]),
                    )
                )

    # 比例从大到小；相同则按类别编号从小到大，保持确定性
    candidates.sort(key=lambda item: (-item[2], item[0], item[1]))

    return [
        (int(true_class), int(predicted_class), round(rate, 3))
        for true_class, predicted_class, rate in candidates[:max(0, k)]
    ]


# ============================================================
# 14-15. 多标签分类
# ============================================================

def multilabel_targets(y):
    """
    为每个数字生成两个布尔标签：
    第 0 列：数字 >= 7
    第 1 列：数字是奇数
    """
    y = np.asarray(y)

    return np.column_stack([
        y >= 7,
        y % 2 == 1,
    ]).astype(bool)


def multilabel_knn(X, Y, n_neighbors=5):
    """训练并返回 KNN 多标签分类器。"""
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X, Y)
    return clf


def multilabel_f1(Y_true, Y_pred):
    """计算各标签 F1 的宏平均。"""
    return float(
        f1_score(
            Y_true,
            Y_pred,
            average="macro",
            zero_division=0,
        )
    )


# ============================================================
# 16-18. 最终测试、保存/加载、预测原始图片
# ============================================================

def final_test_accuracy(model, X_test, y_test):
    """在已经拟合好的模型上计算测试准确率，不重新训练。"""
    predictions = model.predict(X_test)
    return float(accuracy_score(y_test, predictions))


def save_and_reload_classifier(model, path):
    """用 joblib 保存模型，再加载并返回加载后的模型。"""
    joblib.dump(model, path)
    return joblib.load(path)


def predict_digits(model, images):
    """
    输入原始图片 (n, 28, 28)，摊平后送入已拟合的模型。
    返回由 Python int 组成的列表。
    """
    images = np.asarray(images)
    X = images.reshape(len(images), 784).astype(np.float32)

    predictions = model.predict(X)
    return [int(label) for label in predictions]

# Step 3 - train_sgd
import os
import tempfile
import urllib.request

import joblib
import numpy as np

from sklearn.base import clone
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. 加载 MNIST
# ============================================================

def load_mnist(n_train=10000, n_test=2000):
    """
    下载并缓存 MNIST，返回训练集和测试集的前若干行。

    图片保留 0~255 像素范围，摊平为 784 维。
    """
    url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
    path = os.path.join(tempfile.gettempdir(), "mnist.npz")

    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)

    with np.load(path) as data:
        X_train = data["x_train"][:n_train].reshape(-1, 784).astype(np.float32)
        y_train = data["y_train"][:n_train].astype(np.int64)

        X_test = data["x_test"][:n_test].reshape(-1, 784).astype(np.float32)
        y_test = data["y_test"][:n_test].astype(np.int64)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
    }


# ============================================================
# 2-4. “是不是 5”的二分类和交叉验证预测
# ============================================================

def binary_target(y, digit=5):
    """标签等于 digit 的位置为 True，其余为 False。"""
    return np.asarray(y) == digit


def train_sgd(X, y, random_state=42):
    """训练并返回 SGDClassifier。"""
    clf = SGDClassifier(random_state=random_state)
    clf.fit(X, y)
    return clf


def cross_val_predictions(clf, X, y, cv=3, method="predict"):
    """
    生成交叉验证的 out-of-fold 预测或分数。

    每个样本的预测都来自一个没有用该样本训练过的模型。
    """
    return cross_val_predict(
        clone(clf),
        X,
        y,
        cv=cv,
        method=method,
    )


# ============================================================
# 5-6. 二分类混淆矩阵和指标
# ============================================================

def confusion_counts(y_true, y_pred):
    """
    返回固定类别顺序 [False, True] 下的混淆矩阵计数：
    TN、FP、FN、TP。
    """
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[False, True],
    )

    tn, fp, fn, tp = matrix.ravel()

    return {
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def precision_recall_f1(y_true, y_pred):
    """返回 precision、recall、F1。"""
    precision = precision_score(
        y_true, y_pred, zero_division=0
    )
    recall = recall_score(
        y_true, y_pred, zero_division=0
    )
    f1 = f1_score(
        y_true, y_pred, zero_division=0
    )

    return float(precision), float(recall), float(f1)


# ============================================================
# 7-9. 调阈值、按阈值评估和 ROC-AUC
# ============================================================

def threshold_for_precision(y_true, scores, target=0.90):
    """
    找到第一个能达到目标 precision 的 decision score 阈值。
    """
    precisions, thresholds = precision_recall_curve(y_true, scores)

    # precision_recall_curve 返回的 precisions 比 thresholds 多一个值；
    # 最后一个 precision 没有对应阈值，所以只看 precisions[:-1]。
    eligible = np.flatnonzero(precisions[:-1] >= target)

    if eligible.size == 0:
        return float("inf")

    return float(thresholds[eligible[0]])


def evaluate_at_threshold(y_true, scores, threshold):
    """按给定分数阈值分类，并返回分类指标和预测为正类的数量。"""
    y_pred = np.asarray(scores) >= threshold
    precision, recall, f1 = precision_recall_f1(y_true, y_pred)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "positives": int(np.sum(y_pred)),
    }


def roc_auc(y_true, scores):
    """返回 ROC-AUC 和达到至少 90% 召回率时的 FPR。"""
    auc = roc_auc_score(y_true, scores)
    fpr, tpr, _ = roc_curve(y_true, scores)

    indices = np.flatnonzero(tpr >= 0.90)
    fpr_at_recall_90 = float(fpr[indices[0]]) if indices.size else 1.0

    return {
        "auc": float(auc),
        "fpr_at_recall_90": fpr_at_recall_90,
    }


# ============================================================
# 10-13. 多分类 pipeline、交叉验证和混淆分析
# ============================================================

def multiclass_pipeline(random_state=42):
    """
    返回尚未训练的流水线：
    StandardScaler -> SGDClassifier。
    """
    return make_pipeline(
        StandardScaler(),
        SGDClassifier(random_state=random_state),
    )


def multiclass_cv_accuracy(model, X, y, cv=3):
    """返回交叉验证平均 accuracy。"""
    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="accuracy",
    )

    return float(np.mean(scores))


def normalized_confusion(y_true, y_pred):
    """返回固定为 10×10、按真实类别行归一化的混淆矩阵。"""
    return confusion_matrix(
        y_true,
        y_pred,
        labels=range(10),
        normalize="true",
    )


def most_confused_pairs(cm, k=3):
    """
    返回最大的 k 个非对角线混淆项：
    (真实类别, 预测类别, 混淆比例)。
    """
    cm = np.asarray(cm)
    candidates = []

    for true_class in range(cm.shape[0]):
        for predicted_class in range(cm.shape[1]):
            if true_class != predicted_class:
                candidates.append(
                    (
                        true_class,
                        predicted_class,
                        float(cm[true_class, predicted_class]),
                    )
                )

    # 比例从大到小；相同则按类别编号从小到大，保持确定性
    candidates.sort(key=lambda item: (-item[2], item[0], item[1]))

    return [
        (int(true_class), int(predicted_class), round(rate, 3))
        for true_class, predicted_class, rate in candidates[:max(0, k)]
    ]


# ============================================================
# 14-15. 多标签分类
# ============================================================

def multilabel_targets(y):
    """
    为每个数字生成两个布尔标签：
    第 0 列：数字 >= 7
    第 1 列：数字是奇数
    """
    y = np.asarray(y)

    return np.column_stack([
        y >= 7,
        y % 2 == 1,
    ]).astype(bool)


def multilabel_knn(X, Y, n_neighbors=5):
    """训练并返回 KNN 多标签分类器。"""
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X, Y)
    return clf


def multilabel_f1(Y_true, Y_pred):
    """计算各标签 F1 的宏平均。"""
    return float(
        f1_score(
            Y_true,
            Y_pred,
            average="macro",
            zero_division=0,
        )
    )


# ============================================================
# 16-18. 最终测试、保存/加载、预测原始图片
# ============================================================

def final_test_accuracy(model, X_test, y_test):
    """在已经拟合好的模型上计算测试准确率，不重新训练。"""
    predictions = model.predict(X_test)
    return float(accuracy_score(y_test, predictions))


def save_and_reload_classifier(model, path):
    """用 joblib 保存模型，再加载并返回加载后的模型。"""
    joblib.dump(model, path)
    return joblib.load(path)


def predict_digits(model, images):
    """
    输入原始图片 (n, 28, 28)，摊平后送入已拟合的模型。
    返回由 Python int 组成的列表。
    """
    images = np.asarray(images)
    X = images.reshape(len(images), 784).astype(np.float32)

    predictions = model.predict(X)
    return [int(label) for label in predictions]

# Step 4 - cross_val_predictions
import os
import tempfile
import urllib.request

import joblib
import numpy as np

from sklearn.base import clone
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. 加载 MNIST
# ============================================================

def load_mnist(n_train=10000, n_test=2000):
    """
    下载并缓存 MNIST，返回训练集和测试集的前若干行。

    图片保留 0~255 像素范围，摊平为 784 维。
    """
    url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
    path = os.path.join(tempfile.gettempdir(), "mnist.npz")

    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)

    with np.load(path) as data:
        X_train = data["x_train"][:n_train].reshape(-1, 784).astype(np.float32)
        y_train = data["y_train"][:n_train].astype(np.int64)

        X_test = data["x_test"][:n_test].reshape(-1, 784).astype(np.float32)
        y_test = data["y_test"][:n_test].astype(np.int64)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
    }


# ============================================================
# 2-4. “是不是 5”的二分类和交叉验证预测
# ============================================================

def binary_target(y, digit=5):
    """标签等于 digit 的位置为 True，其余为 False。"""
    return np.asarray(y) == digit


def train_sgd(X, y, random_state=42):
    """训练并返回 SGDClassifier。"""
    clf = SGDClassifier(random_state=random_state)
    clf.fit(X, y)
    return clf


def cross_val_predictions(clf, X, y, cv=3, method="predict"):
    """
    生成交叉验证的 out-of-fold 预测或分数。

    每个样本的预测都来自一个没有用该样本训练过的模型。
    """
    return cross_val_predict(
        clone(clf),
        X,
        y,
        cv=cv,
        method=method,
    )


# ============================================================
# 5-6. 二分类混淆矩阵和指标
# ============================================================

def confusion_counts(y_true, y_pred):
    """
    返回固定类别顺序 [False, True] 下的混淆矩阵计数：
    TN、FP、FN、TP。
    """
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[False, True],
    )

    tn, fp, fn, tp = matrix.ravel()

    return {
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def precision_recall_f1(y_true, y_pred):
    """返回 precision、recall、F1。"""
    precision = precision_score(
        y_true, y_pred, zero_division=0
    )
    recall = recall_score(
        y_true, y_pred, zero_division=0
    )
    f1 = f1_score(
        y_true, y_pred, zero_division=0
    )

    return float(precision), float(recall), float(f1)


# ============================================================
# 7-9. 调阈值、按阈值评估和 ROC-AUC
# ============================================================

def threshold_for_precision(y_true, scores, target=0.90):
    """
    找到第一个能达到目标 precision 的 decision score 阈值。
    """
    precisions, thresholds = precision_recall_curve(y_true, scores)

    # precision_recall_curve 返回的 precisions 比 thresholds 多一个值；
    # 最后一个 precision 没有对应阈值，所以只看 precisions[:-1]。
    eligible = np.flatnonzero(precisions[:-1] >= target)

    if eligible.size == 0:
        return float("inf")

    return float(thresholds[eligible[0]])


def evaluate_at_threshold(y_true, scores, threshold):
    """按给定分数阈值分类，并返回分类指标和预测为正类的数量。"""
    y_pred = np.asarray(scores) >= threshold
    precision, recall, f1 = precision_recall_f1(y_true, y_pred)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "positives": int(np.sum(y_pred)),
    }


def roc_auc(y_true, scores):
    """返回 ROC-AUC 和达到至少 90% 召回率时的 FPR。"""
    auc = roc_auc_score(y_true, scores)
    fpr, tpr, _ = roc_curve(y_true, scores)

    indices = np.flatnonzero(tpr >= 0.90)
    fpr_at_recall_90 = float(fpr[indices[0]]) if indices.size else 1.0

    return {
        "auc": float(auc),
        "fpr_at_recall_90": fpr_at_recall_90,
    }


# ============================================================
# 10-13. 多分类 pipeline、交叉验证和混淆分析
# ============================================================

def multiclass_pipeline(random_state=42):
    """
    返回尚未训练的流水线：
    StandardScaler -> SGDClassifier。
    """
    return make_pipeline(
        StandardScaler(),
        SGDClassifier(random_state=random_state),
    )


def multiclass_cv_accuracy(model, X, y, cv=3):
    """返回交叉验证平均 accuracy。"""
    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="accuracy",
    )

    return float(np.mean(scores))


def normalized_confusion(y_true, y_pred):
    """返回固定为 10×10、按真实类别行归一化的混淆矩阵。"""
    return confusion_matrix(
        y_true,
        y_pred,
        labels=range(10),
        normalize="true",
    )


def most_confused_pairs(cm, k=3):
    """
    返回最大的 k 个非对角线混淆项：
    (真实类别, 预测类别, 混淆比例)。
    """
    cm = np.asarray(cm)
    candidates = []

    for true_class in range(cm.shape[0]):
        for predicted_class in range(cm.shape[1]):
            if true_class != predicted_class:
                candidates.append(
                    (
                        true_class,
                        predicted_class,
                        float(cm[true_class, predicted_class]),
                    )
                )

    # 比例从大到小；相同则按类别编号从小到大，保持确定性
    candidates.sort(key=lambda item: (-item[2], item[0], item[1]))

    return [
        (int(true_class), int(predicted_class), round(rate, 3))
        for true_class, predicted_class, rate in candidates[:max(0, k)]
    ]


# ============================================================
# 14-15. 多标签分类
# ============================================================

def multilabel_targets(y):
    """
    为每个数字生成两个布尔标签：
    第 0 列：数字 >= 7
    第 1 列：数字是奇数
    """
    y = np.asarray(y)

    return np.column_stack([
        y >= 7,
        y % 2 == 1,
    ]).astype(bool)


def multilabel_knn(X, Y, n_neighbors=5):
    """训练并返回 KNN 多标签分类器。"""
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X, Y)
    return clf


def multilabel_f1(Y_true, Y_pred):
    """计算各标签 F1 的宏平均。"""
    return float(
        f1_score(
            Y_true,
            Y_pred,
            average="macro",
            zero_division=0,
        )
    )


# ============================================================
# 16-18. 最终测试、保存/加载、预测原始图片
# ============================================================

def final_test_accuracy(model, X_test, y_test):
    """在已经拟合好的模型上计算测试准确率，不重新训练。"""
    predictions = model.predict(X_test)
    return float(accuracy_score(y_test, predictions))


def save_and_reload_classifier(model, path):
    """用 joblib 保存模型，再加载并返回加载后的模型。"""
    joblib.dump(model, path)
    return joblib.load(path)


def predict_digits(model, images):
    """
    输入原始图片 (n, 28, 28)，摊平后送入已拟合的模型。
    返回由 Python int 组成的列表。
    """
    images = np.asarray(images)
    X = images.reshape(len(images), 784).astype(np.float32)

    predictions = model.predict(X)
    return [int(label) for label in predictions]

# Step 5 - confusion_counts
import os
import tempfile
import urllib.request

import joblib
import numpy as np

from sklearn.base import clone
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. 加载 MNIST
# ============================================================

def load_mnist(n_train=10000, n_test=2000):
    """
    下载并缓存 MNIST，返回训练集和测试集的前若干行。

    图片保留 0~255 像素范围，摊平为 784 维。
    """
    url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
    path = os.path.join(tempfile.gettempdir(), "mnist.npz")

    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)

    with np.load(path) as data:
        X_train = data["x_train"][:n_train].reshape(-1, 784).astype(np.float32)
        y_train = data["y_train"][:n_train].astype(np.int64)

        X_test = data["x_test"][:n_test].reshape(-1, 784).astype(np.float32)
        y_test = data["y_test"][:n_test].astype(np.int64)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
    }


# ============================================================
# 2-4. “是不是 5”的二分类和交叉验证预测
# ============================================================

def binary_target(y, digit=5):
    """标签等于 digit 的位置为 True，其余为 False。"""
    return np.asarray(y) == digit


def train_sgd(X, y, random_state=42):
    """训练并返回 SGDClassifier。"""
    clf = SGDClassifier(random_state=random_state)
    clf.fit(X, y)
    return clf


def cross_val_predictions(clf, X, y, cv=3, method="predict"):
    """
    生成交叉验证的 out-of-fold 预测或分数。

    每个样本的预测都来自一个没有用该样本训练过的模型。
    """
    return cross_val_predict(
        clone(clf),
        X,
        y,
        cv=cv,
        method=method,
    )


# ============================================================
# 5-6. 二分类混淆矩阵和指标
# ============================================================

def confusion_counts(y_true, y_pred):
    """
    返回固定类别顺序 [False, True] 下的混淆矩阵计数：
    TN、FP、FN、TP。
    """
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[False, True],
    )

    tn, fp, fn, tp = matrix.ravel()

    return {
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def precision_recall_f1(y_true, y_pred):
    """返回 precision、recall、F1。"""
    precision = precision_score(
        y_true, y_pred, zero_division=0
    )
    recall = recall_score(
        y_true, y_pred, zero_division=0
    )
    f1 = f1_score(
        y_true, y_pred, zero_division=0
    )

    return float(precision), float(recall), float(f1)


# ============================================================
# 7-9. 调阈值、按阈值评估和 ROC-AUC
# ============================================================

def threshold_for_precision(y_true, scores, target=0.90):
    """
    找到第一个能达到目标 precision 的 decision score 阈值。
    """
    precisions, thresholds = precision_recall_curve(y_true, scores)

    # precision_recall_curve 返回的 precisions 比 thresholds 多一个值；
    # 最后一个 precision 没有对应阈值，所以只看 precisions[:-1]。
    eligible = np.flatnonzero(precisions[:-1] >= target)

    if eligible.size == 0:
        return float("inf")

    return float(thresholds[eligible[0]])


def evaluate_at_threshold(y_true, scores, threshold):
    """按给定分数阈值分类，并返回分类指标和预测为正类的数量。"""
    y_pred = np.asarray(scores) >= threshold
    precision, recall, f1 = precision_recall_f1(y_true, y_pred)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "positives": int(np.sum(y_pred)),
    }


def roc_auc(y_true, scores):
    """返回 ROC-AUC 和达到至少 90% 召回率时的 FPR。"""
    auc = roc_auc_score(y_true, scores)
    fpr, tpr, _ = roc_curve(y_true, scores)

    indices = np.flatnonzero(tpr >= 0.90)
    fpr_at_recall_90 = float(fpr[indices[0]]) if indices.size else 1.0

    return {
        "auc": float(auc),
        "fpr_at_recall_90": fpr_at_recall_90,
    }


# ============================================================
# 10-13. 多分类 pipeline、交叉验证和混淆分析
# ============================================================

def multiclass_pipeline(random_state=42):
    """
    返回尚未训练的流水线：
    StandardScaler -> SGDClassifier。
    """
    return make_pipeline(
        StandardScaler(),
        SGDClassifier(random_state=random_state),
    )


def multiclass_cv_accuracy(model, X, y, cv=3):
    """返回交叉验证平均 accuracy。"""
    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="accuracy",
    )

    return float(np.mean(scores))


def normalized_confusion(y_true, y_pred):
    """返回固定为 10×10、按真实类别行归一化的混淆矩阵。"""
    return confusion_matrix(
        y_true,
        y_pred,
        labels=range(10),
        normalize="true",
    )


def most_confused_pairs(cm, k=3):
    """
    返回最大的 k 个非对角线混淆项：
    (真实类别, 预测类别, 混淆比例)。
    """
    cm = np.asarray(cm)
    candidates = []

    for true_class in range(cm.shape[0]):
        for predicted_class in range(cm.shape[1]):
            if true_class != predicted_class:
                candidates.append(
                    (
                        true_class,
                        predicted_class,
                        float(cm[true_class, predicted_class]),
                    )
                )

    # 比例从大到小；相同则按类别编号从小到大，保持确定性
    candidates.sort(key=lambda item: (-item[2], item[0], item[1]))

    return [
        (int(true_class), int(predicted_class), round(rate, 3))
        for true_class, predicted_class, rate in candidates[:max(0, k)]
    ]


# ============================================================
# 14-15. 多标签分类
# ============================================================

def multilabel_targets(y):
    """
    为每个数字生成两个布尔标签：
    第 0 列：数字 >= 7
    第 1 列：数字是奇数
    """
    y = np.asarray(y)

    return np.column_stack([
        y >= 7,
        y % 2 == 1,
    ]).astype(bool)


def multilabel_knn(X, Y, n_neighbors=5):
    """训练并返回 KNN 多标签分类器。"""
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X, Y)
    return clf


def multilabel_f1(Y_true, Y_pred):
    """计算各标签 F1 的宏平均。"""
    return float(
        f1_score(
            Y_true,
            Y_pred,
            average="macro",
            zero_division=0,
        )
    )


# ============================================================
# 16-18. 最终测试、保存/加载、预测原始图片
# ============================================================

def final_test_accuracy(model, X_test, y_test):
    """在已经拟合好的模型上计算测试准确率，不重新训练。"""
    predictions = model.predict(X_test)
    return float(accuracy_score(y_test, predictions))


def save_and_reload_classifier(model, path):
    """用 joblib 保存模型，再加载并返回加载后的模型。"""
    joblib.dump(model, path)
    return joblib.load(path)


def predict_digits(model, images):
    """
    输入原始图片 (n, 28, 28)，摊平后送入已拟合的模型。
    返回由 Python int 组成的列表。
    """
    images = np.asarray(images)
    X = images.reshape(len(images), 784).astype(np.float32)

    predictions = model.predict(X)
    return [int(label) for label in predictions]

# Step 6 - precision_recall_f1
import os
import tempfile
import urllib.request

import joblib
import numpy as np

from sklearn.base import clone
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. 加载 MNIST
# ============================================================

def load_mnist(n_train=10000, n_test=2000):
    """
    下载并缓存 MNIST，返回训练集和测试集的前若干行。

    图片保留 0~255 像素范围，摊平为 784 维。
    """
    url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
    path = os.path.join(tempfile.gettempdir(), "mnist.npz")

    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)

    with np.load(path) as data:
        X_train = data["x_train"][:n_train].reshape(-1, 784).astype(np.float32)
        y_train = data["y_train"][:n_train].astype(np.int64)

        X_test = data["x_test"][:n_test].reshape(-1, 784).astype(np.float32)
        y_test = data["y_test"][:n_test].astype(np.int64)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
    }


# ============================================================
# 2-4. “是不是 5”的二分类和交叉验证预测
# ============================================================

def binary_target(y, digit=5):
    """标签等于 digit 的位置为 True，其余为 False。"""
    return np.asarray(y) == digit


def train_sgd(X, y, random_state=42):
    """训练并返回 SGDClassifier。"""
    clf = SGDClassifier(random_state=random_state)
    clf.fit(X, y)
    return clf


def cross_val_predictions(clf, X, y, cv=3, method="predict"):
    """
    生成交叉验证的 out-of-fold 预测或分数。

    每个样本的预测都来自一个没有用该样本训练过的模型。
    """
    return cross_val_predict(
        clone(clf),
        X,
        y,
        cv=cv,
        method=method,
    )


# ============================================================
# 5-6. 二分类混淆矩阵和指标
# ============================================================

def confusion_counts(y_true, y_pred):
    """
    返回固定类别顺序 [False, True] 下的混淆矩阵计数：
    TN、FP、FN、TP。
    """
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[False, True],
    )

    tn, fp, fn, tp = matrix.ravel()

    return {
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def precision_recall_f1(y_true, y_pred):
    """返回 precision、recall、F1。"""
    precision = precision_score(
        y_true, y_pred, zero_division=0
    )
    recall = recall_score(
        y_true, y_pred, zero_division=0
    )
    f1 = f1_score(
        y_true, y_pred, zero_division=0
    )

    return float(precision), float(recall), float(f1)


# ============================================================
# 7-9. 调阈值、按阈值评估和 ROC-AUC
# ============================================================

def threshold_for_precision(y_true, scores, target=0.90):
    """
    找到第一个能达到目标 precision 的 decision score 阈值。
    """
    precisions, thresholds = precision_recall_curve(y_true, scores)

    # precision_recall_curve 返回的 precisions 比 thresholds 多一个值；
    # 最后一个 precision 没有对应阈值，所以只看 precisions[:-1]。
    eligible = np.flatnonzero(precisions[:-1] >= target)

    if eligible.size == 0:
        return float("inf")

    return float(thresholds[eligible[0]])


def evaluate_at_threshold(y_true, scores, threshold):
    """按给定分数阈值分类，并返回分类指标和预测为正类的数量。"""
    y_pred = np.asarray(scores) >= threshold
    precision, recall, f1 = precision_recall_f1(y_true, y_pred)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "positives": int(np.sum(y_pred)),
    }


def roc_auc(y_true, scores):
    """返回 ROC-AUC 和达到至少 90% 召回率时的 FPR。"""
    auc = roc_auc_score(y_true, scores)
    fpr, tpr, _ = roc_curve(y_true, scores)

    indices = np.flatnonzero(tpr >= 0.90)
    fpr_at_recall_90 = float(fpr[indices[0]]) if indices.size else 1.0

    return {
        "auc": float(auc),
        "fpr_at_recall_90": fpr_at_recall_90,
    }


# ============================================================
# 10-13. 多分类 pipeline、交叉验证和混淆分析
# ============================================================

def multiclass_pipeline(random_state=42):
    """
    返回尚未训练的流水线：
    StandardScaler -> SGDClassifier。
    """
    return make_pipeline(
        StandardScaler(),
        SGDClassifier(random_state=random_state),
    )


def multiclass_cv_accuracy(model, X, y, cv=3):
    """返回交叉验证平均 accuracy。"""
    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="accuracy",
    )

    return float(np.mean(scores))


def normalized_confusion(y_true, y_pred):
    """返回固定为 10×10、按真实类别行归一化的混淆矩阵。"""
    return confusion_matrix(
        y_true,
        y_pred,
        labels=range(10),
        normalize="true",
    )


def most_confused_pairs(cm, k=3):
    """
    返回最大的 k 个非对角线混淆项：
    (真实类别, 预测类别, 混淆比例)。
    """
    cm = np.asarray(cm)
    candidates = []

    for true_class in range(cm.shape[0]):
        for predicted_class in range(cm.shape[1]):
            if true_class != predicted_class:
                candidates.append(
                    (
                        true_class,
                        predicted_class,
                        float(cm[true_class, predicted_class]),
                    )
                )

    # 比例从大到小；相同则按类别编号从小到大，保持确定性
    candidates.sort(key=lambda item: (-item[2], item[0], item[1]))

    return [
        (int(true_class), int(predicted_class), round(rate, 3))
        for true_class, predicted_class, rate in candidates[:max(0, k)]
    ]


# ============================================================
# 14-15. 多标签分类
# ============================================================

def multilabel_targets(y):
    """
    为每个数字生成两个布尔标签：
    第 0 列：数字 >= 7
    第 1 列：数字是奇数
    """
    y = np.asarray(y)

    return np.column_stack([
        y >= 7,
        y % 2 == 1,
    ]).astype(bool)


def multilabel_knn(X, Y, n_neighbors=5):
    """训练并返回 KNN 多标签分类器。"""
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X, Y)
    return clf


def multilabel_f1(Y_true, Y_pred):
    """计算各标签 F1 的宏平均。"""
    return float(
        f1_score(
            Y_true,
            Y_pred,
            average="macro",
            zero_division=0,
        )
    )


# ============================================================
# 16-18. 最终测试、保存/加载、预测原始图片
# ============================================================

def final_test_accuracy(model, X_test, y_test):
    """在已经拟合好的模型上计算测试准确率，不重新训练。"""
    predictions = model.predict(X_test)
    return float(accuracy_score(y_test, predictions))


def save_and_reload_classifier(model, path):
    """用 joblib 保存模型，再加载并返回加载后的模型。"""
    joblib.dump(model, path)
    return joblib.load(path)


def predict_digits(model, images):
    """
    输入原始图片 (n, 28, 28)，摊平后送入已拟合的模型。
    返回由 Python int 组成的列表。
    """
    images = np.asarray(images)
    X = images.reshape(len(images), 784).astype(np.float32)

    predictions = model.predict(X)
    return [int(label) for label in predictions]

# Step 7 - threshold_for_precision
import os
import tempfile
import urllib.request

import joblib
import numpy as np

from sklearn.base import clone
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. 加载 MNIST
# ============================================================

def load_mnist(n_train=10000, n_test=2000):
    """
    下载并缓存 MNIST，返回训练集和测试集的前若干行。

    图片保留 0~255 像素范围，摊平为 784 维。
    """
    url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
    path = os.path.join(tempfile.gettempdir(), "mnist.npz")

    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)

    with np.load(path) as data:
        X_train = data["x_train"][:n_train].reshape(-1, 784).astype(np.float32)
        y_train = data["y_train"][:n_train].astype(np.int64)

        X_test = data["x_test"][:n_test].reshape(-1, 784).astype(np.float32)
        y_test = data["y_test"][:n_test].astype(np.int64)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
    }


# ============================================================
# 2-4. “是不是 5”的二分类和交叉验证预测
# ============================================================

def binary_target(y, digit=5):
    """标签等于 digit 的位置为 True，其余为 False。"""
    return np.asarray(y) == digit


def train_sgd(X, y, random_state=42):
    """训练并返回 SGDClassifier。"""
    clf = SGDClassifier(random_state=random_state)
    clf.fit(X, y)
    return clf


def cross_val_predictions(clf, X, y, cv=3, method="predict"):
    """
    生成交叉验证的 out-of-fold 预测或分数。

    每个样本的预测都来自一个没有用该样本训练过的模型。
    """
    return cross_val_predict(
        clone(clf),
        X,
        y,
        cv=cv,
        method=method,
    )


# ============================================================
# 5-6. 二分类混淆矩阵和指标
# ============================================================

def confusion_counts(y_true, y_pred):
    """
    返回固定类别顺序 [False, True] 下的混淆矩阵计数：
    TN、FP、FN、TP。
    """
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[False, True],
    )

    tn, fp, fn, tp = matrix.ravel()

    return {
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def precision_recall_f1(y_true, y_pred):
    """返回 precision、recall、F1。"""
    precision = precision_score(
        y_true, y_pred, zero_division=0
    )
    recall = recall_score(
        y_true, y_pred, zero_division=0
    )
    f1 = f1_score(
        y_true, y_pred, zero_division=0
    )

    return float(precision), float(recall), float(f1)


# ============================================================
# 7-9. 调阈值、按阈值评估和 ROC-AUC
# ============================================================

def threshold_for_precision(y_true, scores, target=0.90):
    """
    找到第一个能达到目标 precision 的 decision score 阈值。
    """
    precisions, thresholds = precision_recall_curve(y_true, scores)

    # precision_recall_curve 返回的 precisions 比 thresholds 多一个值；
    # 最后一个 precision 没有对应阈值，所以只看 precisions[:-1]。
    eligible = np.flatnonzero(precisions[:-1] >= target)

    if eligible.size == 0:
        return float("inf")

    return float(thresholds[eligible[0]])


def evaluate_at_threshold(y_true, scores, threshold):
    """按给定分数阈值分类，并返回分类指标和预测为正类的数量。"""
    y_pred = np.asarray(scores) >= threshold
    precision, recall, f1 = precision_recall_f1(y_true, y_pred)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "positives": int(np.sum(y_pred)),
    }


def roc_auc(y_true, scores):
    """返回 ROC-AUC 和达到至少 90% 召回率时的 FPR。"""
    auc = roc_auc_score(y_true, scores)
    fpr, tpr, _ = roc_curve(y_true, scores)

    indices = np.flatnonzero(tpr >= 0.90)
    fpr_at_recall_90 = float(fpr[indices[0]]) if indices.size else 1.0

    return {
        "auc": float(auc),
        "fpr_at_recall_90": fpr_at_recall_90,
    }


# ============================================================
# 10-13. 多分类 pipeline、交叉验证和混淆分析
# ============================================================

def multiclass_pipeline(random_state=42):
    """
    返回尚未训练的流水线：
    StandardScaler -> SGDClassifier。
    """
    return make_pipeline(
        StandardScaler(),
        SGDClassifier(random_state=random_state),
    )


def multiclass_cv_accuracy(model, X, y, cv=3):
    """返回交叉验证平均 accuracy。"""
    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="accuracy",
    )

    return float(np.mean(scores))


def normalized_confusion(y_true, y_pred):
    """返回固定为 10×10、按真实类别行归一化的混淆矩阵。"""
    return confusion_matrix(
        y_true,
        y_pred,
        labels=range(10),
        normalize="true",
    )


def most_confused_pairs(cm, k=3):
    """
    返回最大的 k 个非对角线混淆项：
    (真实类别, 预测类别, 混淆比例)。
    """
    cm = np.asarray(cm)
    candidates = []

    for true_class in range(cm.shape[0]):
        for predicted_class in range(cm.shape[1]):
            if true_class != predicted_class:
                candidates.append(
                    (
                        true_class,
                        predicted_class,
                        float(cm[true_class, predicted_class]),
                    )
                )

    # 比例从大到小；相同则按类别编号从小到大，保持确定性
    candidates.sort(key=lambda item: (-item[2], item[0], item[1]))

    return [
        (int(true_class), int(predicted_class), round(rate, 3))
        for true_class, predicted_class, rate in candidates[:max(0, k)]
    ]


# ============================================================
# 14-15. 多标签分类
# ============================================================

def multilabel_targets(y):
    """
    为每个数字生成两个布尔标签：
    第 0 列：数字 >= 7
    第 1 列：数字是奇数
    """
    y = np.asarray(y)

    return np.column_stack([
        y >= 7,
        y % 2 == 1,
    ]).astype(bool)


def multilabel_knn(X, Y, n_neighbors=5):
    """训练并返回 KNN 多标签分类器。"""
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X, Y)
    return clf


def multilabel_f1(Y_true, Y_pred):
    """计算各标签 F1 的宏平均。"""
    return float(
        f1_score(
            Y_true,
            Y_pred,
            average="macro",
            zero_division=0,
        )
    )


# ============================================================
# 16-18. 最终测试、保存/加载、预测原始图片
# ============================================================

def final_test_accuracy(model, X_test, y_test):
    """在已经拟合好的模型上计算测试准确率，不重新训练。"""
    predictions = model.predict(X_test)
    return float(accuracy_score(y_test, predictions))


def save_and_reload_classifier(model, path):
    """用 joblib 保存模型，再加载并返回加载后的模型。"""
    joblib.dump(model, path)
    return joblib.load(path)


def predict_digits(model, images):
    """
    输入原始图片 (n, 28, 28)，摊平后送入已拟合的模型。
    返回由 Python int 组成的列表。
    """
    images = np.asarray(images)
    X = images.reshape(len(images), 784).astype(np.float32)

    predictions = model.predict(X)
    return [int(label) for label in predictions]

# Step 8 - evaluate_at_threshold
import os
import tempfile
import urllib.request

import joblib
import numpy as np

from sklearn.base import clone
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. 加载 MNIST
# ============================================================

def load_mnist(n_train=10000, n_test=2000):
    """
    下载并缓存 MNIST，返回训练集和测试集的前若干行。

    图片保留 0~255 像素范围，摊平为 784 维。
    """
    url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
    path = os.path.join(tempfile.gettempdir(), "mnist.npz")

    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)

    with np.load(path) as data:
        X_train = data["x_train"][:n_train].reshape(-1, 784).astype(np.float32)
        y_train = data["y_train"][:n_train].astype(np.int64)

        X_test = data["x_test"][:n_test].reshape(-1, 784).astype(np.float32)
        y_test = data["y_test"][:n_test].astype(np.int64)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
    }


# ============================================================
# 2-4. “是不是 5”的二分类和交叉验证预测
# ============================================================

def binary_target(y, digit=5):
    """标签等于 digit 的位置为 True，其余为 False。"""
    return np.asarray(y) == digit


def train_sgd(X, y, random_state=42):
    """训练并返回 SGDClassifier。"""
    clf = SGDClassifier(random_state=random_state)
    clf.fit(X, y)
    return clf


def cross_val_predictions(clf, X, y, cv=3, method="predict"):
    """
    生成交叉验证的 out-of-fold 预测或分数。

    每个样本的预测都来自一个没有用该样本训练过的模型。
    """
    return cross_val_predict(
        clone(clf),
        X,
        y,
        cv=cv,
        method=method,
    )


# ============================================================
# 5-6. 二分类混淆矩阵和指标
# ============================================================

def confusion_counts(y_true, y_pred):
    """
    返回固定类别顺序 [False, True] 下的混淆矩阵计数：
    TN、FP、FN、TP。
    """
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[False, True],
    )

    tn, fp, fn, tp = matrix.ravel()

    return {
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def precision_recall_f1(y_true, y_pred):
    """返回 precision、recall、F1。"""
    precision = precision_score(
        y_true, y_pred, zero_division=0
    )
    recall = recall_score(
        y_true, y_pred, zero_division=0
    )
    f1 = f1_score(
        y_true, y_pred, zero_division=0
    )

    return float(precision), float(recall), float(f1)


# ============================================================
# 7-9. 调阈值、按阈值评估和 ROC-AUC
# ============================================================

def threshold_for_precision(y_true, scores, target=0.90):
    """
    找到第一个能达到目标 precision 的 decision score 阈值。
    """
    precisions, thresholds = precision_recall_curve(y_true, scores)

    # precision_recall_curve 返回的 precisions 比 thresholds 多一个值；
    # 最后一个 precision 没有对应阈值，所以只看 precisions[:-1]。
    eligible = np.flatnonzero(precisions[:-1] >= target)

    if eligible.size == 0:
        return float("inf")

    return float(thresholds[eligible[0]])


def evaluate_at_threshold(y_true, scores, threshold):
    """按给定分数阈值分类，并返回分类指标和预测为正类的数量。"""
    y_pred = np.asarray(scores) >= threshold
    precision, recall, f1 = precision_recall_f1(y_true, y_pred)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "positives": int(np.sum(y_pred)),
    }


def roc_auc(y_true, scores):
    """返回 ROC-AUC 和达到至少 90% 召回率时的 FPR。"""
    auc = roc_auc_score(y_true, scores)
    fpr, tpr, _ = roc_curve(y_true, scores)

    indices = np.flatnonzero(tpr >= 0.90)
    fpr_at_recall_90 = float(fpr[indices[0]]) if indices.size else 1.0

    return {
        "auc": float(auc),
        "fpr_at_recall_90": fpr_at_recall_90,
    }


# ============================================================
# 10-13. 多分类 pipeline、交叉验证和混淆分析
# ============================================================

def multiclass_pipeline(random_state=42):
    """
    返回尚未训练的流水线：
    StandardScaler -> SGDClassifier。
    """
    return make_pipeline(
        StandardScaler(),
        SGDClassifier(random_state=random_state),
    )


def multiclass_cv_accuracy(model, X, y, cv=3):
    """返回交叉验证平均 accuracy。"""
    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="accuracy",
    )

    return float(np.mean(scores))


def normalized_confusion(y_true, y_pred):
    """返回固定为 10×10、按真实类别行归一化的混淆矩阵。"""
    return confusion_matrix(
        y_true,
        y_pred,
        labels=range(10),
        normalize="true",
    )


def most_confused_pairs(cm, k=3):
    """
    返回最大的 k 个非对角线混淆项：
    (真实类别, 预测类别, 混淆比例)。
    """
    cm = np.asarray(cm)
    candidates = []

    for true_class in range(cm.shape[0]):
        for predicted_class in range(cm.shape[1]):
            if true_class != predicted_class:
                candidates.append(
                    (
                        true_class,
                        predicted_class,
                        float(cm[true_class, predicted_class]),
                    )
                )

    # 比例从大到小；相同则按类别编号从小到大，保持确定性
    candidates.sort(key=lambda item: (-item[2], item[0], item[1]))

    return [
        (int(true_class), int(predicted_class), round(rate, 3))
        for true_class, predicted_class, rate in candidates[:max(0, k)]
    ]


# ============================================================
# 14-15. 多标签分类
# ============================================================

def multilabel_targets(y):
    """
    为每个数字生成两个布尔标签：
    第 0 列：数字 >= 7
    第 1 列：数字是奇数
    """
    y = np.asarray(y)

    return np.column_stack([
        y >= 7,
        y % 2 == 1,
    ]).astype(bool)


def multilabel_knn(X, Y, n_neighbors=5):
    """训练并返回 KNN 多标签分类器。"""
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X, Y)
    return clf


def multilabel_f1(Y_true, Y_pred):
    """计算各标签 F1 的宏平均。"""
    return float(
        f1_score(
            Y_true,
            Y_pred,
            average="macro",
            zero_division=0,
        )
    )


# ============================================================
# 16-18. 最终测试、保存/加载、预测原始图片
# ============================================================

def final_test_accuracy(model, X_test, y_test):
    """在已经拟合好的模型上计算测试准确率，不重新训练。"""
    predictions = model.predict(X_test)
    return float(accuracy_score(y_test, predictions))


def save_and_reload_classifier(model, path):
    """用 joblib 保存模型，再加载并返回加载后的模型。"""
    joblib.dump(model, path)
    return joblib.load(path)


def predict_digits(model, images):
    """
    输入原始图片 (n, 28, 28)，摊平后送入已拟合的模型。
    返回由 Python int 组成的列表。
    """
    images = np.asarray(images)
    X = images.reshape(len(images), 784).astype(np.float32)

    predictions = model.predict(X)
    return [int(label) for label in predictions]

# Step 9 - roc_auc (not yet solved)
# TODO: implement

# Step 10 - multiclass_pipeline (not yet solved)
# TODO: implement

# Step 11 - multiclass_cv_accuracy (not yet solved)
# TODO: implement

# Step 12 - normalized_confusion (not yet solved)
# TODO: implement

# Step 13 - most_confused_pairs (not yet solved)
# TODO: implement

# Step 14 - multilabel_targets (not yet solved)
# TODO: implement

# Step 15 - multilabel_knn (not yet solved)
# TODO: implement

# Step 16 - final_test_accuracy (not yet solved)
# TODO: implement

# Step 17 - save_and_reload_classifier (not yet solved)
# TODO: implement

# Step 18 - predict_digits (not yet solved)
# TODO: implement

