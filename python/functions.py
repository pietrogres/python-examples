"""
Common functions script
"""
import re
from typing import Dict, List

import unidecode
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import scikitplot as skplt
from matplotlib import pyplot as plt
from sklearn.metrics import (
    mean_absolute_error,
    median_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)
from sklearn.model_selection import cross_val_score, KFold
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from IPython.display import display

distribution_plots = ['hist', 'violin', 'box']
num_types = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
regression_metrics = ['mae', 'mdae', 'rmae', 'mse', 'rmse', 'r2']
classification_metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']


# GENERIC PYTHON FUNCTIONS

def clean_whitespaces(s: str) -> str:
    """
    Clean misplaced whitespaces in a string.
    Multiple subsequent whitespaces are replaced by a single whitespace,
    whitespaces at the beginning or at the end of the string are removed.
    """

    return re.sub(r'\s+', ' ', s.strip())


# DATAFRAME FUNCTIONS

def select_num_cols(df: pd.DataFrame, types: list = num_types):
    """Extract list of numerical features from DataFrame"""
    return df.select_dtypes(include=types).columns.tolist()


def check_inf_cols(df: pd.DataFrame) -> List[str]:
    """Checks whether DataFrame has columns with infinite values"""
    inf_cols = [c for c, v in dict(np.isinf(df).sum()).items() if v > 0]
    if inf_cols:
        print(f'WARNING! Found {len(inf_cols)} columns containing infinite values')
    return inf_cols


def check_null_cols(df: pd.DataFrame) -> List[str]:
    """Checks whether DataFrame has columns with NaN or empty string values"""
    # np_nan_cols = [c for c,v in dict((df==np.nan).sum()).items() if v>0]
    nan_cols = [c for c, v in dict(df.isna().sum()).items() if v > 0]
    empty_cols = [c for c, v in dict((df == '').sum()).items() if v > 0]
    # if np_nan_cols: print(f'WARNING! Found {len(np_nan_cols)} columns containing NumPy NaN values')
    if nan_cols:
        print(f'WARNING! Found {len(nan_cols)} columns containing NaN values')
    if empty_cols:
        print(f'WARNING! Found {len(empty_cols)} columns containing empty string values')
    return nan_cols, empty_cols


def check_outliers(df: pd.DataFrame) -> List[str]:
    """Checks whether DataFrame has columns with outliers with IQR method"""
    num_cols = select_num_cols(df)
    out_cols = []

    for col, vals in df[num_cols].items():
        q1 = vals.quantile(0.25)
        q3 = vals.quantile(0.75)
        irq = q3 - q1
        out_vals = vals[(vals <= q1 - 1.5 * irq) | (vals >= q3 + 1.5 * irq)]
        if len(out_vals) > 0:
            out_cols.append(col)
            print(f'Found {len(out_vals)} outliers for column {col} ({(len(out_vals) * 100 / len(vals)):.2f}%)')
    return out_cols


def plot_histrogram(df: pd.DataFrame, cols: List[str] = None) -> None:
    """
    Plot histogram of specified features from pandas DataFrame.
    If no feature list is provided then all numeric features are used.
    """
    cols = cols if cols else select_num_cols(df)

    for col in cols:
        fig, axs = plt.subplots(nrows=1, figsize=(30, 5))
        sns.histplot(df[col], kde=True, ax=axs)
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
        axs.set_title(col, fontsize=18)


def plot_box(df: pd.DataFrame, cols: list = None) -> None:
    """
    Plot boxplot of specified features from pandas DataFrame.
    If no feature list is provided then all numeric features are used.
    """
    cols = cols if cols else select_num_cols(df)

    for col in cols:
        fig, axs = plt.subplots(nrows=1, figsize=(30, 2))
        sns.boxplot(data=df, x=col, ax=axs)
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
        axs.set_title(col, fontsize=18)


def plot_violin(df: pd.DataFrame, cols: List[str] = None) -> None:
    """
    Plot violinplot of specified features from pandas DataFrame.
    If no feature list is provided then all numeric features are used.
    """
    cols = cols if cols else select_num_cols(df)

    for col in cols:
        fig, axs = plt.subplots(nrows=1, figsize=(30, 2))
        sns.violinplot(data=df, x=col, ax=axs)
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
        axs.set_title(col, fontsize=18)


def plot_distribution(df: pd.DataFrame, cols: List[str] = None, plots: List[str] = distribution_plots) -> None:
    """
    Plot distributions (histogram, violin, box) of specified features from pandas DataFrame.
    If no feature list is provided then all numeric features are used.
    """
    valid_plots = ['hist', 'violin', 'box']
    if not all([p.lower() in valid_plots for p in plots]):
        raise ValueError(f'INPUT ERROR! Input {[p.lower() for p in plots if p.lower() not in valid_plots]}'
                         f' are not supported, please choose among {valid_plots}')

    plots_specs = {
        'hist': {'height': 6, 'ratio': .7},
        'violin': {'height': 2, 'ratio': .17},
        'box': {'height': 1, 'ratio': .13},
    }

    ratios = [plots_specs[plot]['ratio'] / 10 for plot in plots]
    ratios = tuple([ratio / sum(ratios) for ratio in ratios])

    height = sum([plots_specs[plot]['height'] for plot in plots])

    cols = cols if cols else select_num_cols(df)
    for col in cols:
        fig, axs = plt.subplots(nrows=len(plots), sharex=True, gridspec_kw={'height_ratios': ratios},
                                figsize=(30, height))
        index = 0
        if 'hist' in plots:
            sns.histplot(df[col], kde=True, ax=axs[index])
            axs[index].set_xlabel('')
            index = index + 1 if index < len(plots) - 1 else index
        if 'violin' in plots:
            sns.violinplot(data=df, x=col, ax=axs[index])
            axs[index].set_xlabel('')
            index = index + 1 if index < len(plots) - 1 else index
        if 'box' in plots:
            sns.boxplot(data=df, x=col, ax=axs[index])
            axs[index].set_xlabel('')

        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
        axs[0].set_title(col, fontsize=18)
        axs[index].tick_params(axis='x', labelsize=13)


def convert_pd_dataframe_to_np_array(df: pd.DataFrame) -> np.array:
    """Convert pandas DataFrame to numpy array"""
    return df.to_numpy()


def normalize_string_columns(df: pd.DataFrame, cols: List[str] = None) -> pd.DataFrame:
    """Normalize list of object columns in pandas DataFrame"""
    cols = cols if cols else df.select_dtypes(['object']).columns.tolist()
    for col in cols:
        df[col] = df[col].apply(
            lambda x: clean_whitespaces(re.sub(r'[^a-zA-Z\d]+', ' ', unidecode.unidecode(x)).strip().upper())
        )
    return df


def fix_columns(df: pd.DataFrame, suffix: str = None) -> pd.DataFrame:
    """
    Fix pandas DataFrame columns.
    Column names are converted to lowercase and every non-alphanumeric character is converted to underscore.
    If provided, a suffix is attacched to column names.
    Categorical columns are cleaned, stripped and converted to upper-case.
    """
    df_out = df.copy()
    df_out.columns = [re.sub(r'([^a-z\d])', '_', col.lower()) for col in df_out.columns]
    if suffix:
        df_out.columns = [col + '_' + suffix for col in df_out.columns]
    df_out.columns = [re.sub(r'_+', '_', col) for col in df_out.columns]

    obj_cols = df_out.select_dtypes(['object']).columns.tolist()
    df_out = normalize_string_columns(df_out, obj_cols)
    return df_out


def describe_cat_col(df: pd.DataFrame, split_col: str, target_col: str, box: bool = True) -> None:
    """Describe given categorical column and plot side by side boxplot of target column"""

    def plot_describe_box() -> None:
        """Plot side by side boxplot of target column split by other column levels"""
        assert split_col in df.columns.tolist(), 'ERROR: specified variable is not a column'
        levels = [lvl for lvl in list(df[split_col].unique()) if not pd.isnull(lvl)]
        levels.sort()
        plots = [df[df[split_col] == lvl][target_col] for lvl in levels]
        if df.isna().sum()[split_col] > 0:
            levels.append('NaN')
            plots.append(df[df[split_col].isna()][target_col])

        plt.figure(figsize=(30, 3 * len(levels)))
        for i in range(1, len(levels) + 1):
            ax = plt.subplot(1, len(levels), i)
            ax.boxplot(plots[i], labels=levels, vert=False)

    assert split_col in df.columns.tolist(), 'ERROR: specified variable is not a column'

    display(df.groupby(split_col).size().to_frame('nr_obs').reset_index().sort_values('nr_obs', ascending=False))
    if df.isna().sum()[split_col] > 0:
        print(f'Found {len(df[df[split_col].isna()])} observations with NaN value in {split_col} column')
        display(df[df[split_col].isna()].head())

    if box:
        plot_describe_box()


def revert_scaling(df: pd.DataFrame, scal_obj) -> pd.DataFrame:
    """Revert sklearn scaling"""
    reverted_feat = scal_obj.inverse_transform(df)
    return pd.DataFrame(reverted_feat, columns=df.columns.tolist(), index=df.index)


# SCIKIT LEARN MODELS FUNCTIONS

def regression_performances(y: pd.Series, y_pred: pd.Series, metrics: List[str] = regression_metrics,
                            verbose: bool = True) -> Dict[str, float]:
    """Compute regression performances comparing true with predicted values against given metrics"""
    perf = {
        'mae': mean_absolute_error(y, y_pred),
        'mdae': median_absolute_error(y, y_pred),
        'rmae': np.sqrt(mean_absolute_error(y, y_pred)),
        'mse': mean_squared_error(y, y_pred),
        'rmse': mean_squared_error(y, y_pred, squared=False),
        'r2': r2_score(y, y_pred)
    }

    if verbose:
        if 'mae' in metrics:
            print('\tMean Absolute Error (MAE):       ', round(perf['mae'], 2))
        if 'mdae' in metrics:
            print('\tMedian Absolute Error (MdAE):    ', round(perf['mdae'], 2))
        if 'rmae' in metrics:
            print('\tRoot Mean Absolute Error (RMAE): ', round(perf['rmse'], 2))
        if 'mse' in metrics:
            print('\tMean Squared Error (MSE):        ', round(perf['mse'], 2))
        if 'rmse' in metrics:
            print('\tRoot Mean Squared Error (RMSE):  ', round(perf['rmse'], 2))
        if 'r2' in metrics:
            print('\tR2:  ', round(perf['r2'], 2))
    return {k: v for k, v in perf.items() if k in metrics}


def classification_performances(y: pd.Series, y_pred: pd.Series, metrics: List[str] = classification_metrics,
                                multi: bool = False, verbose: bool = True) -> Dict[str, float]:
    """Compute classification performances comparing true with predicted values against given metrics"""
    avg = 'weighted' if multi else 'binary'
    perf = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, average=avg),
        'recall': recall_score(y, y_pred, average=avg),
        'f1': f1_score(y, y_pred, average=avg)
    }
    if multi:
        perf['auc'] = roc_auc_score(y, y_pred, average=avg)
    else:
        perf['auc'] = roc_auc_score(y, y_pred)

    if verbose:
        if 'accuracy' in metrics:
            print('\t- Accuracy:     ', round(perf['accuracy'], 2))
        if 'precision' in metrics:
            print('\t- Precision:    ', round(perf['precision'], 2))
        if 'recall' in metrics:
            print('\t- Recall:       ', round(perf['recall'], 2))
        if 'f1' in metrics:
            print('\t- F1 Score:     ', round(perf['f1'], 2))
        if 'auc' in metrics:
            print('\t- ROC AUC:      ', round(perf['auc'], 2))
    return {k: v for k, v in perf.items() if k in metrics}


def plot_roc_curve(y_true: pd.Series, y_proba: pd.Series, label: str = 'ROC Curve', ax: matplotlib.axes = None) -> None:
    """Plot ROC curve for binary classificator"""
    if not ax:
        fig, ax = plt.subplots(figsize=(8, 8))

    fpr, tpr, auc_thresholds = roc_curve(y_true, y_proba)

    ax.plot(fpr, tpr, linewidth=2, label=label)
    ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    ax.axis([-0.005, 1, 0, 1.005])
    ax.set_title('ROC Curve')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate (Recall)')
    ax.legend(loc='best')


def plot_precision_recall_vs_threshold(y_true: pd.Series, y_proba: pd.Series, ax: matplotlib.axes = None) -> None:
    """Plot regression recall scores for different prediction thresholds"""
    if not ax:
        fig, ax = plt.subplots(figsize=(8, 8))

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

    ax.plot(thresholds, precisions[:-1], 'b--', label='Precision')
    ax.plot(thresholds, recalls[:-1], 'g-', label='Recall')
    ax.set_title('Precision and Recall Scores as a function of the decision threshold')
    ax.set_ylabel('Score')
    ax.set_xlabel('Decision Threshold')
    ax.legend(loc='best')


def plot_classification_performances(y_true: pd.Series, y_proba: pd.Series, figsize: tuple = (35, 8)) -> None:
    """Plot classification performance"""
    fig, axs = plt.subplots(ncols=4, figsize=figsize)
    plot_precision_recall_vs_threshold(y_true, y_proba, ax=axs[0])
    plot_roc_curve(y_true, y_proba, ax=axs[1])
    skplt.metrics.plot_cumulative_gain(y_true, y_proba, ax=axs[2], title_fontsize=15, text_fontsize=10)
    skplt.metrics.plot_lift_curve(y_true, y_proba, ax=axs[3], title_fontsize=15, text_fontsize=10)


def plot_regression_error(
        y: pd.Series,
        y_pred: pd.Series,
        target: str,
        label: str,
        figsize: tuple = (30, 7),
        figposition: int = 111
) -> None:
    """Plot scatter distribution of real against predicted values"""
    valid_labels = ['train', 'test', 'validation']
    if label.lower() not in valid_labels:
        raise ValueError(f'ERROR: input label must be one of {valid_labels}')

    plt.figure(figsize=figsize)
    ax = plt.subplot(figposition)
    ax.scatter(y, y_pred, label='{} error'.format(label.lower()))
    ax.plot(ax.get_xlim(), ax.get_xlim(), 'r-', label='diagonal')
    plt.xlabel(f'{target.title()}')
    plt.ylabel(f'Predicted {target.lower()}')
    plt.title(f'ERRORS ON {label.upper()} SET')
    ax.legend()


def plot_regression_errors(reg_model, x_train, x_val, y_train, y_val, target, figsize=(30, 7)) -> None:
    """Plot scatter distribution of real againsta predicted values for both train and validation sets"""
    # make predictions
    y_train_pred = reg_model.predict(x_train)
    y_val_pred = reg_model.predict(x_val)

    # assess data types
    if y_train is pd.core.frame.DataFrame:
        y_train = convert_pd_dataframe_to_np_array(y_train)
    if y_val is pd.core.frame.DataFrame:
        y_val = convert_pd_dataframe_to_np_array(y_val)

    plt.figure(figsize=figsize)
    plot_regression_error(y_train, y_train_pred, target=target, label='train', figposition=121)
    plot_regression_error(y_val, y_val_pred, target=target, label='validation', figposition=122)


def show_feature_importances(model, cols: list) -> None:
    """Display features importance for given model"""
    feat_imp = model.feature_importances_
    return pd.DataFrame(feat_imp, index=cols, columns=['importance']).sort_values('importance', ascending=False)


def rmse_cv(model, x: pd.Series, y: pd.Series, scoring: str = 'neg_mean_squared_error', n_folds: int = 10) -> None:
    """Calculates cross validation score of a model on data"""
    kf = KFold(n_folds, shuffle=True, random_state=123)
    rmse = np.sqrt(-cross_val_score(model, x, y, scoring=scoring, cv=kf))
    return rmse


def plot_nn_performances(history_dict, kpi) -> None:
    """Plot feed forward neural network performances"""
    train_score = history_dict[kpi]
    val_score = history_dict[f'val_{kpi}']
    epochs = range(1, len(train_score) + 1)

    plt.plot(epochs, train_score, 'b.', label=f'Train {kpi}')
    plt.plot(epochs, val_score, 'r.', label=f'Val {kpi}')
    plt.title(f'{kpi.upper()}: train vs validation')
    plt.xlabel('Epochs')
    plt.ylabel(kpi.title())
    plt.legend()

    plt.show()


# TENSORFLOW FUNCTIONS

def create_classification_nn(
        input_dim: int,
        hidden_layers: int = 1,
        hidden_layers_neurons: int = 128,
        dropout: float = .0,
        learning_rate: float = 0.001
):
    """Create a feed forward neural network for classification"""
    # clear tf
    K.clear_session()

    # initialize nn
    nn = Sequential()

    # add layers
    for _ in range(hidden_layers):
        nn.add(
            Dense(
                hidden_layers_neurons,
                input_dim=input_dim,
                activation='relu'
            ))
        nn.add(Dropout(dropout))

    # add final sigmoid activation function
    nn.add(Dense(1, activation='sigmoid'))

    # add optimizer
    opt = Adam(learning_rate=learning_rate)

    nn.compile(
        loss='binary_crossentropy',
        optimizer=opt,
        metrics=['accuracy'])

    return nn


def create_regression_nn(
        input_dim: int,
        hidden_layers: int = 1,
        hidden_layers_neurons: int = 128,
        dropout: float = .0,
        learning_rate: float = 0.001
):
    """Create a feed forward neural network for regression"""
    # clear tf
    K.clear_session()

    # initialize nn
    nn = Sequential()

    # add layers
    for _ in range(hidden_layers):
        nn.add(
            Dense(
                hidden_layers_neurons,
                input_dim=input_dim,
                kernel_initializer='normal',
                activation='relu'
            ))
        nn.add(Dropout(dropout))

    # add final sigmoid activation function
    nn.add(Dense(1, kernel_initializer='normal'))

    # add optimizer
    opt = Adam(learning_rate=learning_rate)

    nn.compile(
        loss='mean_squared_error',
        optimizer=opt)

    return nn
