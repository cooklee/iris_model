import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.obtain.obtain import prepare_data_from_sklearn
from sklearn.feature_selection import SelectKBest, f_classif
from src.model.model import prepare_data_frame
from sklearn.decomposition import PCA

sns.set_context(context='notebook', font_scale=1.4)
sns.set_style(style='white', rc={'font.family': 'Monospace'})

df = prepare_data_from_sklearn()
# print (df.describe())#%pylab inli# ne


def plot_function(ax, title=None, xlabel=None, ylable=None, path=None, text=None):
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylable is not None:
        ax.set_ylabel(ylable)
    if text is not None:
        ax.text(0, 7, text, fontsize=16)
    if path is not None:
        plt.savefig(path, bbox_inches='tight')


def check_class_inbalance(data_frame, column='iris_type'):
    fig, ax = plt.subplots(figsize=(8, 4))
    data_frame[column].value_counts(normalize=True).plot.barh(ax=ax)

    plot_function(ax, title='There is no class imbalance', xlabel='% of cases', ylable='Class',
                  path='../../figures/01-class-distribution.png')


def variation_across_groups(data_frame, column='iris_type'):
    fig, ax = plt.subplots(figsize=(10, 6))
    data_frame.groupby(column).mean().T.plot.barh(ax=ax)

    plot_function(ax, title='Very litte variation across groups in Sepal Width, \nbut Petal Length might be an important predictor \n',
                  xlabel='\n Mean Values', ylable='Features \n',
                  path='../../figures/02-across-groups-variation.png')


def anova(data_frame):
    skb = SelectKBest(k='all', score_func=f_classif)

    X, y = prepare_data_frame(data_frame)
    skb.fit(X, y)

    anova_results = pd.DataFrame({'F-score': skb.scores_,  'P-value': skb.pvalues_}, index=df.columns[:4])
    fig, ax = plt.subplots(figsize=(8, 4))
    anova_results.plot.barh(ax=ax)

    plot_function(ax,
                  title="ANOVA Results: All predictors are significant \n Petal Length and width are most important \n",
                  xlabel="\n Score",
                  ylable="Predictor \n",
                  path="../../figures/03-ANOVA-results.png")

def principal_components_analysis(data_frame):
    pca = PCA(n_components=2)
    X, y = prepare_data_frame(data_frame)
    pca.fit(X, y)
    (pd.Series(pca.explained_variance_ratio_,
               index=['Component 1', 'Component 2'])
        .plot.barh(title="2-component solution explains >95% of the variance \n", figsize=(8, 4))
        )

    plt.savefig("../../figures/04-PCA-results.png", bbox_inches='tight')
    X_2dimensions = pd.DataFrame(pca.transform(X), columns=['C1', 'C2'])
    fig, ax = plt.subplots(figsize=(7, 4))
    X_2dimensions.plot.scatter(x='C1', y='C2', c=y, cmap='winter', ax=ax)

    ax.set_title("Visualize Classification Boundary in 2-dimensions \n")
    ax.set_xlabel("\n Component 1")
    ax.set_ylabel("Component 2 \n")
    ax.text(7, 0, """
    Setosa will be easily classifiable, 
    but we will find it hard to distinguish 
    between some virginica and versicolor""", fontsize=16)

    plt.savefig("../../figures/05-classification-boundary.png", bbox_inches='tight')


if "__main__" == __name__:
    check_class_inbalance(df)
    variation_across_groups(df)
    anova(df)
    principal_components_analysis(df)