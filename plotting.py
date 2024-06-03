import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils import get_nan_correlation_df, get_categorical_comparison_df

def plot_correlation_heatmap(df):
    """
    """
    # Modified this example to generate this plot:
    # https://seaborn.pydata.org/examples/many_pairwise_correlations.html
    
    # build the correlation matrix for nans appearing in df
    df_nan_corr = get_nan_correlation_df(df)

    #mask = np.triu(np.ones_like(df_nan_corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    ax = sns.heatmap(df_nan_corr, center=0, vmin=-1, vmax=1, cmap=cmap,
                     square=True, linewidths=.5, cbar_kws={"shrink": .5},
                     xticklabels=True, yticklabels=True)
    
    ax.tick_params(labelsize=8)

    return fig, ax

def plot_correlation_clustermap(df):

    # build the correlation matrix for nans appearing in df
    df_nan_corr = get_nan_correlation_df(df)

    #fig, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Generate a heatmap that is ordered by hierarchical clustering
    clustergrid = sns.clustermap(df_nan_corr, cmap=cmap, vmin=-1.0, vmax=1.0, linewidths=.5, cbar_kws={"shrink": .5},
                   xticklabels=True, yticklabels=True)
    
    clustergrid.tick_params(labelsize=8)

    # TODO: investigate the clustergrid object in my detail and find a way to change the font size
    #       of the heatmap labels independently of the legend scale (which does not need to be made
    #       any smaller)
    #plt.xticks(fontsize=5)
    #plt.yticks(fontsize=5)
    clustergrid.tick_params("both", labelsize=8)    

    fig = plt.gcf()
    ax = plt.gca()

    return fig, ax


def plot_comparison_by_unique_value(df1, df2, column_name, ax=None, xrotation=None, label1="keeping", label2="dropping"):
    """
    """
    num1 = (~df1[column_name].isna()).sum()
    num2 = (~df2[column_name].isna()).sum()
    plot_df = get_categorical_comparison_df(df1, df2, column_name, label1=label1, label2=label2)
    if ax is None:
        _, ax = plt.subplots()
    ax = sns.barplot(plot_df, x=column_name, y="proportion", hue="category", ax=ax)
    
    # rotate x-axis labels
    if xrotation is None:
        # if rotation isn't specified, choose it automatically.  Scale between no rotation with very few categories (<10)
        # to 70 degree rotation for many categories (>30)
        num_unique = len(plot_df[column_name].unique())
        if num_unique < 10:
            xrotation = 0
        elif num_unique > 30:
            xrotation = 70
        else:
            xrotation = (70/20) * (num_unique - 10)
    ax.tick_params(axis="x", labelrotation=xrotation)
    
    ax.set_title(f"num_{label1} = {num1}  num_{label2} = {num2}")

    # TODO: it would be nice to be able to modify the labels in the legend to include the number of points,
    #       but this doesn't seem straightforward to do while maintaining the correct colors.  I will leave
    #       this improvement for the future.  Starting point:
    #       https://stackoverflow.com/questions/45201514/how-to-edit-a-seaborn-legend-title-and-labels-for-figure-level-functions
    # ax.legend(labels=[f"keep (N = {num_keep})", f"drop (N = {num_drop})"])

    return ax


def plot_comparison_by_histogram(df1, df2, column, ax=None, transparency=0.7, label1="keeping", label2="dropping"):
    """
    """
    num1 = (~df1[column].isna()).sum()
    num2 = (~df2[column].isna()).sum()

    ds1 = df1[column].dropna()
    ds2 = df2[column].dropna()

    mn = min(ds1.min(), ds2.min())
    mx = max(ds1.max(), ds2.max())
    width = mx - mn
    bin_edges = np.linspace(mn-0.01*width, mx+0.01*width, 20)

    if ax is None:
        _, ax = plt.subplots()

    ax.hist(ds1, bins=bin_edges, density=True, label="keeping", alpha=transparency)
    ax.hist(ds2, bins=bin_edges, density=True, label="dropping", alpha=transparency)
    ax.set_ylabel("probability density", fontsize=16)
    ax.set_xlabel(column, fontsize=16)
    ax.set_title(f"num_keep = {num1}  num_drop = {num2}")
    ax.legend()

    return ax


def plot_pca_component_importance(pca, ax=None, top_n=None):
    """
    """
    num_components = len(pca.explained_variance_ratio_)
    if top_n is None:
        data = [[n, variance_ratio] for n, variance_ratio in enumerate(pca.explained_variance_ratio_)]
    else:
        data = [[n, variance_ratio] for n, variance_ratio in enumerate(pca.explained_variance_ratio_[:top_n])]
    plot_df = pd.DataFrame(data, columns=["component", "variance_ratio"])

    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 9))
    ax = sns.barplot(plot_df, x="component", y="variance_ratio", orient="x", ax=ax)
    ax.set_title("Ranked importance of each PCA component", fontsize=18)
    ax.set_ylabel("variance ratio", fontsize=20)
    ax.set_xlabel("PCA component", fontsize=20)
    xticklabels = ["",] * len(plot_df)
    for i in range(0, len(plot_df), 5):
        xticklabels[i] = str(i)
    ax.set_xticklabels(xticklabels)

    return ax



def plot_pca_component_cumulative_importance(pca, ax=None, top_n=None):
    """
    """
    x = np.arange(0, len(pca.explained_variance_ratio_), 1)
    y = np.cumsum(pca.explained_variance_ratio_)
    if top_n is not None:
        x = x[:top_n]
        y = y[:top_n]

    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 12))

    ax.plot(x, y)
    ax.set_title("Cumulative importance of PCA components", fontsize=18)
    ax.set_xlabel("PCA component", fontsize=20)
    ax.set_ylabel("cumulative variance ratio", fontsize=20)

    return ax


def plot_pca_components_heatmap(df_pca_components, ax=None):
    """
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 18))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    num_eigenvectors = len(df_pca_components.columns) - 2
    xticklabels = ["",]*num_eigenvectors
    for i in range(0, num_eigenvectors, 5):
        xticklabels[i] = str(i)

    columns = df_pca_components["column_name"].tolist()
    data = df_pca_components.iloc[:,2:].to_numpy()
    mn = data.min()
    mx = data.max()

    ax = sns.heatmap(data, xticklabels=xticklabels, yticklabels=columns, ax=ax,
                     center=0, vmin=mn, vmax=mx, cmap=cmap, linewidths=0.5,
                     cbar_kws={"shrink" : 0.5})
    ax.tick_params(labelsize=8)
    ax.set_xlabel("Eigenvector number", fontsize=16)

    return ax


def plot_single_pca_component_bargraph(df_single_component, ax=None):
    """
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(22, 9))

    component_num = df_single_component.columns[-1]
    ax = sns.barplot(df_single_component, x="column_name", y=component_num, ax=ax)
    ax.set_xticklabels(df_single_component.index.tolist())
    ax.set_title(f"sorted eigenvector {component_num} values", fontsize=25)
    ax.set_xlabel("column number", fontsize=20)
    ax.set_ylabel("value in eigenvector", fontsize=20)
    ax.tick_params(axis="x", labelrotation=90)
    

    return ax