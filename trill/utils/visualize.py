import numpy as np
import pandas as pd
import bokeh
from umap import UMAP
# import holoviews as hv
from bokeh.io import output_file, save, show
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from bokeh.plotting import figure
from bokeh.models import HoverTool
# hv.extension('bokeh')




def reduce_dims(name, data, method = 'PCA'):
    incsv = data.split('.csv')[0]
    data = pd.read_csv(data)
    labels = data.iloc[:,-1:]
    data = data.iloc[:,:-1]
    if method == 'PCA':
        reducer = PCA(n_components=2, random_state=123)
        reduced = reducer.fit_transform(data.values)
        var_1, var_2 = reducer.explained_variance_
        reduced_df = pd.DataFrame(reduced, columns=[f'PCA 1: {var_1}', f'PCA 1: {var_2}'])
        reduced_df['Label'] = labels
        reduced_df.to_csv(f'{name}_{method}_{incsv}.csv', index=False)

    elif method == 'tSNE':
        if len(data) <= 30:
            reduced = TSNE(n_jobs=-1, random_state=123, perplexity=(len(data)-1)).fit_transform(data.values)
        else:
            reduced = TSNE(n_jobs=-1, random_state=123).fit_transform(data.values)
        reduced_df = pd.DataFrame(reduced, columns = ['tSNE 1', 'tSNE 2'])
        reduced_df['Label'] = labels
        reduced_df.to_csv(f'{name}_{method}_{incsv}.csv', index=False)

    elif method == 'UMAP':
        reduced = UMAP(random_state=123).fit_transform(data.values)
        reduced_df = pd.DataFrame(reduced, columns = ['UMAP 1', 'UMAP 2'])
        reduced_df['Label'] = labels
        reduced_df.to_csv(f'{name}_{method}_{incsv}.csv', index=False)

    else:
        raise Exception(f'Dimensionality reduction method {method} needs to be either PCA, tSNE or UMAP')


    return reduced_df, incsv


def create_group(row):
    return row.split('_')[-1]


def viz(reduced_df, title, grouped):
    col1, col2, _ = reduced_df.columns
    fig = figure(
        title=title,
        width=600, 
        height=600, 
        x_axis_label = col1, 
        y_axis_label = col2
        )
    if grouped == True:
        reduced_df['Group'] = reduced_df['Label'].apply(create_group)
        # palette from http://mkweb.bcgsc.ca/colorblind/palettes/15.color.blindness.palette.txt
        palette = ['#68023F',
                '#008169',
                '#EF0096',
                '#00DCB5',
                '#FFCFE2',
                '#003C86',
                '#9400E6',
                '#009FFA',
                '#FF71FD',
                '#7CFFFA',
                '#6A0213',
                '#008607',
                '#F60239',
                '#00E307',
                '#FFDC3D']
        for group, color in zip((reduced_df.Group.unique()), palette):
            fig.circle(
            source = reduced_df.loc[reduced_df['Group'] == group, :],
            color = color,
            x = col1,
            y = col2,
            legend_label = group
            )
        

        fig.add_tools(
            HoverTool(
                tooltips=[("Protein", "@Label")]
                )
            )
        fig.legend.click_policy="hide"

        return(fig)
    else:
        fig.circle(
        source = reduced_df,
        color = '#68023F',
        x = col1,
        y = col2,
        )
    

        fig.add_tools(
            HoverTool(
                tooltips=[("Protein", "@Label")]
                )
            )

        return(fig)
        




    