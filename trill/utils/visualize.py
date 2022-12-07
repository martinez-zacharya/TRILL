import numpy as np
import pandas as pd
import bokeh
# import holoviews as hv
from bokeh.io import output_file, save, show
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from bokeh.plotting import figure
from bokeh.models import HoverTool
# hv.extension('bokeh')

def tsne(data, labels):
    data = np.load('../data/' + str(data) + '.npy')
    tsne = PCA(n_components=20).fit_transform(data) #why 20???
    tsne = TSNE(n_components=2, n_jobs = -1).fit_transform(tsne)
    tsnedf = pd.DataFrame(data = tsne, columns = ['tSNE 1', 'tSNE 2'])

    tsnedf = pd.concat([tsnedf, labels], axis = 1)
    
    return tsnedf

def scatter_viz(df):

    fig = figure(
        width=800, 
        height=800, 
        x_axis_label = 'tSNE 1', 
        y_axis_label = 'tSNE 2'
    )
    
    
    palette = ['#648FFF', '#785EF0', '#DC267F', '#FE6100', '#FFB000']
    
    for label, color in zip((df.Label.unique()), palette):
        label = str(label)
        fig.circle(
            source = df.loc[df['Label'] == label, :],
            color = color,
            x = 'tSNE 1',
            y = 'tSNE 2',
            legend_label = label
        )

    fig.add_tools(
        HoverTool(
            tooltips=[("Gene", "@Gene")]
            )
        )
    fig.legend.click_policy="hide"

#     save(fig)

    return fig
    