import numpy as np
import pandas as pd
import bokeh
import holoviews as hv
from bokeh.io import output_file, save, show
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from bokeh.plotting import figure
from bokeh.models import HoverTool
hv.extension('bokeh')

def tsne(data, labels):
	data = np.load('../data/' + str(data) + '.npy')
	# tsne = PCA(n_components=20).fit_transform(data) #why 20???
	tsne = PCA(n_components=2).fit_transform(data) #temp for debugging
	tsne = TSNE(n_components=2).fit_transform(tsne)
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

	fig.circle(
		source = df.loc[df['Label'] == 1, :],
		color = '#4daf4a',
		x = 'tSNE 1', 
		y = 'tSNE 2', 
		size = 3
		)

	fig.circle(
		source = df.loc[df['Label'] == 0, :],
		color = '#e41a1c',
		x = 'tSNE 1', 
		y = 'tSNE 2', 
		size = 3
		)

	fig.add_tools(
		HoverTool(
			tooltips=[("Gene", "@Gene")]
			)
		)

	save(fig)
	# tsne_plot = hv.Points(
	# 	data=df,
	# 	kdims=["tSNE 1", "tSNE 2"],
	# 	vdims = ['Label']
	# ).groupby(
	# 	['Label']
	# ).overlay(
	# ).opts(
	# 	legend_position='right',
	# 	height=600,
	# 	width=750,
	# 	xlabel = "tSNE 1",
	# 	ylabel = "tSNE 2"
	# )

	# renderer = hv.renderer('bokeh')
	# renderer.save(tsne_plot, 'tSNE')

	return True