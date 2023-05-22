import numpy as np
import pandas as pd
import bokeh
from umap import UMAP
# import holoviews as hv
from bokeh.io import output_file, save, show
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from bokeh.models import CustomJSFilter, CDSView, ColumnDataSource, TextInput, CustomJS, HoverTool, GroupFilter
from bokeh.plotting import figure, curdoc, output_notebook, show
from bokeh.layouts import column
from bokeh.resources  import settings

# from bokeh.models import ColumnDataSource, HoverTool, CustomJS, TextInput, CDSView, CustomJSFilter, GroupFilter
# from bokeh.io import output_notebook, show
# from bokeh.layouts import column
# import pandas as pd
# from bokeh.plotting import figure
# hv.extension('bokeh')




def reduce_dims(name, data, method = 'PCA'):
    incsv = data.split('.csv')[0].split('/')[-1]
    data = pd.read_csv(data)
    labels = data.iloc[:,-1:]
    data = data.iloc[:,:-1]
    if method == 'PCA':
        reducer = PCA(n_components=2, random_state=123)
        reduced = reducer.fit_transform(data.values)
        var_1, var_2 = reducer.explained_variance_ratio_
        reduced_df = pd.DataFrame(reduced, columns=[f'PCA 1: {var_1}', f'PCA 2: {var_2}'])
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

def viz(df, title, grouped):
    col1, col2, _ = df.columns
    
    # Check if grouped is True
    if grouped:
        # Add 'Group' column based on 'Label' values
        df['Group'] = df['Label'].apply(create_group)
        # Create a column data source from the dataframe
        source = ColumnDataSource(df)
        
        # Define the palette for coloring groups
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
        
        # Create a figure
        fig = figure(title=title, width=600, height=600, x_axis_label=col1, y_axis_label=col2)
        # Create a TextInput widget for the search feature
        text_input = TextInput(value='', title='Search:')
        custom_filter = CustomJSFilter(args=dict(text_input=text_input), code="""
            var indices = [];
            var data = source.data;
            var names = data['Label'];
            var value = text_input.value;  // get the value from TextInput widget

            // Iterate over the data and select the indices of points to keep based on the filter value
            for (var i = 0; i < names.length; i++) {
                if (names[i].includes(value)) {
                    indices.push(i);
                }
            }
            return indices;
        """)
        
        # Create a scatter plot for each group and add them to the figure
        scatter_renderers = []
        for group, color in zip(df['Group'].unique(), palette):
            # Create a GroupFilter for the group
            group_filter = GroupFilter(column_name='Group', group=group)
            
            # Create a scatter plot for the group
            scatter_renderer = fig.circle(col1, col2, source=source, color=color, legend_label=group, view=CDSView(source=source, filters=[custom_filter, group_filter]))
            scatter_renderers.append(scatter_renderer)
        
        # Add hover tool
        fig.add_tools(HoverTool(renderers=scatter_renderers, tooltips=[('Protein', '@Label')]))
        
        # Update the CustomJS callback when the input value changes
        text_input.js_on_change('value', CustomJS(args=dict(source=source, scatter_renderers=scatter_renderers), code="""
            for (var i = 0; i < scatter_renderers.length; i++) {
                scatter_renderers[i].data_source.change.emit();
            }
        """))
        
        # Add legend
        fig.legend.click_policy = 'hide'
        
        # Create the layout with the TextInput widget and the figure
        layout = column(text_input, fig)
        
        # Show the plot
        return layout

    else:
        source = ColumnDataSource(df)
        # Create a simple scatter plot without grouping
        fig = figure(title=title, width=600, height=600, x_axis_label=col1, y_axis_label=col2)
        # Create a CustomJSFilter
        # Create a CustomJSFilter
        text_input = TextInput(value='', title='Search:')
        custom_filter = CustomJSFilter(args=dict(text_input=text_input, data=dict(source.data)), code="""
            var indices = [];
            var names = data['Label'];
            var value = text_input.value;  // get the value from TextInput widget

            // Iterate over the data and select the indices of points to keep based on the filter value
            for (var i = 0; i < names.length; i++) {
                if (names[i].includes(value)) {
                    indices.push(i);
                }
            }
            return indices;
        """)
        view = CDSView(source=source, filters=[custom_filter])
        scatter_renderer = fig.circle(col1, col2, source=source, view=view, color='#68023F')
        # Add hover tool
        fig.add_tools(HoverTool(renderers=[scatter_renderer], tooltips=[('Protein', '@Label')]))
        # Update the CustomJS callback when the input value changes
        text_input.js_on_change('value', CustomJS(args=dict(source=source, scatter_renderer=scatter_renderer), code="""
            scatter_renderer.data_source.change.emit();
        """))

        # Create the layout with the TextInput widget and the figure
        layout = column(text_input, fig)
        return layout




    
