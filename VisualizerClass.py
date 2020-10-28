class Visualizer:
    
    """[A class made for ease of visualizations - built using the plotly graph_objects library]
    
    Color Options: 
    	  ["Teal" : "#42cbf5",
    	  "Neon Green" : "#39ff14",
    	  "Neon Orange" : "#FD5F00"]
    	  
    Returns:
        [fig]: [a plotly figure (of your choice) that can be viewed using fig.show()]
    """    

    # ============= BAR CHART =============    
    def bar(title, x, y, x_title, y_title, height, width, colors):
        fig = go.Figure(data=[go.Bar(x=x, y=y, text=y, textposition='auto', 
                              marker_color=colors)])

        fig.update_layout(
            title_text=title, bargap=0.1, 
            height=height, width=width, template='plotly_dark', 
            xaxis=dict(title=x_title, showgrid=False),
            yaxis=dict(title=y_title, showgrid=False)
        )

        return fig

    # ============= HISTOGRAM =============
    def histogram(title, x, x_title, y_title, nbinsx, height, width, colors):
        fig = go.Figure(data=[go.Histogram(x=x, nbinsx=nbinsx, marker_colors=colors)])

        fig.update_layout(
            title_text = title, bargap=0.025, 
            height=height, width=width, template='plotly_dark',
            xaxis=dict(title=x_title, showgrid=False),
            yaxis=dict(title=y_title, showgrid=False)
        )

        return fig   

    # ============= DISTPLOT ==============
    def distplot(title, hist_data, group_labels, bin_size, colors, height, width):
        fig = ff.create_distplot(hist_data, group_labels, colors=colors, bin_size=bin_size,
                                 show_hist=False, show_curve=True, show_rug=False)
        
        fig.update_layout(title_text=title, 
                          height=height, width=width, template='plotly_dark',
                          xaxis=dict(showgrid=False),
                          yaxis=dict(showgrid=False))

        return fig

    # ============ SCATTERPLOT ============
    def scatterplot(title, x, y, x_title, y_title, height, width, color):
        fig = go.Scatter(x=z, y=y, mode='lines+markers', name='Loss', marker_color=color)

        fig.update_layout(title_text=title, 
                          height=height, width=width, template='plotly_dark',
                          xaxis=dict(title=x_title, showgrid=False),
                          yaxis=dict(title=y_title, showgrid=False))

        return fig 

    # ============= PIECHART ==============
    def piechart(title, x, y):
        fig = go.Figure(data=[go.Pie(labels=x, values=y,  
                                     textinfo='label+percent', insidetextorientation='auto')])               

        fig.update_layout(title_text=title, template='seaborn')
        fig.update_traces(hoverinfo='label+percent')

        return fig