import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

def default_chart(chart_title,chart_key,chart_df,chart_height=630, radio_horizontal=True, color_theme='streamlit'):
    with st.container(height=chart_height):
        # st.subheader(chart_title)
        st.markdown(f'**{chart_title}**')
        col_1, col_2 = st.columns([4,1])
        with col_2:
            chart_type = st.radio(
                "",
                ["Line Chart", "Bar Chart"], horizontal=radio_horizontal, key=chart_key)

        if chart_type == 'Line Chart':
            fig = px.line(data_frame=chart_df) # chart_df=df.groupby('Purchase Date')['Net Sales'].sum()
        else:
            fig = px.bar(data_frame=chart_df)

        fig.update(layout_showlegend=False)
        # fig.update_traces(line=dict(color="Yellow", width=0.4))

        st.plotly_chart(fig, theme=color_theme, use_container_width=True)


def twin_axis_chart():
    x = ['Jan', 'Feb', 'Mar', 'Apr', 'May']
    y1 = [10, 20, 15, 25, 30]  # Line plot data
    y2 = [50, 40, 35, 30, 25]  # Bar plot data

    # Create traces
    fig = go.Figure()

    # Line plot
    fig.add_trace(go.Scatter(x=x, y=y1, mode='lines', name='Line Plot'))

    # Bar plot
    fig.add_trace(go.Bar(x=x, y=y2, name='Bar Plot', yaxis='y2'))

    # Create layout with y-axis and y-axis2
    fig.update_layout(
        title='Twin Axis Chart',
        yaxis=dict(title='Line Plot'),
        yaxis2=dict(title='Bar Plot', overlaying='y', side='right')
    )

    # Show the figure
    fig.show()

