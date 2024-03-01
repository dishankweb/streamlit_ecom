import streamlit as st
import pandas as pd
import plotly.express as px
# from faker import Faker
import random
import numpy as np
import math
import datetime
import uuid
# import matplotlib.pyplot as plt
import altair as alt
import plotly.graph_objects as go
from dateutil.relativedelta import relativedelta

# src imports
from src.charts.tile import kpi_tile
from src.charts.charts import default_chart,twin_axis_chart

date_today = datetime.datetime.now().date()
date_today = datetime.date(2024, 2, 23)

st.set_page_config(layout="wide", page_title="E-commerce Dashboard")

def date_change_timedelta(string_1): # convert date string to timedelta
    return datetime.datetime.strptime(string_1, "%Y-%m-%d").date()

def previous_time_delta_percentage(dataframe, date_today):
    start_date = date_today.replace(day=1)
    dataframe_ = dataframe[(dataframe['Purchase Date'] >= start_date) & (dataframe['Purchase Date'] <= date_today)]
    delta_start_date = start_date - relativedelta(months=1)
    delta_end_date = date_today - relativedelta(months=1)
    dataframe_delta = dataframe[(dataframe['Purchase Date'] >= delta_start_date) & (dataframe['Purchase Date'] <= delta_end_date)]
    return dataframe_, dataframe_delta

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Main df read
df = pd.read_csv('fake_ecom_data.csv', encoding='utf-8')
df['Purchase Date'] = df['Purchase Date'].apply(date_change_timedelta) # date string to timedelta

# Streamlit dashboard layout

col1, col2 = st.columns([3, 1])    
with col1:
    st.title('E-Commerce Executive Dashboard')
with col2:
    option = st.selectbox('',
        ('This Month','Last 7 Days', 'Last 15 Days', 'Last 30 Days', 'This Quarter', 'This Year', 'All Data', 'Custom Range'))

if option == 'Last 7 Days': 
    end_date = df.iloc[df['Purchase Date'].idxmax()]['Purchase Date']
    start_date = end_date - pd.Timedelta(days=7)
    df = df[(df['Purchase Date'] > start_date) & (df['Purchase Date'] <= end_date)]
elif option == 'Last 15 Days':
    end_date = df.iloc[df['Purchase Date'].idxmax()]['Purchase Date']
    start_date = end_date - pd.Timedelta(days=15)
    df = df[(df['Purchase Date'] > start_date) & (df['Purchase Date'] <= end_date)]
elif option == 'Last 30 Days':
    end_date = df.iloc[df['Purchase Date'].idxmax()]['Purchase Date']
    start_date = end_date - pd.Timedelta(days=30)
    df = df[(df['Purchase Date'] > start_date) & (df['Purchase Date'] <= end_date)]
elif option == 'This Month':
    # end_date = df.iloc[df['Purchase Date'].idxmax()]['Purchase Date']
    # start_date = end_date - pd.Timedelta(days=7)
    # df = df[(df['Purchase Date'] > start_date) & (df['Purchase Date'] <= end_date)]
    df,df_delta = previous_time_delta_percentage(dataframe=df, date_today=date_today)
else:
    pass

# KPIs
total_revenue = df['Net Sales'].sum()
total_orders = df['Order ID'].nunique()
average_order_value = df['Net Sales'].mean()
new_customers = df[df['Customer Type'] == 'New'].shape[0]
repeat_customers = df[df['Customer Type'] == 'Repeat'].shape[0]

# Display KPIs

listTabs = ['Business Overview', 'Website vs Marketplace', 'Geo Sales', 'Product Insights', 'Retention', 
            'Cross-Sell', 'Performance Marketing']
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([s.center(20,"\u2001") for s in listTabs]) # [s.center(29,"\u2001")

# listTabs = ['Business Overview', 'Website vs Marketplace', 'Geo Sales', 'Product Insights', 'Retention', 'Performance Marketing']
# tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([s.center(20,"\u2001") for s in listTabs]) # [s.center(29,"\u2001")
# tab1, tab2 = st.tabs(['Revenue', 'Retention'])

total_orders = df['Order ID'].nunique()
total_revenue = df['Net Sales'].sum()
aov = total_revenue / total_orders if total_orders else 0
# Run the Streamlit app by saving this script as `app.py` and running `streamlit run app.py` in your terminal.

with tab1:
    
    empty, empty,empty, col_4 = st.columns(4)
    with col_4:
        # option = st.selectbox(
        # '',
        # ('Last 7 Days', 'Last 15 Days', 'Last 30 Days', 'This Month', 'This Quarter', 'This Year', 'All Data', 'Custom Range'))
        # df = generate_data()

        # KPIs
        new_customers = df[df['Customer Type'] == 'New'].shape[0]
        repeat_customers = df[df['Customer Type'] == 'Repeat'].shape[0]
 
        total_orders = df['Order ID'].nunique()
        total_orders_website = df.loc[df['Sales Channel']== 'Mobile App', :]['Order ID'].nunique()
        total_orders_marketplace = df.loc[df['Sales Channel']== 'Online', :]['Order ID'].nunique()
        total_revenue = df['Net Sales'].sum()
        total_revenue_website = df.loc[df['Sales Channel']== 'Mobile App', :]['Net Sales'].sum()
        total_revenue_marketplace = df.loc[df['Sales Channel']== 'Online', :]['Net Sales'].sum()
        aov = total_revenue / total_orders if total_orders else 0
        aov_website = total_revenue_website / total_orders_website if total_orders_website else 0
        aov_marketplace = total_revenue_marketplace / total_orders_marketplace if total_orders_marketplace else 0

        # Delta KPIs
        new_customers_delta = df_delta[df_delta['Customer Type'] == 'New'].shape[0]
        repeat_customers_delta = df_delta[df_delta['Customer Type'] == 'Repeat'].shape[0]

        total_orders_delta = df_delta['Order ID'].nunique()
        total_orders_website_delta = df_delta.loc[df_delta['Sales Channel']== 'Mobile App', :]['Order ID'].nunique()
        total_orders_marketplace_delta = df_delta.loc[df_delta['Sales Channel']== 'Online', :]['Order ID'].nunique()
        total_revenue_delta = df_delta['Net Sales'].sum()
        total_revenue_website_delta = df_delta.loc[df_delta['Sales Channel']== 'Mobile App', :]['Net Sales'].sum()
        total_revenue_marketplace_delta = df_delta.loc[df_delta['Sales Channel']== 'Online', :]['Net Sales'].sum()
        aov_delta = total_revenue_delta / total_orders_delta if total_orders else 0
        aov_website_delta = total_revenue_website_delta / total_orders_website_delta if total_orders_website_delta else 0
        aov_marketplace_delta = total_revenue_marketplace_delta / total_orders_marketplace_delta if total_orders_marketplace_delta else 0
    
    
    kpi1, kpi2, kpi3, kpi4 = st.columns(4) # row 1 - 4 KPIs
    
    
    with st.container():
        with kpi1:
            kpi_tile(kpi1,tile_text='Blended Revenue', tile_label='', tile_value=total_revenue,
                     tile_value_prefix='$',delta_value=(total_revenue-total_revenue_delta)*100/total_revenue_delta,integer=True)
            # tile = kpi1.container(height=240)
            # tile.header('Blended Revenue')
            # tile.metric(label="", value=f"${total_revenue:,.2f}", delta='9.3%')

        with kpi2:
            kpi_tile(kpi2,tile_text='Blended Orders', tile_label='', tile_value=total_orders,
                     tile_value_prefix='',delta_value=(total_orders-total_orders_delta)*100/total_orders_delta,integer=True)
            # tile = kpi2.container(height=240)
            # tile.header('Blended Orders')
            # tile.metric(label="", value=f"{total_orders:,}", delta='-1.2%')
            
        with kpi3:
            kpi_tile(kpi3,tile_text='Blended AOV', tile_label='', tile_value=aov,
                     tile_value_prefix='$',delta_value=(aov-aov_delta)*100/aov_delta,integer=True)
            # tile = kpi3.container(height=240)
            # tile.header('Blended AOV')
            # tile.metric(label="", value=f"${aov:,.2f}", delta='11%')

        with kpi4:
            kpi_tile(kpi4,tile_text='Blended New Customers', tile_label='', tile_value=new_customers,
                     tile_value_prefix='',delta_value=2,integer=True)
            # tile = kpi4.container(height=240)
            # tile.header('Blended New Customers')
            # tile.metric(label="", value=f"{new_customers:,}", delta='6%') 

        # Next row
        kpi5, kpi6, kpi7 = st.columns(3) # row 2 - 3 KPIs

        with kpi5:
            kpi_tile(kpi5,tile_text="**Blended Repeat Customers**", tile_label='', tile_value=repeat_customers,
                     tile_value_prefix='',delta_value=1.2,integer=True)
            # tile = kpi5.container(height=240)
            # tile.header('Blended Repeat Customers')
            # tile.metric(label="", value=f"{math.floor(total_orders*0.75):,}", delta='6%')     

        with kpi6:
            kpi_tile(kpi6,tile_text='Blended Cancellation Rate', tile_label='', tile_value=1.3,
                     tile_value_prefix='',delta_value=0.8,integer=False, delta_color_inversion='inverse', tile_value_suffix='%')
            # tile = kpi6.container(height=240)
            # tile.header('Blended Cancellation Rate')
            # tile.metric(label="", value=f"{math.floor(total_orders*0.75):,}", delta='6%', delta_color='inverse') 

        with kpi7:
            kpi_tile(kpi7,tile_text='Blended Discounts', tile_label='', tile_value=total_revenue*0.12,
                     tile_value_prefix='$',delta_value=1.5,integer=True, delta_color_inversion='inverse')
            # tile = kpi7.container(height=240)
            # tile.header('Blended Discounts')
            # tile.metric(label="", value=f"{math.floor(total_orders*0.75):,}", delta='6%', delta_color='inverse') 
        
    # Total Revenue
            
    default_chart(chart_title='Blended Revenue [Daily]',chart_key='tab1_1',chart_df=df.groupby('Purchase Date')['Net Sales'].sum(),
                  chart_height=630, radio_horizontal=True, color_theme='streamlit')
    # with st.container(height=630):
    #     st.subheader('Blended Revenue [Daily]')
    #     col_1, col_2 = st.columns([4,1])
    #     with col_2:
    #         chart_type = st.radio(
    #             "",
    #             ["Line Chart", "Bar Chart"], horizontal=True, key=1)
    #     if chart_type == 'Line Chart':
    #         fig = px.line(data_frame=df.groupby('Purchase Date')['Net Sales'].sum())
    #     else:
    #         fig = px.bar(data_frame=df.groupby('Purchase Date')['Net Sales'].sum())
    #     fig.update(layout_showlegend=False)
    #     st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    # Total orders  
    default_chart(chart_title='Blended Orders [Daily]',chart_key='tab1_2',chart_df=df.groupby('Purchase Date')['Order ID'].nunique(),
                  chart_height=600, radio_horizontal=True, color_theme='streamlit')
    # with st.container(height=630):
    #     st.subheader('Blended Orders [Daily]')
    #     col_1, col_2 = st.columns([4,1])
    #     with col_2:
    #         chart_type_1 = st.radio(
    #             "",
    #             ["Line Chart", "Bar Chart"], horizontal=True, key=2)

    #     if chart_type_1 == 'Line Chart':
    #         fig = px.line(data_frame=df.groupby('Purchase Date')['Order ID'].nunique())
    #     else:
    #         fig = px.bar(data_frame=df.groupby('Purchase Date')['Order ID'].nunique())

    #     # fig = px.bar(data_frame=df.groupby('Purchase Date')['Order ID'].nunique(),
    #     #           title='Total Orders [Daily]')

    #     fig.update(layout_showlegend=False)
    #     # fig.update_traces(color="Yellow", width=0.4)

    #     st.plotly_chart(fig, theme="streamlit", use_container_width=True)


    with st.container(height=630):
        st.subheader('Blended AOV [Daily]')
        col_1, col_2 = st.columns([4,1])
        with col_2:
            chart_type_1 = st.radio(
                "",
                ["Line Chart", "Bar Chart"], horizontal=True, key=3)

        if chart_type_1 == 'Line Chart':
            fig = px.line(data_frame=df.groupby('Purchase Date')['Net Sales'].sum() / df.groupby('Purchase Date')['Order ID'].nunique())
        else:
            fig = px.bar(data_frame=df.groupby('Purchase Date')['Net Sales'].sum() / df.groupby('Purchase Date')['Order ID'].nunique())
        # fig = px.line(data_frame=df.groupby('Purchase Date')['Net Sales'].sum() / df.groupby('Purchase Date')['Order ID'].nunique(),
        #                 title='AOV [Daily]')

        fig.update(layout_showlegend=False)
        # fig.update_traces(line=dict(color="Yellow", width=0.4))

        st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    # tab = st.selectbox('Tab to shorten and make scrollable:', [1,2,3])

with tab2:
    col_0,_, col_00 = st.columns([1,3,1])
    col_1, col_2, col_4 = st.columns([1,3,1])
    col2_1, col2_2, col2_4 = st.columns([1,3,1])
    col3_1, col3_2, col3_4 = st.columns([1,3,1])

    with st.container():
        tile = col_0.container(height=100)
        tile.header('Website')
        tile = col_00.container(height=100)
        tile.header('Marketplace')

        

        kpi_tile(col_1,tile_text='Revenue', tile_label='', tile_value=total_orders,
                        tile_value_prefix='',delta_value=3,integer=True)
        # tile = col_1.container(height=240)
        # tile.subheader('Revenue')
        # # tile.metric(label="", value=f"{total_orders:,}", delta='-1.2%')
        # tile.metric(label="", value=f"${total_revenue_website:,.2f}", delta='9.3%')

        with col_2:
            df_website = df.loc[df['Sales Channel']== 'Mobile App', ['Purchase Date', 'Net Sales']].groupby('Purchase Date').sum()
            df_marketplace = df.loc[df['Sales Channel']== 'Online', ['Purchase Date', 'Net Sales']].groupby('Purchase Date').sum()
            result_sales = pd.merge(df_website, df_marketplace, on='Purchase Date', how='outer')
            result_sales.rename(columns={'Net Sales_x': 'Website_sales', 'Net Sales_y': 'Marketplace_sales'}, inplace=True)
            # chart_data = pd.DataFrame(np.random.randn(20, 2), columns=["a", "b"])
            chart_data = result_sales
            st.line_chart(chart_data)

        kpi_tile(col2_1,tile_text='Orders', tile_label='', tile_value=total_orders_marketplace,
                        tile_value_prefix='',delta_value=1.2,integer=True)
        # tile = col2_1.container(height=240)
        # tile.subheader('Orders')
        # tile.metric(label="", value=f"{total_orders_marketplace:,}", delta='9.3%')

        with col2_2:
            df_website = df.loc[df['Sales Channel']== 'Mobile App', ['Purchase Date', 'Order ID']].groupby('Purchase Date').count()
            df_marketplace = df.loc[df['Sales Channel']== 'Online', ['Purchase Date', 'Order ID']].groupby('Purchase Date').count()
            result_order = pd.merge(df_website, df_marketplace, on='Purchase Date', how='outer')
            result_order.rename(columns={'Order ID_x': 'Website_order', 'Order ID_y': 'Marketplace_order'}, inplace=True)
            # chart_data = pd.DataFrame(np.random.randn(20, 2), columns=["a", "b"])
            chart_data = result_order
            st.bar_chart(chart_data)
            # chart_data = pd.DataFrame(np.random.randn(20, 2), columns=["a", "b"])
            # st.bar_chart(chart_data)

        kpi_tile(col3_1,tile_text='AOV', tile_label='', tile_value=total_orders_marketplace,
                        tile_value_prefix='',delta_value=2.1,integer=True)
        
        # tile = col3_1.container(height=240)
        # tile.subheader('AOV')
        # tile.metric(label="", value=f"${aov_website:,.2f}", delta='9.3%')

        with col3_2:
            result_aov = pd.merge(result_sales, result_order, on='Purchase Date', how='outer')
            result_aov = pd.merge(result_sales, result_order, on='Purchase Date', how='outer')
            result_aov['aov_marketplace'] = result_aov['Marketplace_sales']/result_aov['Marketplace_order']
            result_aov['aov_website'] = result_aov['Website_sales']/result_aov['Website_order']
            chart_data = result_aov[['aov_marketplace', 'aov_website']]
            # chart_data = pd.DataFrame(np.random.randn(20, 2), columns=["a", "b"])
            st.line_chart(chart_data)
    
    with st.container():

        tile = col_4.container(height=240)
        tile.subheader('Revenue')
        # tile.metric(label="", value=f"{total_orders:,}", delta='-1.2%')
        tile.metric(label="", value=f"${total_revenue_marketplace:,.2f}", delta='9.3%')

        tile = col2_4.container(height=240)
        tile.subheader('Orders')
        # tile.metric(label="", value=f"{total_orders:,}", delta='-1.2%')
        tile.metric(label="", value=f"{total_orders_website:,}", delta='9.3%')

        tile = col3_4.container(height=240)
        tile.subheader('AOV')
        # tile.metric(label="", value=f"{total_orders:,}", delta='-1.2%')
        tile.metric(label="", value=f"${aov_marketplace:,.2f}", delta='9.3%')


with tab3:
    with st.container(height=1100):
    
        # fig = px.line(data_frame=df.groupby('Purchase Date')['Net Sales'].sum(),
        #                   title='Total Revenue [Daily]')

        # fig.update(layout_showlegend=False)
        # # fig.update_traces(line=dict(color="Yellow", width=0.4))

        # st.plotly_chart(fig, theme="streamlit", use_container_width=True)
        
        cat_pie = px.pie(data_frame=df.groupby(['Country'])['Net Sales'].sum().abs().reset_index(),
                            # names='Product Category',
                            hole=0.5,
                            color='Country',
                            values='Net Sales',
                            title='Country'
                        )
        st.plotly_chart(cat_pie, theme='streamlit', use_container_width=True)

        col_2, col_3 = st.columns(2)
        with col_2:
            select_country = st.selectbox('Select Country: ', df['Country'].unique().tolist())
            sc_pie = px.pie(data_frame=df.loc[df['Country']==select_country, :].groupby(['State'])['Net Sales'].sum().abs().reset_index(),
                                # names='Product Category',
                               hole=0.5,
                               color='State',
                               values='Net Sales',
                             title='Statewise Sales'
                            )
            st.plotly_chart(sc_pie, theme='streamlit', use_container_width=True)

        with col_3:
            select_state = st.selectbox('Select State: ', df.loc[df['Country']==select_country, :]['State'].unique().tolist())
            sc_pie = px.pie(data_frame=df.loc[df['Country']==select_country, :].loc[df['State']==select_state, :].groupby(['City'])['Net Sales'].sum().abs().reset_index(),
                                # names='Product Category',
                               hole=0.5,
                               color='City',
                               values='Net Sales',
                             title='Citywise Sales'
                            )
            st.plotly_chart(sc_pie, theme='streamlit', use_container_width=True)
    
    # fig = px.line(data_frame=df.groupby('Purchase Date')['Net Sales'].sum() / df.groupby('Purchase Date')['Order ID'].nunique(),
    #                   title='AOV [Daily]')

    # fig.update(layout_showlegend=False)
    # # fig.update_traces(line=dict(color="Yellow", width=0.4))

    # st.plotly_chart(fig, theme="streamlit", use_container_width=True)

# Start Avi calculations ################################################################################################
df = pd.read_csv('fake_ecom_data.csv', encoding='utf-8')
df['Purchase Date'] = pd.to_datetime(df['Purchase Date'])
df['Quarter'] = df['Purchase Date'].dt.to_period('Q').astype(str)
df['Order Group'] = pd.cut(df['Items per Order'], bins=[0, 1, 3, float('inf')], right=False, labels=['1 item', '2-3 items', '4+ items'])


checkout_abandonment_rate = 20
churn_rate = 5
opportunity_lost_revenue = 10000

total_revenue = df['Net Sales'].sum()
total_customers = df['Customer Type'].value_counts().sum()
total_orders = df['Order ID'].nunique()
average_order_value = df['Net Sales'].mean()
aov = total_revenue / total_orders if total_orders else 0
new_customers = df[df['Customer Type'] == 'New'].shape[0]
repeat_customers_count = df[df['Customer Type'] == 'Returning'].shape[0]
repeat_rate = round((repeat_customers_count / total_customers) * 100, 2) if total_customers else 0
customer_retention_rate = (repeat_customers_count / total_orders) * 100
customer_ids = [uuid.uuid4() for _ in range(len(df) // 10)]

# Group the data by quarter and order group, then count the number of orders
quarterly_order_counts = df.groupby(['Quarter', 'Order Group'])['Order ID'].nunique().reset_index()

quarterly_order_counts_pivot = quarterly_order_counts.pivot(index='Quarter', columns='Order Group', values='Order ID').fillna(0)

quarter_order = (df.groupby(['Sales Channel',
                             df['Purchase Date'].dt.to_period('Q')])
                 ['Order ID']
                 .nunique()
                 .rename('Order_count')
                 .reset_index()
                 .assign(Quarter=lambda x: x['Purchase Date'].astype(str)))

quarter_type = (df.groupby(['Customer Type',
                            df['Purchase Date'].dt.to_period('Q')])
                ['Order ID']
                .nunique()
                .rename('Order_count')
                .reset_index()
                .assign(Quarter=lambda x: x['Purchase Date'].astype(str)))


df['YearMonth'] = df['Purchase Date'].dt.to_period('M')

# Calculate the total number of products sold per month
monthly_total_products = df.groupby('YearMonth')['Items per Order'].sum()

# Calculate the number of cross-sell products sold per month (assuming items per order > 1)
monthly_cross_sell_products = df[df['Items per Order'] > 1].groupby('YearMonth')['Items per Order'].sum() - df[df['Items per Order'] > 1].groupby('YearMonth').size()

# Calculate the cross-sell ratio per month
monthly_cross_sell_ratio = (monthly_cross_sell_products / monthly_total_products).fillna(0)
monthly_cross_sell_ratio = monthly_cross_sell_ratio.apply(lambda x: x * np.random.uniform(0.9, 1.1))

# Reset index to use in Plotly
monthly_cross_sell_ratio = monthly_cross_sell_ratio.reset_index()
monthly_cross_sell_ratio.columns = ['YearMonth', 'Cross-Sell Ratio']

# Cross-sell success rate over time (assuming 'Purchase Date' is converted to datetime)
df['Month'] = df['Purchase Date'].dt.to_period('M').astype(str)
cross_sell_success_rate_over_time = df[df['Items per Order'] > 1].groupby('Month').apply(
    lambda x: (x[x['Order Value'] > x['Order Value'].median()].shape[0] / x.shape[0]) * 100
).reset_index(name='Success Rate')


# Placeholder calculations for cross-sell metrics
cross_sell_opportunities = df[df['Items per Order'] > 1].shape[0]
cross_sell_successes = df[(df['Items per Order'] > 1) & (df['Order Value'] > df['Order Value'].median())].shape[0]
cross_sell_success_rate = (cross_sell_successes / cross_sell_opportunities) * 100 if cross_sell_opportunities else 0


# Calculate cross-sell conversion rate
df['Cross_Sell'] = df['Items per Order'] > 1
cross_sell_conversion_rate = df['Cross_Sell'].mean() * 100
cross_sell_rate_over_time = df.groupby('Month')['Cross_Sell'].mean().reset_index()

df['Product Category'] = df['Product Name']  # Replace with actual 'Product Category' if available

avg_categories_per_order = df.groupby('Order ID')['Product Category'].nunique().mean()

top_cross_sold_categories = df[df['Cross_Sell']].groupby('Product Category').size().sort_values(ascending=False).head(5)

top_categories = top_cross_sold_categories.index.tolist()
pairing_matrix = df[df['Product Category'].isin(top_categories)].pivot_table(index='Order ID', columns='Product Category', values='Cross_Sell', aggfunc='max', fill_value=0)


# Calculate Recency as days since last purchase
current_date = datetime.datetime.now()
df['Customer ID'] = [random.choice(customer_ids) for _ in range(len(df))]


recency_df = df.groupby('Customer ID')['Purchase Date'].max().reset_index()
recency_df['Recency'] = (current_date - recency_df['Purchase Date']).dt.days

# Calculate Frequency as count of purchases per customer
frequency_df = df.groupby('Customer ID').size().reset_index(name='Frequency')

# Calculate Monetary as total spend per customer
monetary_df = df.groupby('Customer ID')['Net Sales'].sum().reset_index(name='Monetary')

# Merge recency, frequency, and monetary dataframes
rfm_df = recency_df.merge(frequency_df, on='Customer ID').merge(monetary_df, on='Customer ID')

# Manually define bin edges if necessary
recency_edges = [0, 30, 60, 90, max(rfm_df['Recency'])]
frequency_edges = [0, 2, 5, 10, max(rfm_df['Frequency'])]
monetary_edges = [0, 100, 500, 1000, max(rfm_df['Monetary'])]

# create quantiles for scoring
rfm_df['R_Score'] = pd.qcut(rfm_df['Recency'], 4, ['1','2','3','4'], duplicates='drop')
rfm_df['F_Score'] = pd.qcut(rfm_df['Frequency'], 4, ['4','3','2','1'], duplicates='drop')
rfm_df['M_Score'] = pd.qcut(rfm_df['Monetary'], 4, ['4','3','2','1'], duplicates='drop')



rfm_df['RFM_Segment'] = rfm_df['R_Score'].astype(str) + rfm_df['F_Score'].astype(str) + rfm_df['M_Score'].astype(str)

# map clusters
rfm_df['Customer_Segment'] = np.where(rfm_df['RFM_Segment'] == '111', 'Top',
                                  np.where(rfm_df['RFM_Segment'].str[1] == '1', 'Loyal',
                                          np.where(rfm_df['RFM_Segment'].str[2] == '1', 'Most Monies',
                                                  np.where(rfm_df['RFM_Segment'].isin(['311', '411']), 'Dropped',
                                                          np.where(rfm_df['RFM_Segment'] == '444', 'Waste', 'Nomad')
                                                          )
                                                  )
                                          )
                                 )
avg_rfm_scores = rfm_df[['R_Score', 'F_Score', 'M_Score']].astype(int).mean().reset_index()
avg_rfm_scores.columns = ['Metric', 'Score']

# Convert the scores to a format suitable for a radar chart
avg_rfm_scores_for_radar = avg_rfm_scores.copy()
avg_rfm_scores_for_radar['Customer Segment'] = 'Average'

segment_counts = rfm_df['RFM_Segment'].value_counts().reset_index()
segment_counts.columns = ['RFM_Segment', 'Count']

# End Avi calculations ###########################################################

with tab5:  # Avi

    retention_11, retention_12, retention_13 = st.columns(3)
    retention_21, retention_22, retention_23 = st.columns(3)
    
    
    with retention_11:
        
        tile = retention_11.container(height=300)
        tile.header('Retention %')        
        tile.metric(label="", 
                    value=f"{customer_retention_rate:.2f}%",
                    delta='2.4%'
                   )
        
    with retention_12:
        
        tile = retention_12.container(height=300)
        tile.header('Repeat Customers')
        tile.metric(label="", 
                    value=f"{repeat_customers_count}",
                    delta='-6.8%'                   )

    with retention_13:
        
        tile = retention_13.container(height=300)
        tile.header('Churn Rate')
        tile.metric(label="", 
                    value=f"{churn_rate:.2f}%",
                    delta='4.4%',
                    delta_color='inverse'
                   )
        
    with retention_21:
        
        tile = retention_21.container(height=300)
        tile.header('Checkout Abandonment %')
        tile.metric(label="", 
                    value=f"{checkout_abandonment_rate:.2f}%",
                    delta='2.4%',
                    delta_color='inverse'
                   )
        
    with retention_22:
        
        tile = retention_22.container(height=300)
        tile.header('Opportunity Lost (Revenue)')
        tile.metric(label="", 
                    value=f"{opportunity_lost_revenue}$",
                    delta='5.1%',
                    delta_color='inverse'
                   )

    with retention_23:
        
        tile = retention_23.container(height=300)
        tile.header('Repeat Rate')
        tile.metric(label="", 
                    value=f"{repeat_rate}%",
                    delta='5.1%'
                   )

    with st.container(height=500):

        acq_col_1, acq_col_2 = st.columns([0.7, 0.3])

        with acq_col_1:

            

            fig_acq_qt = px.bar(data_frame=quarter_order,
                                   x='Quarter',
                                   y='Order_count',
                                   color='Sales Channel',
                                   title='QoQ Acquisition by Channel'
                                  )

            st.plotly_chart(fig_acq_qt, use_container_width=True)


        with acq_col_2:

            # Acquisition by channel
            fig_acquisition = px.pie(data_frame=df.groupby('Sales Channel')['Order ID'].nunique().rename('order_count').reset_index(),
                                     values='order_count',
                                     names='Sales Channel',
                                     hole=0.4,
                                     title='Customer Acquisition by Channel'
                                    )
            fig_acquisition.update(layout_showlegend=False)
            st.plotly_chart(fig_acquisition, use_container_width=True)

        
        
    with st.container(height=500):
        
        acq_col_3, acq_col_4 = st.columns([0.7, 0.3])
        
        with acq_col_3:
            
            

            fig_returning = px.bar(data_frame=quarter_type,
                                       x='Quarter',
                                       y='Order_count',
                                       color='Customer Type',
                                       title='QoQ Returning vs New Customers'
                                      )
            st.plotly_chart(fig_returning, use_container_width=True)
            
        with acq_col_4:
            
            fig_ct_pie = px.pie(data_frame=df.groupby('Customer Type')['Order ID'].nunique().rename('order_count').reset_index(),
                                     values='order_count',
                                     # names='Sales Channel',
                                     names='Customer Type',
                                     hole=0.4,
                                     title='Returning vs New Customers Overall'
                                    )
            fig_ct_pie.update(layout_showlegend=False)
            st.plotly_chart(fig_ct_pie, use_container_width=True)
            
    
    rfm_col_1, rfm_col_2 = st.columns([0.6, 0.4])
    rfm_col_3, rfm_col_4 = st.columns([0.5, 0.6])

    
    with rfm_col_1:
        
        # Distribution of R, F, and M scores
        fig_rfm_distribution = px.histogram(rfm_df, x=['R_Score', 'F_Score', 'M_Score'], barmode='overlay', title='RFM Score Distribution')
        st.plotly_chart(fig_rfm_distribution, use_container_width=True)

    with rfm_col_2:
        
        heatmap_data = rfm_df.pivot_table(index='F_Score', columns='R_Score', values='Monetary', aggfunc='mean').fillna(0)
        fig_heatmap = px.imshow(heatmap_data, aspect='auto', color_continuous_scale='viridis', title='RF Heatmap')
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
    with rfm_col_3:
        
        fig_rfm_scatter = px.scatter_3d(rfm_df, x='Recency', y='Frequency', z='Monetary', color='Customer_Segment', title='RFM Scores')
        st.plotly_chart(fig_rfm_scatter, use_container_width=True)
        
    with rfm_col_4:
        
        
        fig = px.pie(data_frame=rfm_df.Customer_Segment.value_counts().reset_index(),
                     values='count',
                     names='Customer_Segment',
                     hole=0.4,
                     title='RFM Segments Overall')
                     # category_orders={'Order Group': ['1 item', '2-3 items', '4+ items']}
                    # )
        fig.update(layout_showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
    
with tab6:  # Avi

    cs_col_11, cs_col_12, cs_col_13 = st.columns(3)
    cs_col_21, cs_col_22 = st.columns(2)    
    
    with cs_col_11:
        
        tile = cs_col_11.container(height=300)
        tile.header('Cross-Sell Opportunities')        
        tile.metric(label="", 
                    value=f"{cross_sell_opportunities}",
                    delta='9.4%'
                   )
        
    with cs_col_12:
        
        tile = cs_col_12.container(height=300)
        tile.header('Cross-Sell Successes')
        tile.metric(label="", 
                    value=f"{cross_sell_successes}",
                    delta='3.8%'                   )

    with cs_col_13:
        
        tile = cs_col_13.container(height=300)
        tile.header('Cross-Sell Success Rate')
        tile.metric(label="", 
                    value=f"{cross_sell_success_rate:.2f}%",
                    delta='4.4%'
                    # delta_color='inverse'
                   )  

    with cs_col_21:
        
        tile = cs_col_21.container(height=300)
        tile.header('Cross-Sell Conversion Rate')        
        tile.metric(label="", 
                    value=f"{cross_sell_conversion_rate}%",
                    delta='1.4%'
                   )
        
    with cs_col_22:
        
        tile = cs_col_22.container(height=300)
        tile.header('Average Number of Categories per Order')
        tile.metric(label="", 
                    value=f"{avg_categories_per_order}",
                    delta='2.8%'                   )

        
        
    with st.container(height=500):
        
        cs_col_1, cs_col_2 = st.columns([0.7, 0.3])

        with cs_col_1:


            # Create a stacked bar chart
            fig = px.bar(quarterly_order_counts_pivot, title='QoQ Cross-Sell Segments',
                         labels={'value': 'Number of Orders', 'Quarter': 'Quarter'},
                         category_orders={'Order Group': ['1 item', '2-3 items', '4+ items']})

            # Update the layout to stack the bars
            fig.update_layout(barmode='stack')

            # Display the chart in the Streamlit app
            st.plotly_chart(fig, use_container_width=True)

        with cs_col_2:

            fig = px.pie(data_frame=df.groupby('Order Group')['Order ID'].nunique().rename('order_count').reset_index(),
                                     values='order_count',
                                     # names='Sales Channel',
                                     names='Order Group',
                                     hole=0.4,
                                     title='Cross-Sell Segments Overall',
                                     category_orders={'Order Group': ['1 item', '2-3 items', '4+ items']}
                                    )
            fig.update(layout_showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
    with st.container(height=500):
        
        cs_col_3, cs_col_4 = st.columns([0.7, 0.3])
        
        with cs_col_3:

            fig_cross_sell = px.line(monthly_cross_sell_ratio.assign(PurchaseDate=lambda x: x['YearMonth'].astype(str)),
                                     x='PurchaseDate',
                                     y='Cross-Sell Ratio',
                                     title='Month-on-Month Cross-Sell Ratio')
            fig_cross_sell.update_layout(xaxis_title='Month', yaxis_title='Cross-Sell Ratio', yaxis=dict(tickformat=".2%"))
            st.plotly_chart(fig_cross_sell, use_container_width=True)
            
        with cs_col_4:
            
            fig = px.pie(data_frame=df[df['Cross_Sell']].groupby('Product Category').size().sort_values().rename('count').reset_index().tail(20),
                                     values='count',
                                     # names='Sales Channel',
                                     names='Product Category',
                                     hole=0.4,
                                     title='Cross-Sold Product Categories Overall',
                                     # category_orders={'Order Group': ['1 item', '2-3 items', '4+ items']}
                                    )
            # fig.update(layout_showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
    
    
    
    fig_success_rate = px.line(cross_sell_success_rate_over_time, x='Month', y='Success Rate', title='Cross-Sell Success Rate Over Time')
    st.plotly_chart(fig_success_rate, use_container_width=True)


#     # Distribution of items per order (as a proxy for cross-sell opportunities)
#     fig_items_per_order = px.histogram(df, x='Items per Order', title='Distribution of Items per Order')
#     st.plotly_chart(fig_items_per_order, use_container_width=True)
    
    fig_conversion_rate = px.line(cross_sell_rate_over_time, x='Month', y='Cross_Sell', title='Cross-Sell Conversion Rate Over Time')
    st.plotly_chart(fig_conversion_rate, use_container_width=True)


# Start Deepak calculations ########################################

ecom_df = pd.read_csv('fake_ecom3.csv',encoding='utf-8')
ecom_product_agg = ecom_df.groupby(['Product Name']).agg({'Order ID':'nunique', 'Gross Sales':'sum'}).reset_index()

ecom_daily_product_agg = ecom_df.groupby(['Purchase Date','Product Name']).agg({'Order ID':'nunique', 'Gross Sales':'sum'}).reset_index()
ecom_daily_product_agg['Purchase Date'] = pd.to_datetime(ecom_daily_product_agg['Purchase Date'])
ecom_daily_product_agg['Month'] = ecom_daily_product_agg['Purchase Date'].dt.strftime('%Y-%m')
ecom_daily_product_agg = ecom_daily_product_agg.sort_values(by=['Month'],ascending=True).reset_index(drop=True)

product_sales = ecom_daily_product_agg.groupby('Product Name')['Gross Sales'].sum().reset_index()
# Sort products based on total sales in descending order
product_sales_sorted = product_sales.sort_values(by='Gross Sales', ascending=False).reset_index()
product_sales_sorted['percentange'] = (product_sales_sorted['Gross Sales']/product_sales_sorted['Gross Sales'].sum())*100
product_sales_sorted = product_sales_sorted.assign(cumsum_per=product_sales_sorted['percentange'].cumsum())
top_products = product_sales_sorted[product_sales_sorted['cumsum_per']>=70]['Product Name'].unique()
worst_products = product_sales_sorted[product_sales_sorted['cumsum_per']<70]['Product Name'].unique()

ecom_monthly_product_agg = ecom_daily_product_agg.groupby(['Month','Product Name']).agg({'Order ID':'nunique', 'Gross Sales':'sum'}).reset_index()
ecom_monthly_product_agg = ecom_monthly_product_agg.sort_values(by=['Month'],ascending=True).reset_index(drop=True)



def create_heatmap(data):
    fig = go.Figure(data=go.Heatmap(
        z=data.values,
        x=data.columns,
        y=data.index,
        colorscale='Viridis',
    ))
    fig.update_layout(
        autosize=True,
        margin=dict(t=50, l=50, r=50, b=50),
    )
    return fig

########### Deepak#############
# read ad performance data
ads_data = pd.read_csv('fake_ads_data.csv', encoding='utf-8')
ads_data['date'] = pd.to_datetime(ads_data['date'])
ads_data['Month'] = ads_data['date'].dt.strftime('%Y-%m')
ads_data = ads_data[ads_data['date']>='2023-01-01']
ads_data.sort_values(by=['Month'], inplace=True) 

ads_agg = ads_data.groupby(['Month']).agg({'impressions':'sum', 'ad_spend':'sum'}).reset_index()
ads_agg.sort_values(by=['Month'], inplace=True) 

vendor_df = ads_data.groupby(['Month','vendor']).agg({'impressions':'sum', 'ad_spend':'sum', 'CTR':'mean', 'ROAS':'mean'}).reset_index()
channel_df = ads_data.groupby(['Month','marketing_channel']).agg({'impressions':'sum', 'ad_spend':'sum', 'CTR':'mean', 'ROAS':'mean'}).reset_index()

campaign = ads_data.groupby(['campaigns']).agg({'impressions':'sum', 'ad_spend':'sum', 'clicks':'sum', 'CTR':'mean', 'ROAS':'max'}).reset_index()

top_campaign = campaign.sort_values(by=['impressions'], ascending=False)[:10]

ecom_df = pd.read_csv('fake_ecom3.csv', encoding='utf-8')
ecom_product_agg = ecom_df.groupby(['Product Name']).agg({'Order ID':'nunique', 'Gross Sales':'sum'}).reset_index()

ecom_daily_product_agg = ecom_df.groupby(['Purchase Date','Product Name']).agg({'Order ID':'nunique', 'Gross Sales':'sum'}).reset_index()
ecom_daily_product_agg['Purchase Date'] = pd.to_datetime(ecom_daily_product_agg['Purchase Date'])
ecom_daily_product_agg['Month'] = ecom_daily_product_agg['Purchase Date'].dt.strftime('%Y-%m')
ecom_daily_product_agg = ecom_daily_product_agg.sort_values(by=['Month'],ascending=True).reset_index(drop=True)

product_sales = ecom_daily_product_agg.groupby('Product Name')['Gross Sales'].sum().reset_index()
# Sort products based on total sales in descending order
product_sales_sorted = product_sales.sort_values(by='Gross Sales', ascending=False).reset_index()
product_sales_sorted['percentange'] = (product_sales_sorted['Gross Sales']/product_sales_sorted['Gross Sales'].sum())*100
product_sales_sorted = product_sales_sorted.assign(cumsum_per=product_sales_sorted['percentange'].cumsum())
top_products = product_sales_sorted[product_sales_sorted['cumsum_per']>=70]['Product Name'].unique()
worst_products = product_sales_sorted[product_sales_sorted['cumsum_per']<70]['Product Name'].unique()

ecom_monthly_product_agg = ecom_daily_product_agg.groupby(['Month','Product Name']).agg({'Order ID':'nunique', 'Gross Sales':'sum'}).reset_index()
ecom_monthly_product_agg = ecom_monthly_product_agg.sort_values(by=['Month'],ascending=True).reset_index(drop=True)

# End Deepak calculations ##############################################################

with tab4: #Deepak

    # empty, empty,empty, options = st.columns(4)
    # with options:
    #     option = st.selectbox(
    #     '',
    #     ('Last 7 Days', 'Last 15 Days', 'Last 30 Days', 'This Month', 'This Quarter', 'This Year', 'All Data', 'Custom Range'),
    #     key='tab4')
    #     df = generate_data()
    #     if option == 'Last 7 Days': 
    #         end_date = df.iloc[df['Purchase Date'].idxmax()]['Purchase Date']
    #         start_date = end_date - pd.Timedelta(days=7)
    #         df = df[(df['Purchase Date'] >= start_date) & (df['Purchase Date'] <= end_date)]
    #     elif option == 'Last 15 Days':
    #         end_date = df.iloc[df['Purchase Date'].idxmax()]['Purchase Date']
    #         start_date = end_date - pd.Timedelta(days=15)
    #         df = df[(df['Purchase Date'] >= start_date) & (df['Purchase Date'] <= end_date)]
    #     else:
    #         pass

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    with kpi1:
        tile = kpi1.container(height=240)
        tile.header('Add to Cart rate')
        tile.metric(label="", value=f"{10:,}", delta='-1.2%')

    with kpi2:
        tile = kpi2.container(height=240)
        tile.header('Time spent on page')
        tile.metric(label="", value=f"{'3 minutes'}", delta='9.3%')
        
    with kpi3:
        tile = kpi3.container(height=240)
        tile.header('Inventory turnover')
        tile.metric(label="", value=f"{'7 turns'}", delta='11%')

    with kpi4:
        tile = kpi4.container(height=240)
        tile.header('Cross sell active')
        tile.metric(label="", value=f"{'25'}%", delta='-11%')

    with st.container(height=450):
        col_1, col_2 = st.columns(2)
        with col_1:
            cat_pie = px.pie(data_frame=ecom_df.groupby(['Product Name'])['Order ID'].nunique().abs().reset_index(),
                               hole=0.5,
                               color='Product Name',
                               values='Order ID',
                             title='Transaction by Products'
                            )
            st.plotly_chart(cat_pie, theme='streamlit', use_container_width=True)

        with col_2:
            cat_pie = px.pie(data_frame=ecom_df.groupby(['Product Name'])['Gross Sales'].nunique().abs().reset_index(),
                               hole=0.5,
                               color='Product Name',
                               values='Gross Sales',
                             title='Revenue by Products'
                            )
            st.plotly_chart(cat_pie, theme='streamlit', use_container_width=True)


    with st.container(height=1300):

        st.write("Product revenue trend")
        chart = alt.Chart(ecom_monthly_product_agg).mark_bar(
            cornerRadiusTopLeft=3,
            cornerRadiusTopRight=3
        ).encode(
            x='Month',
            y='Gross Sales',
            color='Product Name'
        )
        st.altair_chart(chart, theme="streamlit", use_container_width=True)

        st.write("Best performing products")
        temp = ecom_monthly_product_agg[ecom_monthly_product_agg['Product Name'].isin(top_products)]
        chart = alt.Chart(temp).mark_line().encode(
        x='Month',
        y='Gross Sales',
        color='Product Name',
        strokeDash='Product Name',
    )
        st.altair_chart(chart, theme="streamlit", use_container_width=True)

        
        st.write("Worst selling products")
        temp = ecom_monthly_product_agg[ecom_monthly_product_agg['Product Name'].isin(worst_products)]
        chart = alt.Chart(temp).mark_line().encode(
        x='Month',
        y='Gross Sales',
        color='Product Name',
        strokeDash='Product Name',
    )
        st.altair_chart(chart, theme="streamlit", use_container_width=True)

    
    with st.container(height=600):
        st.write('Product Affinity Analysis')
        # Define the list of products
        products = ['Face Cream', 'Dress', 'Shampoo', 'Laptop', 'Knife Set', 'Smartphone', 'Sneakers', 'Jeans', 'Blender']
        # Simulate random affinity matrix data for the products
        np.random.seed(42)
        affinity_matrix = pd.DataFrame(np.random.rand(len(products), len(products)), columns=products, index=products)
        # Set a threshold for strong relationships (adjust as needed)
        threshold = 0.7
        # Filter the affinity matrix to show only strong relationships
        strong_relationships = affinity_matrix[affinity_matrix > threshold]
        # Create a heatmap with products on both axes for strong relationships
        fig = px.imshow(strong_relationships, x=strong_relationships.columns, y=strong_relationships.index, color_continuous_scale='purples')
        fig.update_layout(width=1200, height=500)  # Adjust width and height as needed
        st.plotly_chart(fig, use_container_width=True)

with tab7: #Deepak

    # empty, empty,empty, options = st.columns(4)
    # with options:
    #     option = st.selectbox(
    #     '',
    #     ('Last 7 Days', 'Last 15 Days', 'Last 30 Days', 'This Month', 'This Quarter', 'This Year', 'All Data', 'Custom Range'),
    #     key='tab6')
    #     df = generate_data()
    #     if option == 'Last 7 Days': 
    #         end_date = df.iloc[df['Purchase Date'].idxmax()]['Purchase Date']
    #         start_date = end_date - pd.Timedelta(days=7)
    #         df = df[(df['Purchase Date'] >= start_date) & (df['Purchase Date'] <= end_date)]
    #     elif option == 'Last 15 Days':
    #         end_date = df.iloc[df['Purchase Date'].idxmax()]['Purchase Date']
    #         start_date = end_date - pd.Timedelta(days=15)
    #         df = df[(df['Purchase Date'] >= start_date) & (df['Purchase Date'] <= end_date)]
    #     else:
    #         pass

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi5, kpi6, kpi7, kpi8 = st.columns(4)

    impressions = 18972
    ad_spends = 5677
    aov = 1100
    cac = 789
    ctr = 4
    cpm = 3.5
    active_campaigns = 36
    roas = 4

    with kpi1:
        tile = kpi1.container(height=240)
        tile.header('Impressions')
        tile.metric(label="", value=f"{impressions:,}", delta='-1.2%')

    with kpi2:
        tile = kpi2.container(height=240)
        tile.header('Ad Spends')
        tile.metric(label="", value=f"${ad_spends:,.2f}", delta='9.3%')
        
    with kpi3:
        tile = kpi3.container(height=240)
        tile.header('Ad Revenue')
        tile.metric(label="", value=f"${aov:,.2f}", delta='11%')

    with kpi4:
        tile = kpi4.container(height=240)
        tile.header('CAC')
        tile.metric(label="", value=f"${cac:,.2f}", delta='11%')

    with kpi5:
        tile = kpi5.container(height=240)
        tile.header('CTR')
        tile.metric(label="", value=f"{ctr:,.2f}", delta='11%')

    with kpi6:
        tile = kpi6.container(height=240)
        tile.header('CPM')
        tile.metric(label="", value=f"{cpm:,.2f}", delta='11%')

    with kpi7:
        tile = kpi7.container(height=240)
        tile.header('Active campaigns')
        tile.metric(label="", value=f"{active_campaigns:,.2f}", delta='11%')

    with kpi8:
        tile = kpi8.container(height=240)
        tile.header('ROAS')
        tile.metric(label="", value=f"{roas:,.2f}%", delta='11%')
    
    with st.container(height=1000):
        # Selectbox with two options
        option = st.selectbox('Select Performance by:', ['Vendor', 'Channel'])

        if option == 'Vendor':
            col_1, col_2 = st.columns(2)
            col_3, col_4 = st.columns(2)

            with col_1:
                st.write('Impressions by Vendor')
                chart = alt.Chart(vendor_df).mark_bar(
                cornerRadiusTopLeft=3,
                cornerRadiusTopRight=3
            ).encode(
                x='Month',
                y='impressions',
                color='vendor'
            )
                st.altair_chart(chart, theme="streamlit", use_container_width=True)
    
            with col_2:
                st.write('Spends by Vendor')
                chart = alt.Chart(vendor_df).mark_bar(
                cornerRadiusTopLeft=3,
                cornerRadiusTopRight=3
            ).encode(
                x='Month',
                y='ad_spend',
                color='vendor'
            )
                st.altair_chart(chart, theme=None, use_container_width=True)

            with col_3:
                st.write('CTR by Vendor')
                chart = alt.Chart(vendor_df).mark_bar(
                cornerRadiusTopLeft=3,
                cornerRadiusTopRight=3
            ).encode(
                x='Month',
                y='CTR',
                color='vendor'
            )
                st.altair_chart(chart, theme="streamlit", use_container_width=True)

            with col_4:
                st.write('ROI by Vendor')
                chart = alt.Chart(vendor_df).mark_bar(
                cornerRadiusTopLeft=3,
                cornerRadiusTopRight=3
            ).encode(
                x='Month',
                y='ROAS',
                color='vendor'
            )
                st.altair_chart(chart, theme=None, use_container_width=True)

        elif option == 'Channel':
            col_1, col_2 = st.columns(2)
            col_3, col_4 = st.columns(2)

            with col_1:
                st.write('Impressions by Channel')
                chart = alt.Chart(channel_df).mark_bar(
                cornerRadiusTopLeft=3,
                cornerRadiusTopRight=3
            ).encode(
                x='Month',
                y='impressions',
                color='marketing_channel'
            )
                st.altair_chart(chart, theme="streamlit", use_container_width=True)
    
            with col_2:
                st.write('Spends by Channel')
                chart = alt.Chart(channel_df).mark_bar(
                cornerRadiusTopLeft=3,
                cornerRadiusTopRight=3
            ).encode(
                x='Month',
                y='ad_spend',
                color='marketing_channel'
            )
                st.altair_chart(chart, theme=None, use_container_width=True)

            with col_3:
                st.write('CTR by Channel')
                chart = alt.Chart(channel_df).mark_bar(
                cornerRadiusTopLeft=3,
                cornerRadiusTopRight=3
            ).encode(
                x='Month',
                y='CTR',
                color='marketing_channel'
            )
                st.altair_chart(chart, theme="streamlit", use_container_width=True)

            with col_4:
                st.write('ROI by Channel')
                chart = alt.Chart(channel_df).mark_bar(
                cornerRadiusTopLeft=3,
                cornerRadiusTopRight=3
            ).encode(
                x='Month',
                y='ROAS',
                color='marketing_channel'
            )
                st.altair_chart(chart, theme=None, use_container_width=True)

    st.write('Top 10 Campaigns')
    with st.container(height=1300):

        st.write('Impressions')
        chart = alt.Chart(top_campaign).mark_bar().encode(
        x='impressions',
        y='campaigns',
        color='campaigns'
    )
        st.altair_chart(chart, theme=None, use_container_width=True)

        st.write('ROAS')
        mean_roas = campaign['ROAS'].mean()
        fig = px.bar(top_campaign, x='ROAS', y='campaigns', orientation='h')
        fig.add_shape(type='line',
        x0=mean_roas, y0=-0.5,
        x1=mean_roas, y1=len(top_campaign)-0.5,
        line=dict(color='red', width=5))
        st.plotly_chart(fig)

        st.write('Campaign metrics')
        temp = top_campaign.reset_index(drop=True)
        temp.columns = [col.title() for col in temp.columns]
        st.table(temp)