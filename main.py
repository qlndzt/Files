import streamlit as st
import pandas as pd
import altair as alt
import pydeck as pdk

DATA = None
st.set_page_config(
    page_title="HDB Resale & Rental",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded")
alt.themes.enable("dark")



def createYM(df):
    c = 'month' if df.equals(resale_df) else 'rent_approval_date'
    df["year"] = df[c].str.split("-").str[0].astype(int)
    df['month'] = df[c].str.split("-").str[1].astype(int)


# read resale data frame and create year column
resale_df = pd.read_csv('ResaleflatpricesbasedonregistrationdatefromJan2017onwards.csv')
rental_df = pd.read_csv('RentingOutofFlats.csv')
createYM(resale_df)
createYM(rental_df)
hdb_locations = pd.read_csv("town_coordinates.csv")


def chooseDf(selected):
    global DATA
    if selected == 'Resale':
        DATA = 'Resale'
        return resale_df
    DATA = 'Rental'
    return rental_df


# sidebar that is able to filter years and two dataframes
with st.sidebar:
    st.title('🏠 HDB Resale & Rental')
    data_type = ['Resale', 'Rental']
    curr_df = chooseDf(st.selectbox('Select a dataframe', data_type, index=len(data_type) - 1))
    year_list = list(curr_df.year.unique())[::-1]
    selected_year = st.selectbox('Select a year', year_list, index=len(year_list) - 1)
    df_selected_year = curr_df[curr_df.year == selected_year]
    df_selected_year_sorted = df_selected_year.sort_values(by="town", ascending=False)


transaction_counts = df_selected_year['town'].value_counts().reset_index()
transaction_counts.columns = ['town', 'transactions']
max_transactions = transaction_counts['transactions'].max()
transaction_counts['is_max'] = transaction_counts['transactions'] == max_transactions


# calculating the AVERAGE PRICE
def average_price():
    data = []
    town = curr_df.drop_duplicates(subset='town')
    for i in town['town']:
        average = df_selected_year_sorted.loc[df_selected_year['town'] == str(i)]
        mean = round(average['resale_price'].mean() if curr_df.equals(resale_df) else average['monthly_rent'].mean(), 1)
        final = [mean, i]
        data.append(final)
    return data


# creating a df with lat, lng, and average values of each town
map_df = average_price()
map_df = pd.DataFrame(map_df)
map_df = map_df.rename(columns={0: 'average_price', 1: 'town'})
map_df = pd.merge(map_df, hdb_locations, on='town')
map_df.columns = map_df.columns.str.strip()
map_df['coordinates'] = map_df[['lat', 'lon']].apply(list, axis=1)


# create a map that is able to display town name and average price of each given coordinates in df
def drawMap(df):
    tooltip = {
        "html": "<b>town:</b> {town}<br><b>Average Price:</b> {average_price}",  # Display the aggregated average price
        "style": {
            "backgroundColor": "steelblue",
            "color": "white"
        }
    }
    st.pydeck_chart(pdk.Deck(
        map_style=None,
        initial_view_state=pdk.ViewState(
            latitude=1.285981,
            longitude=103.852986,
            zoom=10.5,
            pitch=50,
        ),
        layers=[pdk.Layer(
            'ScatterplotLayer',
            data=df,
            get_position='[lon, lat]',
            get_color="[220, 30, 20, 160]",  # Set the color of the circles (RGBA)
            get_radius='average_price',  # Set the radius of the circles based on the average price
            pickable=True,
            radius_scale=20,  # Adjust the scale of the radius if needed
            radius_min_pixels=1,  # Minimum radius of the circles in pixels
            radius_max_pixels=20,  # Maximum radius of the circles in pixels
            tooltip=tooltip
        )], tooltip=tooltip
    ))


def drawMonthlyGraph(df, text):
    if DATA == 'Rental':
        c = 'monthly_rent'
    else:
        c = 'resale_price'
    seasonal_effect = df.groupby('month')[c].mean().reset_index()
    graph = alt.Chart(seasonal_effect).mark_line(point=True).encode(
        x=alt.X('month:O', title='Month', axis=alt.Axis(labelAngle=0)),
        y=alt.Y(f'{c}:Q', title=f'Average {text}').scale(zero=False),
        tooltip=['month', c]
    ).properties(
        width=600,
        height=400,
        title=f'Seasonal Effect on {text}'
    )
    return graph


def drawAnnualGraph(df, text):
    if DATA == 'Rental':
        c = 'monthly_rent'
    else:
        c = 'resale_price'
    average_year_price = df.groupby(['town', 'year'])[c].mean().reset_index()
    graph = alt.Chart(average_year_price).mark_line().encode(
        x=alt.X('year:O', axis=alt.Axis(title='Year')),
        y=alt.Y(f'{c}:Q', axis=alt.Axis(title=f'Average {text}')),
        color=alt.Color('town:N', legend=alt.Legend(title="Town")),
        tooltip=['town', 'year', c]
    ).properties(
        height=550,
        title=f'Trend of the {text} for Each Town Across Different Years'
    ).interactive()
    return graph


def drawMetric(df, v, i):
    value = f'${v.iloc[i]}' if df.equals(map_df_sorted) else v.iloc[i]
    st.markdown(f"""
            <style>
            .metric {{
                font-size: 16px;
                font-weight: bold;
                color: #fff; 
                background-color: #333; 
                padding: 10px; 
                border-radius: 5px;
                text-align: center;
            }}
            .metric-value {{
                font-size: 40px;
                font-weight: bold;
                color: #fff;
            }}
            .metric-label {{
                margin-bottom: 1px;  /* Reduce the bottom margin of the label */
            }}
            </style>
            <div class="metric">
                <div class="metric-label">{df.town.iloc[i]}</div>
                <div class="metric-value">{value}</div>
            </div>
            """, unsafe_allow_html=True)


def drawCountGraph(df):
    graph = alt.Chart(df).mark_bar().encode(
        x=alt.X('transactions:Q', title='Number of Transactions'),
        y=alt.Y('town:N', sort='-x', title='Town'),
        color=alt.condition(
            alt.datum.is_max,
            alt.value('orange'),
            alt.value('steelblue')
        ),
        tooltip=['town', 'transactions']
    ).properties(
        width=350,
        title='Popularity of Towns Based on Transactions'
    )
    return graph


# slice the page into 3 columns
alt.data_transformers.enable('default', max_rows=None)
col = st.columns((1.5, 4.5, 2), gap='medium')
t = 'Resale Price' if DATA == 'Resale' else 'Monthly Rent'

with col[0]:
    st.write(f'Max Average {t}:')
    map_df_sorted = map_df.sort_values("average_price")
    drawMetric(map_df_sorted, map_df_sorted.average_price, -1)

    st.write('\n')
    st.write(f'Min Average {t}:')
    drawMetric(map_df_sorted, map_df_sorted.average_price, 0)

    st.write('\n')
    st.write('Max Transaction amount:')
    transaction_counts_sorted = transaction_counts.sort_values("transactions")
    drawMetric(transaction_counts_sorted, transaction_counts_sorted.transactions, -1)

    st.write('\n')
    st.write('Min Transaction amount:')
    drawMetric(transaction_counts_sorted, transaction_counts_sorted.transactions, 0)

    st.altair_chart(drawMonthlyGraph(df_selected_year_sorted, t), use_container_width=True)

with col[1]:
    st.markdown(f'#### Average {t} Map')
    drawMap(map_df)
    st.altair_chart(drawAnnualGraph(curr_df, t), use_container_width=True)

with col[2]:
    st.markdown('#### Top Town')
    st.dataframe(map_df.sort_values("average_price", ascending=False),
                 column_order=("town", "average_price"),
                 hide_index=True,
                 width=None,
                 height=500,
                 column_config={
                     "town": st.column_config.TextColumn(
                         "Town",
                     ),
                     "average_price": st.column_config.ProgressColumn(
                         t,
                         format="%f",
                         min_value=0,
                         max_value=max(map_df.average_price),
                     )}
                 )

    st.altair_chart(drawCountGraph(transaction_counts))
