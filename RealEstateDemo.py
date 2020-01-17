import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd

import plotly.figure_factory as ff

import plotly.graph_objs as go
from plotly.subplots import make_subplots
from datetime import datetime

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import re
from pandas.io.json import json_normalize
import json
#from sklearn.preprocessing import MinMaxScaler,StandardScaler

#parsing
import cfscrape
from lxml import etree
#graphs
import colorlover as cl
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode, iplot
from plotly.subplots import make_subplots

st.title('The hottest zipcodes and demography statistics')

#hottest market by Realtor
url_zip = 'https://www.realtor.com/research/hottest-zip-codes-2019/'
scraper = cfscrape.create_scraper()
scraped_html=scraper.get(url_zip).content
html = etree.HTML(scraped_html)

hot_zip = {}
for i in range(1,11):
    zipcode = html.xpath("//*[@id='post-2223']/div/table[1]/tbody/tr")[i].xpath("td[2]/a/text()")
    name = html.xpath("//*[@id='post-2223']/div/table[1]/tbody/tr")[i].xpath("td[3]/a/text()")
    hot_zip[zipcode[0]] = name[0]
hot_zip = pd.DataFrame(list(hot_zip.items()), columns=['ZIPcode', 'City, State'])
hot_zip['Rank'] = np.arange(1, 11)
hot_zip['ZIPcode'] = hot_zip['ZIPcode'].astype(int)



st.write("Despite of the fact that zipcodes don't realy correspond with real boundaries of market processes "
         "they are still used as a benchmark for market analysis. Here a ranking calculated by Realtor in order to"
         "mark the hottest area with the most active real estate market.")
st.markdown('Hottest zipcodes according to Realtor Ranking')
st.write(hot_zip)

from uszipcode import SearchEngine

zip_data = pd.DataFrame()
for i in hot_zip['ZIPcode'].unique():
    search = SearchEngine(simple_zipcode=False)
    zipcode = search.by_zipcode(i)
    table = json_normalize(json.loads(zipcode.to_json()))
    zip_data = pd.concat([zip_data, table], axis=0)
zip_data = zip_data.reset_index(drop=True)

for i in range(0, hot_zip.shape[0]):
    if zip_data['zipcode'].unique()[i] != hot_zip['ZIPcode'].unique().astype(str)[i]:
        hot_zip.loc[i:i, 'ZIPcode'] = zip_data['zipcode'].unique()[i]

to_parse = ['sources_of_household_income____average_income_per_household_by_income_source',
            'head_of_household_by_age',
            'families_vs_singles',
            'employment_status',
            'educational_attainment_for_population_25_and_over',
            'children_by_age',
            'household_investment_income____average_income_per_household_by_income_source',
            'households_with_kids',
            'housing_occupancy',
            'means_of_transportation_to_work_for_workers_16_and_over',
            'travel_time_to_work_in_minutes',
            'population_by_age',
            'population_by_gender',
            'population_by_race',
            'rental_properties_by_number_of_rooms',
            'school_enrollment_age_3_to_17',
            'vancancy_reason',
            'annual_individual_earnings',
            'household_income',
            'owner_occupied_home_values',
            'monthly_rent_including_utilities_1_b',
            'monthly_rent_including_utilities_2_b',
            'monthly_rent_including_utilities_3plus_b',
            'monthly_rent_including_utilities_studio_apt']
col_names = ['income_source',
             'house_hold_head_age',
             'family',
             'employment',
             'education',
             'children_age',
             'investment_income',
             'kids',
             'occupancy',
             'commute type',
             'commute time',
             'age',
             'gender',
             'race',
             'rental prop',
             'school enrlolment',
             'vacancy',
             'earnings',
             'household_income',
             'owner_home_values',
             'rent_1b',
             'rent_2b',
             'rent_3b',
             'rent_studio']
def dem_transform(df):
    data = pd.DataFrame()
    for n in range(0, df.shape[0]):
        stat_row = pd.DataFrame()
        for i, j in enumerate(to_parse):
            if j in ['annual_individual_earnings',
                     'household_income',
                     'owner_occupied_home_values']:
                stat = pd.DataFrame(df[j][n][0]['values'])
                stat['x'] = stat['x'].str.replace(r'[^-\d]', '').str.replace(r'(000$)|(999$)', 'K').str.replace(
                    r'(000-)', 'K-')
                stat['x'] = col_names[i] + ': ' + stat['x'].astype(str)
                stat = stat.set_index('x')
                stat = stat.T
                stat_row = pd.concat([stat_row, stat], axis=1)
            elif j in ['monthly_rent_including_utilities_1_b',
                       'monthly_rent_including_utilities_2_b',
                       'monthly_rent_including_utilities_3plus_b',
                       'monthly_rent_including_utilities_studio_apt',
                       'travel_time_to_work_in_minutes']:
                stat = pd.DataFrame(df[j][n][0]['values'])
                stat['x'] = stat['x'].str.replace(r'[^-\d]', '')
                stat['x'] = col_names[i] + ': ' + stat['x'].astype(str)
                stat = stat.set_index('x')
                stat = stat.T
                stat_row = pd.concat([stat_row, stat], axis=1)
            else:
                stat = pd.DataFrame(df[j][n][0]['values'])
                stat['x'] = col_names[i] + ': ' + stat['x'].astype(str)
                stat = stat.set_index('x')
                stat = stat.T
                stat_row = pd.concat([stat_row, stat], axis=1)

        single_col = df[['zipcode', 'county', 'lat', 'lng', 'major_city', 'median_home_value', 'population',
                         'median_household_income', 'occupied_housing_units', 'state', 'polygon']].loc[n:n,
                     :].reset_index(drop=True)
        stat_row = pd.concat([stat_row.reset_index(drop=True), single_col], axis=1)

        data = pd.concat([data, stat_row])
    data = data.reset_index(drop=True)

    data['Age: under 10'] = data['age: Under 5'] + data['age: 5-9']
    data['Age: 10-20'] = data['age: 10-14'] + data['age: 15-19']
    data['Age: 20-35'] = data['age: 20-24'] + data['age: 25-29'] + data['age: 30-34']
    data['Age: 35-50'] = data['age: 35-39'] + data['age: 40-44'] + data['age: 45-49']
    data['Age: 50-65'] = data['age: 50-54'] + data['age: 55-59'] + data['age: 60-64']
    data['Age: 65-80'] = data['age: 65-69'] + data['age: 70-74'] + data['age: 75-79']
    data['Age: over 80'] = data['age: 80-84'] + data['age: 85 Plus']
    data.drop(columns=[x for x in data.columns if re.search(r'(^age:)', x) != None], inplace=True)

    data['Children_age: under 5'] = data['children_age: 0'] + data['children_age: 1'] + data['children_age: 2'] + data[
        'children_age: 3'] + data['children_age: 4']
    data['Children_age: 5-9'] = data['children_age: 5'] + data['children_age: 6'] + data['children_age: 7'] + data[
        'children_age: 8'] + data['children_age: 9']
    data['Children_age: 10-14'] = data['children_age: 10'] + data['children_age: 11'] + data['children_age: 12'] + data[
        'children_age: 13'] + data['children_age: 14']
    data['Children_age: 15-19'] = data['children_age: 15'] + data['children_age: 16'] + data['children_age: 17'] + data[
        'children_age: 18'] + data['children_age: 19']
    data.drop(columns=[x for x in data.columns if re.search(r'(children_age:)', x) != None], inplace=True)

    data['House_hold_head_age: 25-45'] = data['house_hold_head_age: 25-34'] + data['house_hold_head_age: 35-44']
    data['House_hold_head_age: 45-65'] = data['house_hold_head_age: 45-54'] + data['house_hold_head_age: 55-64']
    data['House_hold_head_age: 65-85'] = data['house_hold_head_age: 65-74'] + data['house_hold_head_age: 75-84']
    data['House_hold_head_age: over 85'] = data['house_hold_head_age: 85 Plus']
    data.drop(columns=[x for x in data.columns if re.search(r'(house_hold_head_age:)', x) != None], inplace=True)

    data['Earnings: under 19K'] = data['earnings: 10K'] + data['earnings: 10K-19K']
    data['Earnings: 20K-40K'] = data['earnings: 20K-29K'] + data['earnings: 30K-39K']
    data['Earnings: 40K-65K'] = data['earnings: 40K-49K'] + data['earnings: 50K-64K']
    data['Earnings: 65K-100K'] = data['earnings: 65K-74K'] + data['earnings: 75K-99K']
    data['Earnings: over 100K'] = data['earnings: 100K']
    data.drop(columns=[x for x in data.columns if re.search(r'(earnings:)', x) != None], inplace=True)

    data['Commute time: under 19 min'] = data['commute time: 10'] + data['commute time: 10-19']
    data['Commute time: 20-40 min'] = data['commute time: 20-29'] + data['commute time: 30-39']
    data['Commute time: 40-60 min'] = data['commute time: 40-44'] + data['commute time: 45-59']
    data['Commute time: 60-90 min'] = data['commute time: 60-89']
    data['Commute time: over 90 min'] = data['commute time: 90']
    data.drop(columns=[x for x in data.columns if re.search(r'(commute time:)', x) != None], inplace=True)

    data['Household_income: under 25K'] = data['household_income: 25K']
    data['Household_income: 25K-60K'] = data['household_income: 25K-44K'] + data['household_income: 45K-59K']
    data['Household_income: 60K-100K'] = data['household_income: 60K-99K']
    data['Household_income: 100K-150K'] = data['household_income: 100K-149K']
    data['Household_income: 150K-200K'] = data['household_income: 150K-199K']
    data['Household_income: over 200K'] = data['household_income: 200K']
    data.drop(columns=[x for x in data.columns if re.search(r'(household_income:)', x) != None], inplace=True)

    for i in ['1b', '2b', '3b', 'studio']:
        data['Rent_%s: under 500' % (i)] = data.loc[:, 'rent_%s: 200' % (i):'rent_%s: 300-499' % (i)].sum(axis=1).values
        data['Rent_%s: 500-1000' % (i)] = data.loc[:, 'rent_%s: 500-749' % (i):'rent_%s: 750-999' % (i)].sum(
            axis=1).values
        data['Rent_%s: over 1000' % (i)] = data['rent_%s: 1000' % (i)].values
    data.drop(columns=[x for x in data.columns if re.search(r'(rent_)', x) != None], inplace=True)

    new_columns = ['zipcode', 'major_city', 'county', 'state', 'lat', 'lng', 'population']
    for i in ['Age', 'gender', 'race', 'Children_age', 'kids', 'school enrlolment', 'House_hold_head_age',
              'family', 'Earnings', 'employment', 'education', 'income_source', 'investment_income',
              'median_household_income',
              'Household_income', 'occupied_housing_units', 'occupancy', 'median_home_value', 'owner_home_values',
              'rental prop', 'Rent_', 'vacancy']:
        columns = [x for x in data.columns if i in x]
        new_columns.extend(columns)
    data = data[new_columns].reset_index(drop=True)
    data.columns = [x.lower() for x in data.columns]
    data.loc[:, 'lat':'vacancy: vacant for other reasons'] = data.loc[:,
                                                             'lat':'vacancy: vacant for other reasons'].astype(float)
    return data

zip_dem = dem_transform(zip_data)


zip_dem.to_csv('/Users/macbook/PycharmProjects/streamlit/RealEstateDemo/zip_dem.csv')
zip_dem = pd.read_csv('/Users/macbook/PycharmProjects/streamlit/RealEstateDemo/zip_dem.csv')
st.write( "We are going to investigate census demographic data of the hottest ZIP indexes "
         "according to ranking. Comparing ZIP codes with hot real estate market to other areas in the city"
         "we're revealing differences and similarities in demographic data. In the menu below you can select "
          "among about 100 different stat data fields with regard to the zipcode")

columns = st.multiselect('Select stat data', zip_dem.columns.to_list())
list = ['zipcode', 'major_city']
list.extend(columns)
st.write(zip_dem[list])

''' 
'''
city_zip = []
for i in range(0, zip_dem.shape[0]):
    search = SearchEngine(simple_zipcode=False)
    res = search.query(city=zip_dem['major_city'][i], state=zip_dem['state'][i], returns=100)
    for zipcode in res:
        city_zip.append(zipcode.zipcode)

city_data = pd.DataFrame()
for i in city_zip:
    zipcode = search.by_zipcode(i)
    table = json_normalize(json.loads(zipcode.to_json()))
    city_data = pd.concat([city_data, table], axis=0)
city_data = city_data.reset_index(drop=True)

city_data.dropna(inplace=True)
city_data = city_data.replace({None:np.nan})
city_data = city_data.reset_index(drop=True)

city_zip_dem = dem_transform(city_data)
#city_zip_dem.to_csv('/Users/macbook/PycharmProjects/streamlit/RealEstateDemo/city_zip_dem.csv')
#city_zip_dem = pd.read_csv('/Users/macbook/PycharmProjects/streamlit/RealEstateDemo/city_zip_dem.csv')

def plot_hot(feature):
    colors = cl.scales['5']['qual']['Pastel1'] + cl.scales['5']['qual']['Pastel2']
    for i, j in enumerate(city_zip_dem['major_city'].unique()):
        x = city_zip_dem[city_zip_dem['major_city'] == j]['zipcode'].values
        y = city_zip_dem[city_zip_dem['major_city'] == j][feature].values#.sort_values(feature)
        zip_list = city_zip_dem[city_zip_dem['major_city'] == j]['zipcode'].to_list()
        index = np.where(np.array(zip_list) == hot_zip['ZIPcode'].unique().astype(str)[i])[0][0]
        width = np.zeros(len(zip_list)).tolist()
        width[index]= 2.5
        a='ZIP '+ x
        fig.add_trace(go.Bar(x=a,
                             y=y,
                             marker = dict(color=colors[i], line = dict(color = 'red', width = width))
                             ),
                      #row=row,
                      #col=col
                     )
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)',
                      showlegend=False,
                      xaxis = dict(showticklabels=True)
                     )

#fig = make_subplots(rows=3, cols=1, vertical_spacing = 0.06, shared_xaxes=True,
                   #subplot_titles=("median_household_income","population", "owner_home_values: 150k-199k")) #, shared_xaxes=True,
#col1 = st.sidebar.multiselect('Choose feature to analyse', city_zip_dem.columns)


st.title('Select features to compare')
option1 = st.selectbox('Feature 1', city_zip_dem.columns.to_list())
option2 = st.selectbox('Feature 2', city_zip_dem.columns.to_list())

fig = go.Figure()
plot_hot(option1)
fig.update_layout(title_text=option1,
                  width=1200,
                  height=1000,
                  xaxis = dict(tickmode = 'array', tickangle=-45,tickvals = [7, 26, 46, 53, 68, 77, 80, 87, 93, 107], ticktext = city_zip_dem['major_city'].unique()))
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#EEEEEE')
st.plotly_chart(fig)

fig = go.Figure()
plot_hot(option2)
fig.update_layout(title_text=option2,
                  width=1200,
                  height=1000,
                  xaxis = dict(tickmode = 'array', tickangle=-45,tickvals = [7, 26, 46, 53, 68, 77, 80, 87, 93, 107], ticktext = city_zip_dem['major_city'].unique()))
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#EEEEEE')
st.plotly_chart(fig)

st.write('Select feature to boxplot')
option3 = st.selectbox('Feature', city_zip_dem.columns.to_list())
fig = go.Figure()
y = city_zip_dem[option3]
x = city_zip_dem['major_city']

fig.add_trace(go.Box(y=y, x=x,fillcolor='rgba(0,0,0,0)', line = dict(color = 'blue')))
fig.update_layout(title_text=option3,
                  width=900,
                  height=500,
                  plot_bgcolor='rgba(0,0,0,0)',
                  xaxis=dict(tickangle=-45),
                  yaxis=dict(gridcolor='#EEEEEE', nticks=15)
                 )
st.plotly_chart(fig)

group_df = city_zip_dem.groupby(['major_city']).agg({'median_home_value': 'median',
                                                          'population': 'sum',
                                                          'median_household_income':'median',
                                                          'earnings: 65k-100k': 'sum',
                                                          'occupancy: households vacant': 'sum',
                                                          'occupied_housing_units': 'sum',
                                                          'earnings: under 19k': 'sum'}
                                                        )
fig = go.Figure([go.Bar(x=group_df.index, y=group_df['population'].sort_values(), width=0.7)])
fig.update_layout(title_text='population',
                  width=700,
                  height=400,
                  plot_bgcolor='rgba(0,0,0,0)',
                  xaxis=dict(tickangle=-45),
                  bargap=0.1,
                 )
st.plotly_chart(fig)

group_df['earnings: 65k-100k per population'] = group_df['earnings: 65k-100k']/group_df['population']*100
group_df['earnings: under 19k per population'] = group_df['earnings: under 19k']/group_df['population']*100
group_df['vacant units per total households'] = group_df['occupancy: households vacant']/(group_df['occupancy: households vacant'] + group_df['occupied_housing_units'] )*100
group_df = group_df.astype(int)

fig = go.Figure()
colors = ['magenta', 'green', 'blue']
stat = st.multiselect('Choose stat to display', ['earnings: 65k-100k per population',
                                                         'earnings: under 19k per population',
                                                         'vacant units per total households'])
for i,j in enumerate(stat):
    fig.add_trace(go.Bar(x=group_df.index,
                         y=group_df[j].sort_values().values,
                         name=j,
                         marker_color=colors[i]
                        )
                 )
fig.update_layout(barmode='group',
                  xaxis_tickangle=-45,
                  yaxis_nticks=10,
                  plot_bgcolor='rgba(0,0,0,0)'
                 )
fig.update_layout(
    legend=go.layout.Legend(
        x=0,
        y=1,
        traceorder="normal",
        font=dict(
            size=9,
            color="black"
        ),
        bgcolor='rgba(0,0,0,0)'
    )
)
st.plotly_chart(fig)







#streamlit run RealEstateDemo.py
