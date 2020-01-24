import streamlit as st
import pandas as pd
from uszipcode import SearchEngine
import numpy as np
import plotly.graph_objs as go
import re
from pandas.io.json import json_normalize
import json
import cfscrape
from lxml import etree
import colorlover as cl
import plotly.graph_objects as go
from PIL import Image
import requests
from io import BytesIO
import pydeck as pdk


#title image
response = requests.get('https://images.unsplash.com/photo-1501595685668-178fc57e6146?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=1500&q=80')
image = Image.open(BytesIO(response.content))

#title
st.title('The hottest zipcodes and demography statistics')
st.subheader('This app helps to get US Census demographics data, visualize hidden data patterns '
         'and investigate how hottest areas differ from ordinary ones')
st.image(image, use_column_width=True)

#getting hot zip data
@st.cache(allow_output_mutation=True)
def get_hot_zips():
    url_zip = 'https://www.realtor.com/research/hottest-zip-codes-2019/'
    scraper = cfscrape.create_scraper()
    scraped_html=scraper.get(url_zip).content
    html = etree.HTML(scraped_html)

    hot_zip = {}
    for i in range(1,11):
        zipcode = html.xpath("//*[@id='post-2223']/div/table[1]/tbody/tr")[i].xpath("td[2]/a/text()")
        name = html.xpath("//*[@id='post-2223']/div/table[1]/tbody/tr")[i].xpath("td[3]/a/text()")
        hot_zip[zipcode[0]] = name[0]
    hot_zip = pd.DataFrame(list(hot_zip.items()), columns=['ZIPcode', 'City, State'], index=range(0,10))
    hot_zip['Rank'] = np.arange(1, 11)
    hot_zip['ZIPcode'] = hot_zip['ZIPcode'].astype(int)
    return hot_zip
hot_zip = get_hot_zips()

st.write('While zip codes rather represent urban planning logic and poorly reveal '
         'real economic activity, they remain valuable real estate instruments for data '
         'segmentation. Having demographics data connected to zip codes we can easily '
         'compare areas with active and slow real estate markets.')

#getting Realtor survey link
@st.cache(allow_output_mutation=True, show_spinner=False)
def make_clickable(url, text):
    return f'<a target="_blank" href="{url}">{text}</a>'

url_zip = 'https://www.realtor.com/research/hottest-zip-codes-2019/'
link = make_clickable(url_zip,'Realtor')
st.write('Here is a list of zip codes marked as :fire: real estate markets by {} in 2019.'.format(link), unsafe_allow_html = True)
st.write(hot_zip)

#getting other zip codes data
@st.cache(allow_output_mutation=True, show_spinner=False)
def get_other_zip():
    zip_data = pd.DataFrame()
    for i in hot_zip['ZIPcode'].unique():
        search = SearchEngine(simple_zipcode=False)
        zipcode = search.by_zipcode(i)
        table = json_normalize(json.loads(zipcode.to_json()))
        zip_data = pd.concat([zip_data, table], axis=0)
    zip_data = zip_data.reset_index(drop=True)
    return zip_data

zip_data = get_other_zip()

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
@st.cache(allow_output_mutation=True, show_spinner=False)
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
            elif j == 'population_by_age':
                stat = pd.DataFrame(df[j][n][2]['values'])
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
    for i in ['Age:','gender:','race:','Children_age:','kids:','school enrlolment:','House_hold_head_age:',
              'family:','Earnings:','employment:','education:','Household_income:','median_household_income','median_home_value','income_source','investment_income',
              'occupied_housing_units','occupancy:','owner_home_values:',
              'rental prop','Rent_','vacancy']:
        columns = [x for x in data.columns if i in x]
        new_columns.extend(columns)
    data = data[new_columns].reset_index(drop=True)
    data.columns = [x.lower() for x in data.columns]
    data.loc[:, 'lat':'vacancy: vacant for other reasons'] = data.loc[:,
                                                             'lat':'vacancy: vacant for other reasons'].astype(float)
    return data
zip_dem = dem_transform(zip_data)

#zip_dem.to_csv('/Users/macbook/PycharmProjects/streamlit/RealEstateDemo/zip_dem.csv')
#zip_dem = pd.read_csv('/Users/macbook/PycharmProjects/streamlit/RealEstateDemo/zip_dem.csv')
st.write(' ')
st.write(' ')
st.write( "Below are the instruments which help us connect to the US Census database, visualize data and "
          "choose any statistical measures with regards to earnings, education, age, household income "
          "and more than dozen different data sections.")
st.write('**Select demographic data to show in the table below**')
columns = st.multiselect('choose column', zip_dem.columns.to_list())
list = ['zipcode', 'major_city','population']
list.extend(columns)
st.write(zip_dem[list])

''' 
'''
@st.cache(allow_output_mutation=True, show_spinner=False)
def get_city_data():
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
    return city_data
city_data = get_city_data()
city_zip_dem = dem_transform(city_data)
#city_zip_dem.to_csv('/Users/macbook/PycharmProjects/streamlit/RealEstateDemo/city_zip_dem.csv')
#city_zip_dem = pd.read_csv('/Users/macbook/PycharmProjects/streamlit/RealEstateDemo/city_zip_dem.csv')




def grouped_bars(df, stat, tickformat=''):
    for i,j in enumerate(stat):
        fig.add_trace(go.Bar(x=df[j].sort_values().index,
                         y=df[j].sort_values().values,
                         name=j,
                        )
                 )
    fig.update_layout(barmode='group',
                      xaxis_tickangle=-45, width=700, height=500,
                      yaxis=dict(gridcolor='#EEEEEE', nticks=10, tickformat = tickformat),
                      plot_bgcolor='rgba(0,0,0,0)',
                      margin = go.layout.Margin(l=0, r=0, b=50, t=0),
                    )
    fig.update_layout(
        legend=go.layout.Legend(
            x=0.98, y=1, traceorder="normal",
            font=dict(size=10,color="black"),
            bgcolor='rgba(0,0,0,0)'
        )
    )


st.subheader('**Choose statistical features to compare on the bar plot**')
data_field = st.radio("What kind of data would you like to investigate?", ('demographics (age, employment, education, etc)',
                                                                      'income',
                                                                    'real estate (occupancy, home values, rent, etc)'))
f = city_zip_dem.columns.get_loc
#@st.cache(allow_output_mutation=True)
def grouped_df(data_field):
    if data_field == 'demographics (age, employment, education, etc)':
        demographics = city_zip_dem.iloc[:, np.r_[f('major_city'), f('population'):f('household_income: over 200k')]]
        group_df = demographics.groupby(['major_city']).sum()

    if data_field == 'income':
        income = city_zip_dem.iloc[:, np.r_[f('major_city'), f('median_household_income'):f('investment_income: capital gains')]]
        group_df = income.groupby(['major_city']).median()

    if data_field == 'real estate (occupancy, home values, rent, etc)':
        real_estate = city_zip_dem.iloc[:, np.r_[f('major_city'), f('occupied_housing_units'):f('vacancy: vacant for other reasons')]]
        group_df = real_estate.groupby(['major_city']).sum()

    return group_df

group_df = grouped_df(data_field)

#group_df = group_df.astype('float').round(2)
select = st.multiselect('See how age, earnings or home values differ in the cities', group_df.iloc[:,0:].columns)

if len(select)<1:
    stat = [group_df.columns[1]]
    stat.extend(select)
else:
    stat = select

data_type = st.radio("Percentage or Numeric values?", ('numeric', 'percentage'))

if data_type == 'percentage': #and data_field != 'income' and data_field != 'real estate (occupancy, home values, rent, etc)':
    df = group_df.copy()
    for col in df.columns[1:]:
        df[col] = (df[col]/pd.DataFrame(df.iloc[:,0:1].values)[0].values).values
    df.iloc[:,0:1] = 1
    st.write(df)
    fig = go.Figure()
    grouped_bars(df, stat,'%')
    st.plotly_chart(fig)

elif data_type == 'numeric':
    fig = go.Figure()
    grouped_bars(group_df, stat)
    st.write(group_df)
    st.plotly_chart(fig)
else:
   st.write('_Available only for demographics data_')

stat_subsection = ['age','gender','race','children_age','kids','school enrlolment',
 'house_hold_head_age','family','earnings','employment','education',
'household_income','income_source','investment_income','occupancy',
'owner_home_values','rental prop','rent_1b','rent_2b','rent_3b','rent_studio','vacancy']
dist_feat = st.selectbox('choose stat subsection to show distribution', stat_subsection)

bubble_df = pd.DataFrame([])
for field in ['demographics (age, employment, education, etc)',
              'income',
              'real estate (occupancy, home values, rent, etc)']:
    group_df = grouped_df(field)
    for col in group_df.columns[1:]:
        group_df[col] = (group_df[col]/pd.DataFrame(group_df.iloc[:,0:1].values)[0].values).values
    bubble_df = pd.concat([bubble_df,group_df], axis=1)

index_start = np.where([re.search('^{}:'.format(dist_feat), x) for x in bubble_df.columns])[0][0]
index_end = np.where([re.search('^{}:'.format(dist_feat), x) for x in bubble_df.columns])[0][-1]

x1 = bubble_df.columns[index_start:index_end+1]
y_plot = []
x_plot = []
size1 = []
for i in bubble_df.index:
    y_plot.extend([i]*len(x1))
    x_plot.extend(x1)
    size1.extend(bubble_df.loc[i][bubble_df.columns[index_start]:bubble_df.columns[index_end]].values)

coef_size = np.abs(np.log(np.abs(max(size1)*100)))
size2 = [x*550/coef_size for x in size1]
text = [str(round(x*100,1)) + '%' for x in size1]
trace1 = go.Scatter(y = y_plot, x = x_plot, mode='markers+text', textposition="middle center", text=text, name="",
                        marker=dict(opacity=0.2, size=size2, sizemin=2),
                        textfont=dict(size=10))
#font=dict(size=10,color="black"),
layout = go.Layout(barmode='stack', margin=dict(l=0, r=0, b=50, t=0), height=900, width=600, title='X Distribution',
                   plot_bgcolor='#fff', paper_bgcolor='#fff', showlegend=False)

fig = go.Figure(data=[trace1], layout=layout)
st.plotly_chart(fig)

#@st.cache()
def bar_stacked_plot(major_city, first_col, last_col):
    plot = city_zip_dem[city_zip_dem['major_city'] == major_city].copy()
    plot = plot.loc[:, first_col:last_col].join(plot['zipcode'])
    plot['zipcode'] = 'ZIP ' + plot['zipcode'].astype(str)
    plot['Total'] = plot.loc[:, first_col:last_col].sum(axis=1)

    for i in plot.columns[0:-2]:
        plot[i] = plot[i] / plot['Total'] * 100
    plot.drop(columns=['Total'], inplace=True)
    x = plot['zipcode'].to_list()

    for i, j in enumerate(plot.columns[0:-1]):
        fig.add_trace(go.Bar(y=x,
                             x=plot[j].values,
                             orientation='h',
                             name=j,
                             text=plot[j].values.astype(int),
                             textposition='inside',
                             texttemplate="%{text}%",
                             textfont={'family': "PT Sans", 'size': 12,
                                       'color': 'white'
                                       },
                             )
                      )

st.write('**Select data to show distribution**')
option3 = st.selectbox('Chose feature', city_zip_dem.loc[:,'population':].columns.to_list())
fig = go.Figure()
y = city_zip_dem[option3]
x = city_zip_dem['major_city']

fig.add_trace(go.Box(y=y, x=x,fillcolor='rgba(0,0,0,0)', line = dict(color = 'blue')))
fig.update_layout(#title_text=option3,
                  width=700,
                  height=500,
                  plot_bgcolor='rgba(0,0,0,0)',
                  xaxis=dict(tickangle=-45),
                  yaxis=dict(gridcolor='#EEEEEE', nticks=15),
                  margin = go.layout.Margin(l=0, r=0, b=50, t=0),
                 )
st.plotly_chart(fig)



#@st.cache()
def plot_hot(feature):
    #colors = cl.scales['5']['qual']['Pastel1'] + cl.scales['5']['qual']['Pastel2']
    for i, j in enumerate(city_zip_dem['major_city'].unique()):
        x = city_zip_dem[city_zip_dem['major_city'] == j]['zipcode'].values
        y = city_zip_dem[city_zip_dem['major_city'] == j][feature].values#.sort_values(feature)
        zip_list = city_zip_dem[city_zip_dem['major_city'] == j]['zipcode'].to_list()
        if hot_zip['ZIPcode'].unique().astype(str)[i] in zip_list:
            index = np.where(np.array(zip_list) == hot_zip['ZIPcode'].unique().astype(str)[i])[0][0]
        else:
            index = 0
        width = np.zeros(len(zip_list)).tolist()
        width[index]= 1.5
        a='ZIP '+ x
        fig.add_trace(go.Bar(x=a,
                             y=y,
                             marker = dict(#color=colors[i],
                                           line = dict(color = 'red', width = width))
                             )
                     )
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)',
                      showlegend=False,
                      xaxis=dict(showticklabels=True),
                      margin=go.layout.Margin(l=0, r=0, b=0, t=0)
                     )

#col1 = st.sidebar.multiselect('Choose feature to analyse', city_zip_dem.columns)


st.markdown('**Select data to compare two statistical features** :fire: _zip codes marked red_')
option1 = st.selectbox('first feature', city_zip_dem.loc[:,'population':].columns.to_list())
option2 = st.selectbox('second feature', city_zip_dem.loc[:,'population':].columns.to_list())


fig = go.Figure()
plot_hot(option1)
fig.update_layout(#title_text=option1,
                  width=700,
                  height=500,
                  xaxis = dict(tickmode = 'array', tickangle=-45,tickvals = [7, 26, 46, 53, 68, 77, 80, 87, 93, 107], ticktext = city_zip_dem['major_city'].unique()),
                  )

fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#EEEEEE')
st.plotly_chart(fig)

fig = go.Figure()
plot_hot(option2)
fig.update_layout(#title_text=option2,
                  width=700,
                  height=500,
                  xaxis = dict(tickmode = 'array', tickangle=-45,tickvals = [7, 26, 46, 53, 68, 77, 80, 87, 93, 107], ticktext = city_zip_dem['major_city'].unique()),
                  )

fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#EEEEEE')
st.plotly_chart(fig)




st.subheader('Choose city to show internal statisticks')
city = st.selectbox('choose city', city_zip_dem['major_city'].unique())
feature = st.selectbox('choose data to plot', stat_subsection)
fire_zip = hot_zip[hot_zip['City, State'].str.contains(city)]['ZIPcode'].values[0]
st.write(':fire: _zip code in {} is **{}**_'.format(city, fire_zip))
index_start = np.where([re.search('^{}:'.format(feature), x) for x in city_zip_dem.columns])[0][0]
index_end = np.where([re.search('^{}:'.format(feature), x) for x in city_zip_dem.columns])[0][-1]

fig = go.Figure()
bar_stacked_plot(city, city_zip_dem.columns[index_start], city_zip_dem.columns[index_end])

if city_zip_dem[city_zip_dem['major_city'] == city].shape[0] < 5:
    height = city_zip_dem[city_zip_dem['major_city'] == city].shape[0]*70
else:
    height = city_zip_dem[city_zip_dem['major_city'] == city].shape[0]*28

fig.update_layout(barmode='stack',
                  #title_text=feature,
                  width=700,
                  height=1000,
                  yaxis_nticks =50,
                  paper_bgcolor='rgba(0,0,0,0)',
                  plot_bgcolor='rgba(0,0,0,0)',
                  bargap=0,
                  margin = go.layout.Margin(l=0, r=0, b=00, t=10),
                 )
fig.update_layout(
        legend=go.layout.Legend(
            x=0.95, y=1, traceorder="normal",
            font=dict(size=10,color="black"),
            bgcolor='rgba(0,0,0,0)'
        )
    )

st.plotly_chart(fig)
