---
layout: post
title:  "Monitoring Airbnb reviews over COVID-19 with folium HeatMapWithTime"
description: Maybe you're familar with a heatmap, what about adding timestamp on it?
tags: EDA data-visulization tutorial time-series pandas folium plotly
usemathjax: true
---

<br />

<iframe src="https://airbnb-map.netlify.app" width="700" height="500"></iframe>

<br />

## Introduction

  

This artile will walk through visualizing the fluctuations of Airbnb busineess affected by COVID-19 pandemic. Intuitively, we might have a rough guess what the curves will be looking like, however, I thought this would be interesting to practice both data processing and visulization when time-series attributes involved, what's more important, to better explore the data and express the insight to a wide variety of audience in a more approachable manner, an interactive visulization significantly helps.

  

## Data

  

The dataset was downloaded from [InsideAirbnb-Hawaii-data](http://insideairbnb.com/get-the-data.html), the analysis results were based on a span of two year from July, 2019 to July, 2021.

## Prerequisite

In addition to commonly used python data science packages(numpy, pandas, matplotlib, seaborn), here we also need install `plotly`, `folium`, `chart-studio`, those can be easily installed via pip under conda environment:
~~~sh
$ pip install plotly
$ pip install folium
$ pip install chart-studio
~~~

<br />

## Part 1. Visualizing with `matplotlib` and `plotly`

<br />

### Inspecting Data

I will mainly working on `reviews.csv` and `listings.csv`, merging operations will be included later for visulization. 


```python
df_review = pd.read_csv('reviews.csv')
df_listing = pd.read_csv('listings.csv')
```


df_listing contains all the metadata for all the listings and df_review has all the review information for each review associated with its list id. Noticed `listing_id` in df_review is `id` in df_listing, it will be the key for our merging.

<br />

### Data processing:  the number of reviews

Apparently, there is no obvious feature indicating the fluctuations of Airbnb busineess activity, however, from df_review we can tell that each listing has received multiple comments in various time, with that being said, we can get all the listings that have received any comment on different dates, then **the count of the total comments on each date will be good enough to indicate the change of business activity**. In addition, we can add any metadata associated with listing based on our needs, for instance, we may consider adding geographic features(`neighbourhood`,`latitude`,`longitude`) since we are creating a folium map later, also this kind of feature helps to segragate the data to better examine the change focusing on a particular area.

#### Inspect the number of reviews over time


```python
def process_count(df_review, df_listing):
    df1 = df_review.drop(['reviewer_id', 'reviewer_name', 'comments'], axis=1)
    df2 = df_listing[['id', 'neighbourhood_group_cleansed']]
    df3 = pd.merge(df2, df1, left_on='id', right_on='listing_id')
    df3 = df3[['date', 'listing_id', 'neighbourhood_group_cleansed']]
    df3.date = pd.to_datetime(df3.date, format="%Y-%m-%d")
    df3 = df3[df3['date'].isin(pd.date_range("2019-07-10", "2021-07-10"))]
    df3 = df3.set_index(df3.date).drop('date', axis=1)
    #df3.to_pickle("ts.pkl")
    return df3
```


```python
listing_reveived_review = process_count(df_review, df_listing)
listing_reveived_review
```

output:


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>listing_id</th>
      <th>neighbourhood_group_cleansed</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019-08-19</th>
      <td>5065</td>
      <td>Hawaii</td>
    </tr>
    <tr>
      <th>2020-02-16</th>
      <td>5065</td>
      <td>Hawaii</td>
    </tr>
    <tr>
      <th>2020-02-19</th>
      <td>5065</td>
      <td>Hawaii</td>
    </tr>
    <tr>
      <th>2020-02-28</th>
      <td>5065</td>
      <td>Hawaii</td>
    </tr>
    <tr>
      <th>2020-03-09</th>
      <td>5065</td>
      <td>Hawaii</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2021-07-08</th>
      <td>50599343</td>
      <td>Hawaii</td>
    </tr>
    <tr>
      <th>2021-07-08</th>
      <td>50682739</td>
      <td>Honolulu</td>
    </tr>
    <tr>
      <th>2021-07-08</th>
      <td>50710529</td>
      <td>Hawaii</td>
    </tr>
    <tr>
      <th>2021-07-03</th>
      <td>50736557</td>
      <td>Honolulu</td>
    </tr>
    <tr>
      <th>2021-07-07</th>
      <td>50752275</td>
      <td>Kauai</td>
    </tr>
  </tbody>
</table>
<p>271528 rows × 2 columns</p>
</div>



Then we specify the area to Honolulu and count how many listings that have received comments on each date.


```python
def count_listings(df, loc='Honolulu'):
    df = df[df.neighbourhood_group_cleansed == loc]
    df = df.groupby('date', sort=True)['listing_id'].count().rename('review_count').reset_index().set_index('date')
    return df
```


```python
count_per_day_honolulu = count_listings(listing_reveived_review, loc='Honolulu')
count_per_day_honolulu
```

output:


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>review_count</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019-07-10</th>
      <td>163</td>
    </tr>
    <tr>
      <th>2019-07-11</th>
      <td>151</td>
    </tr>
    <tr>
      <th>2019-07-12</th>
      <td>145</td>
    </tr>
    <tr>
      <th>2019-07-13</th>
      <td>164</td>
    </tr>
    <tr>
      <th>2019-07-14</th>
      <td>208</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>2021-07-04</th>
      <td>95</td>
    </tr>
    <tr>
      <th>2021-07-05</th>
      <td>142</td>
    </tr>
    <tr>
      <th>2021-07-06</th>
      <td>71</td>
    </tr>
    <tr>
      <th>2021-07-07</th>
      <td>48</td>
    </tr>
    <tr>
      <th>2021-07-08</th>
      <td>24</td>
    </tr>
  </tbody>
</table>
<p>729 rows × 1 columns</p>
</div>



#### Create time variable


```python
# define time variable
ts = count_per_day_honolulu.review_count
```

#### Fit moving average


```python
def fit_moving_avg(series, window=5):
    """calculate moving average number of reviews
    """
    return series.rolling(window, center=True).mean()
```


```python
avg_ts = fit_moving_avg(ts)
```

<br />

### Visualization

#### via matplotlib


```python
# plot time variable
ts.plot(figsize=(16, 4), label='Number of reviews', title='Number of Reviews over time', fontsize=14, alpha=0.6)
# plot moving average
avg_ts.plot(label='Average number of reviews', fontsize=14)
plt.legend(fontsize=14)
#plt.savefig('moving_avg.png')
```

output:

    
<img src="/assets/images/moving_avg.png">
    

<br />

#### Insights
- This analysis takes the number of reviews per day as an indicator of Airbnb business activity.  It dramaticly decreased after outbreak of COVID-19 in March, 2020.
- The number of reviews keeps increasing which indicates the popularity of Airbnb was thriving before the pandemic. Meanwhile there is a clear seasonality pattern before the mid of Feb, 2020.

<br />

#### Interactive visualization with `Plotly`


```python
# set up for plotly
r = go.Scatter(x=ts.index, y=ts.values, 
               line=go.scatter.Line(color='red', width = 1), opacity=0.8, 
               name='Reviews', text=[f'Reviews: {x}' for x in ts.values])

layout = go.Layout(title='Number of Reviews over time', xaxis=dict(title='Date'),
                   yaxis=dict(title='Count'))

fig1 = go.Figure(data=[r],layout=layout)
iplot(fig1)
```

output:

<iframe width="700" height="450" frameborder="0" scrolling="no" src="//plotly.com/~huyuan17/2.embed"></iframe>

<br />

#### Add a `rangeselector`

[rangeSelector](https://gist.github.com/galaxie500/a45e67f59691ddd482956e708a53863e)

output:

<iframe width="700" height="450" frameborder="0" scrolling="no" src="//plotly.com/~huyuan17/4.embed"></iframe>


Use following snippet to push the plotly object to `plotly express`, which generates embedding information for hosting the interactive image on web pages.


```python
username = '' # your username
api_key = '' # your api key for plotly express - go to profile > settings > regenerate key
chart_studio.tools.set_credentials_file(username=username, api_key=api_key)

py.plot(fig1, filename = 'review_over_time', auto_open=True)
py.plot(fig2, filename = 'review_over_time', auto_open=True)
```



<br />

## Part 2. Reviews over time via `folium HeatMapWithTime` plugin


### Data processing

#### Count reviews received for each listing each month

As you probably already know, `folium` creates great interactive maps for visualization, here I am going to create a heat map along with time stamp by using its `HeatMapWithTime` plugin, before that there still some processing work need to complete in order to fit our data to the plugin. [Demo](https://github.com/python-visualization/folium/blob/master/examples/HeatMapWithTime.ipynb).

First, count how many reviews each listing received each day, then change the time range to Month, that is to record how many reviews each listing received each month. The reason that change time range from day to month is we don't want the final display moving too frequently so that we can clearly spot the moving trend.

Also, features like `latitude` and `longitude` will be kept for rendering the map.


```python
def process_listing_total_count(df_review, df_listing):
        # process data for timestamped folium map
        df2 = df_review.groupby(['listing_id', 'date'])['id'].count().rename('review_count').reset_index()
        df2['date'] = pd.to_datetime(df2['date'])
        df2 = df2[df2['date'].isin(pd.date_range("2019-07-10", "2021-07-10"))]
        df2 = df2.groupby(['listing_id', pd.Grouper(key='date', freq='M')])['review_count'] \
            .sum().reset_index()

        merged = pd.merge(df_listing, df2, left_on='id', right_on='listing_id')
        #merged.to_pickle("timestamped_review.pkl")
        return merged
```


```python
listings_with_total_review_count = process_listing_total_count(df_review, df_listing)
listings_with_total_review_count[['id','date','review_count','latitude','longitude']]
```

output:

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>date</th>
      <th>review_count</th>
      <th>latitude</th>
      <th>longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5065</td>
      <td>2019-08-31</td>
      <td>1</td>
      <td>20.042660</td>
      <td>-155.432590</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5065</td>
      <td>2020-02-29</td>
      <td>3</td>
      <td>20.042660</td>
      <td>-155.432590</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5065</td>
      <td>2020-03-31</td>
      <td>2</td>
      <td>20.042660</td>
      <td>-155.432590</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5269</td>
      <td>2019-07-31</td>
      <td>1</td>
      <td>20.027400</td>
      <td>-155.702000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5269</td>
      <td>2019-09-30</td>
      <td>3</td>
      <td>20.027400</td>
      <td>-155.702000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>116719</th>
      <td>50599343</td>
      <td>2021-07-31</td>
      <td>2</td>
      <td>19.607210</td>
      <td>-155.976120</td>
    </tr>
    <tr>
      <th>116720</th>
      <td>50682739</td>
      <td>2021-07-31</td>
      <td>1</td>
      <td>21.286110</td>
      <td>-157.840150</td>
    </tr>
    <tr>
      <th>116721</th>
      <td>50710529</td>
      <td>2021-07-31</td>
      <td>1</td>
      <td>19.700066</td>
      <td>-155.073502</td>
    </tr>
    <tr>
      <th>116722</th>
      <td>50736557</td>
      <td>2021-07-31</td>
      <td>1</td>
      <td>21.360280</td>
      <td>-158.048220</td>
    </tr>
    <tr>
      <th>116723</th>
      <td>50752275</td>
      <td>2021-07-31</td>
      <td>1</td>
      <td>22.220010</td>
      <td>-159.476470</td>
    </tr>
  </tbody>
</table>
<p>116724 rows × 5 columns</p>
</div>


<br />

#### Transform data to the form that `folium` can take

For next step, our dataframe should be looking like below:

| time-index  |  latitude | longitude  | review_count  | 
|---|---|---|---|
|  date 1 |  a list of latitude | a list of longitude |  a list of review_count |   
|  date 2 |  a list of latitude | a list of longitude |  a list of review_count |   
|  date 3 |  a list of latitude | a list of longitude |  a list of review_count |   
|  ... |  ... | ...  | ...  |


```python
review_count_time_map = listings_with_total_review_count.drop(['neighbourhood_group_cleansed', 'listing_id'], axis=1)
review_count_time_map = review_count_time_map.groupby('date').agg(lambda x: list(x))
review_count_time_map[['latitude','longitude','review_count']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>latitude</th>
      <th>longitude</th>
      <th>review_count</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019-07-31</th>
      <td>[20.0274, 19.43081, 21.88151, 22.2208, 21.8813...</td>
      <td>[-155.702, -155.88069, -159.47346, -159.46989,...</td>
      <td>[1, 1, 4, 1, 3, 3, 1, 1, 2, 3, 1, 2, 1, 2, 1, ...</td>
    </tr>
    <tr>
      <th>2019-08-31</th>
      <td>[20.04266, 19.56604, 21.88151, 22.2208, 21.881...</td>
      <td>[-155.43259, -155.96199, -159.47346, -159.4698...</td>
      <td>[1, 2, 4, 2, 3, 5, 2, 3, 3, 3, 1, 1, 2, 1, 1, ...</td>
    </tr>
    <tr>
      <th>2019-09-30</th>
      <td>[20.0274, 21.88151, 22.2208, 21.88139, 19.6066...</td>
      <td>[-155.702, -159.47346, -159.46989, -159.47248,...</td>
      <td>[3, 4, 1, 3, 5, 4, 6, 4, 2, 2, 2, 3, 2, 1, 1, ...</td>
    </tr>
    <tr>
      <th>2019-10-31</th>
      <td>[20.0274, 19.43081, 19.56604, 21.88151, 22.220...</td>
      <td>[-155.702, -155.88069, -155.96199, -159.47346,...</td>
      <td>[2, 2, 1, 2, 1, 2, 4, 3, 1, 4, 3, 3, 4, 2, 1, ...</td>
    </tr>
    <tr>
      <th>2019-11-30</th>
      <td>[19.43081, 19.56604, 21.88151, 22.2208, 21.881...</td>
      <td>[-155.88069, -155.96199, -159.47346, -159.4698...</td>
      <td>[1, 1, 5, 1, 6, 1, 3, 2, 2, 2, 1, 2, 1, 3, 1, ...</td>
    </tr>
    <tr>
      <th>2019-12-31</th>
      <td>[19.43081, 19.56604, 21.88151, 22.2208, 21.881...</td>
      <td>[-155.88069, -155.96199, -159.47346, -159.4698...</td>
      <td>[1, 1, 1, 1, 4, 5, 2, 3, 1, 2, 1, 1, 2, 2, 2, ...</td>
    </tr>
    <tr>
      <th>2020-01-31</th>
      <td>[19.43081, 19.56604, 21.88151, 22.2208, 21.881...</td>
      <td>[-155.88069, -155.96199, -159.47346, -159.4698...</td>
      <td>[2, 1, 2, 2, 3, 1, 4, 2, 2, 1, 3, 1, 1, 4, 4, ...</td>
    </tr>
    <tr>
      <th>2020-02-29</th>
      <td>[20.04266, 19.43081, 21.88151, 21.88139, 19.60...</td>
      <td>[-155.43259, -155.88069, -159.47346, -159.4724...</td>
      <td>[3, 2, 2, 2, 1, 3, 2, 3, 2, 3, 4, 2, 1, 2, 2, ...</td>
    </tr>
    <tr>
      <th>2020-03-31</th>
      <td>[20.04266, 20.0274, 19.43081, 19.56604, 21.881...</td>
      <td>[-155.43259, -155.702, -155.88069, -155.96199,...</td>
      <td>[2, 1, 1, 1, 1, 1, 2, 4, 1, 1, 2, 1, 1, 2, 2, ...</td>
    </tr>
    <tr>
      <th>2020-04-30</th>
      <td>[20.89861, 19.60885, 21.58052, 20.76291, 20.76...</td>
      <td>[-156.68151, -155.96764, -158.10854, -156.4573...</td>
      <td>[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...</td>
    </tr>
    <tr>
      <th>2020-05-31</th>
      <td>[19.57365, 20.72916, 20.76291, 20.87319, 19.72...</td>
      <td>[-155.96716, -156.45055, -156.45734, -156.6745...</td>
      <td>[1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...</td>
    </tr>
    <tr>
      <th>2020-06-30</th>
      <td>[19.57365, 21.28334, 20.72916, 19.43428, 19.81...</td>
      <td>[-155.96716, -157.8379, -156.45055, -155.21609...</td>
      <td>[2, 2, 2, 1, 1, 1, 1, 1, 1, 3, 1, 1, 2, 1, 2, ...</td>
    </tr>
    <tr>
      <th>2020-07-31</th>
      <td>[21.88151, 19.60668, 20.89861, 19.39402, 19.52...</td>
      <td>[-159.47346, -155.97585, -156.68151, -154.9306...</td>
      <td>[3, 2, 3, 2, 1, 1, 1, 2, 2, 1, 2, 1, 1, 1, 2, ...</td>
    </tr>
    <tr>
      <th>2020-08-31</th>
      <td>[21.88151, 19.52067, 21.27437, 19.48092, 21.28...</td>
      <td>[-159.47346, -154.84706, -157.82043, -155.9064...</td>
      <td>[2, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 3, 1, 1, 3, ...</td>
    </tr>
    <tr>
      <th>2020-09-30</th>
      <td>[19.39402, 19.45962, 21.27918, 21.64094, 19.48...</td>
      <td>[-154.9306, -155.88118, -157.82846, -158.06281...</td>
      <td>[1, 1, 2, 1, 2, 2, 1, 1, 1, 2, 4, 1, 1, 1, 2, ...</td>
    </tr>
    <tr>
      <th>2020-10-31</th>
      <td>[20.72413, 19.52373, 19.57365, 19.59802, 19.48...</td>
      <td>[-156.44767, -154.84746, -155.96716, -154.9389...</td>
      <td>[1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2, ...</td>
    </tr>
    <tr>
      <th>2020-11-30</th>
      <td>[21.88151, 22.2208, 22.21789, 20.89861, 20.757...</td>
      <td>[-159.47346, -159.46989, -159.47184, -156.6815...</td>
      <td>[1, 1, 1, 1, 2, 2, 1, 1, 2, 1, 3, 2, 2, 1, 1, ...</td>
    </tr>
    <tr>
      <th>2020-12-31</th>
      <td>[19.56604, 21.88151, 22.2208, 19.39402, 19.520...</td>
      <td>[-155.96199, -159.47346, -159.46989, -154.9306...</td>
      <td>[1, 1, 1, 2, 2, 4, 4, 2, 2, 2, 3, 1, 1, 3, 1, ...</td>
    </tr>
    <tr>
      <th>2021-01-31</th>
      <td>[20.0274, 19.56604, 20.72413, 19.39402, 20.757...</td>
      <td>[-155.702, -155.96199, -156.44767, -154.9306, ...</td>
      <td>[1, 2, 2, 1, 3, 2, 1, 1, 1, 1, 4, 3, 2, 3, 1, ...</td>
    </tr>
    <tr>
      <th>2021-02-28</th>
      <td>[19.43081, 19.56604, 19.60668, 20.72413, 20.89...</td>
      <td>[-155.88069, -155.96199, -155.97585, -156.4476...</td>
      <td>[2, 1, 3, 1, 1, 1, 3, 2, 2, 3, 2, 2, 3, 1, 1, ...</td>
    </tr>
    <tr>
      <th>2021-03-31</th>
      <td>[19.43081, 19.56604, 22.2208, 19.60668, 22.217...</td>
      <td>[-155.88069, -155.96199, -159.46989, -155.9758...</td>
      <td>[3, 2, 1, 3, 1, 3, 5, 4, 2, 1, 2, 3, 3, 3, 3, ...</td>
    </tr>
    <tr>
      <th>2021-04-30</th>
      <td>[20.0274, 19.43081, 19.56604, 21.88151, 22.220...</td>
      <td>[-155.702, -155.88069, -155.96199, -159.47346,...</td>
      <td>[1, 1, 1, 1, 1, 1, 2, 3, 2, 1, 3, 3, 3, 2, 1, ...</td>
    </tr>
    <tr>
      <th>2021-05-31</th>
      <td>[19.43081, 19.56604, 21.88151, 22.2208, 21.881...</td>
      <td>[-155.88069, -155.96199, -159.47346, -159.4698...</td>
      <td>[1, 1, 2, 5, 3, 3, 4, 2, 2, 4, 4, 2, 3, 2, 1, ...</td>
    </tr>
    <tr>
      <th>2021-06-30</th>
      <td>[19.43081, 19.56604, 21.88151, 22.2208, 21.881...</td>
      <td>[-155.88069, -155.96199, -159.47346, -159.4698...</td>
      <td>[1, 3, 2, 3, 2, 1, 4, 2, 1, 2, 2, 3, 1, 2, 2, ...</td>
    </tr>
    <tr>
      <th>2021-07-31</th>
      <td>[22.21789, 19.39402, 19.59074, 19.52067, 22.22...</td>
      <td>[-159.47184, -154.9306, -155.97143, -154.84706...</td>
      <td>[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, ...</td>
    </tr>
  </tbody>
</table>
</div>

<br />

Lastly, we just extract the repective `latitude`, `longitude`, and `review_count` to a final form for pass in to the plugin.


```python
# transform data for folium
def generate_location_points(all_points):
    # all_points = pd.read_pickle("timestamped_review.pkl")
    # loc_points = all_points[all_points.neighbourhood_group_cleansed == location]
    loc_points = all_points.drop(['neighbourhood_group_cleansed', 'listing_id'], axis=1)
    loc_points = loc_points.groupby('date').agg(lambda x: list(x))

    to_draw = []
    for i in range(loc_points.shape[0]):
        single_draw = []
        for j in list(zip(loc_points.iloc[i].latitude, loc_points.iloc[i].longitude, loc_points.iloc[i].review_count)):
            single_draw.append(list(j))
        to_draw.append(single_draw)

    time_index = []
    for t in loc_points.index:
        time_index.append(t.strftime("%Y-%m-%d"))

    return to_draw, time_index
```


```python
points, indices = generate_location_points(listings_with_total_review_count)[0], \
                      generate_location_points(listings_with_total_review_count)[1]
```

<br />

### Visualization - click display button 


```python
# create folium object and add timestamp object
time_map = folium.Map([21.3487, -157.944], zoom_start=10.5)
hm = plugins.HeatMapWithTime(points, index=indices, auto_play=True, max_opacity=0.6)
hm.add_to(time_map)

# display map
#time_map
#time_map.save("index.html")
```

<br />

### Insight

- The heat map above indicates how the number of reviews of Airbnb listings changed on Oahu island during 2019-7 to 2021-7
- Time variable is incremented by month, this can be adjusted to a bigger of smaller increment.
- The trend showed from above matches the result acquired by matplotlib.
- Moreover, timestamped visualization delivers a more approachable result to a variaty of audiences.
- By taking advantage of geographaic information, we could monitor many other attributes as long as we render them with suitable weights.
