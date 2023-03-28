from django.shortcuts import render
from django.http import JsonResponse
from django.middleware.csrf import get_token
import numpy as np 
import pandas as pd 
import random
import math
import time
import json
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.svm import SVR
import datetime

# Create your views here.
def predict(request):
    if request.method == 'GET':
        confirmed_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
        deaths_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
        latest_data = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/01-15-2023.csv')
        confirmed_cols = confirmed_df.keys()
        deaths_cols = deaths_df.keys()
        confirmed = confirmed_df.loc[:, confirmed_cols[4]:]
        deaths = deaths_df.loc[:, deaths_cols[4]:]
        ck = confirmed.keys()
        num_dates = len(confirmed.keys())
        world_cases = []
        for i in range(num_dates):
            confirmed_sum = confirmed[ck[i]].sum()
            world_cases.append(confirmed_sum)
        days_since_1_22 = np.array([i for i in range(len(ck))]).reshape(-1, 1)
        world_cases = np.array(world_cases).reshape(-1, 1)
        days_in_future = 10
        future_forcast = np.array([i for i in range(len(ck)+days_in_future)]).reshape(-1, 1)
        adjusted_dates = future_forcast[:-10]
        start = '1/22/2020'
        start_date = datetime.datetime.strptime(start, '%m/%d/%Y')
        future_forcast_dates = []
        for i in range(len(future_forcast)):
            future_forcast_dates.append((start_date + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))
        days_to_skip = 922
        X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(days_since_1_22[days_to_skip:], world_cases[days_to_skip:], test_size=0.07, shuffle=False) 
        svm_confirmed = SVR(shrinking=True, kernel='poly',gamma=0.01, epsilon=1,degree=3, C=0.1)
        svm_confirmed.fit(X_train_confirmed, y_train_confirmed)
        svm_pred = svm_confirmed.predict(future_forcast)
        svm_df = pd.DataFrame({'Date': future_forcast_dates[-10:], 'Cases': np.round(svm_pred[-10:])})
        json_list = json.loads(json.dumps(list(svm_df.T.to_dict().values())))
        return JsonResponse(json_list,safe = False)
    else:
        print(get_token(request))