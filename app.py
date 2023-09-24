import datetime
from tracemalloc import start
import streamlit as st
import pandas as pd
import numpy as np
# import pandas_datareader as data
from pandas_datareader import data as pdr
import yfinance as yfin
import matplotlib.pyplot as plt
from datetime import date
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
from dateutil.relativedelta import relativedelta
import os
path = os.getcwd()
os.chdir(".")


# title 
st.title("**StockAssist**")

page_names = ['Analysis','Prediction']
st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;} </style>', unsafe_allow_html=True)

st.write('<style>div.st-bf{flex-direction:column;} div.st-ag{font-weight:bold;padding-left:2px;padding-right:20px;}</style>', unsafe_allow_html=True)

page = st.sidebar.radio('Explore:',page_names)

if page == 'Analysis':
    try:
        stock_ticker = st.sidebar.text_input('Enter Stock Ticker:')
        maxx_value=datetime.date.today()
        d = st.sidebar.date_input("Start date",datetime.date(2019, 7, 6),max_value=maxx_value)
        e= st.sidebar.date_input("End date",max_value=maxx_value)
       
       # start = st.sidebar.text_input('Start Date:')
        #end = st.sidebar.text_input('End Date:')
        yfin.pdr_override()

        df = pdr.get_data_yahoo(stock_ticker, start=d, end=e)

        # print(spy)
        # df = data.DataReader(stock_ticker, 'yahoo', start = d, end = e)
        # Describing Date
        st.subheader('Description of Stock:')
        st.write(df.describe())

        # Visualizations
        st.subheader('Closing Price vs Time chart:')
        fig = plt.figure(figsize = (12,6))
        # plt.plot(df.Close)
        st.line_chart(df.Close)

        st.subheader('Volume vs Time chart:')
        fig = plt.figure(figsize = (12,6))
        # plt.plot(df.Volume)
        st.line_chart(df.Volume)

        st.subheader('100 and 200 days moving average vs Time:')
        data = pd.DataFrame(
            {
                'ma1' : df.Close.rolling(100).mean(),
                'ma2' : df.Close.rolling(200).mean()
            },
            columns=['ma1', 'ma2'])
        fig = plt.figure(figsize = (12,6))
        # plt.plot(ma1, 'r')
        # plt.plot(ma2, 'g')
        st.line_chart(data)

        st.subheader('Volatility of Stock:')
        df['Log returns'] = np.log(df['Close']/df['Close'].shift())
        df['Log returns'].std()
        volatility = df['Log returns'].std()*252**.5
        str_vol = str(round(volatility, 4)*100)

        fig, ax = plt.subplots()
        df['Log returns'].hist(ax=ax, bins=50, alpha=0.6, color='b')
        ax.set_xlabel("Log return")
        ax.set_ylabel("Freq of log return")
        ax.set_title("AAPL volatility:" + str_vol + "%")
        # fig = plt.figure(figsize = (12,6))
        st.pyplot(fig)
    except Exception as e:
        warning = '<p style="color:Red; font-size: 30px; font-weight: bold;">Enter Details First!!</p>'
        st.markdown(warning, unsafe_allow_html=True)
else:
    try:
        stock_ticker = st.sidebar.selectbox('Select Stock Ticker:', options=['', 'AAPL', 'RS', 'BAJAJ-AUTO.NS', 'ADANIPORTS.NS', 'SBIN.NS', 'BHARTIARTL.NS'] )
        start =  (date.today() - relativedelta(years = 1)).strftime("%Y-%m-%d")
        end = date.today().strftime("%Y-%m-%d")
        n = st.sidebar.number_input('Enter no. of days of prediction:', step=1)
        yfin.pdr_override()

        df = pdr.get_data_yahoo(stock_ticker, start=start, end=end)
        # df = pdr.DataReader(stock_ticker, 'yahoo', start, end)

        df = df.reset_index()
        prediction_input = df[-100:]

        fun_inp = pd.DataFrame(prediction_input['Close'])
        fun_inp = scaler.fit_transform(fun_inp)
        inp = fun_inp
        fun_inp = fun_inp.reshape(1,100)
        tmp_inp = list(fun_inp)
        tmp_inp = tmp_inp[0].tolist()

        model = load_model('%s_final_model.h5'%(stock_ticker))

        lst_out = []
        n_steps = 100

        i = 0

        while(i<n):
            
            if(len(tmp_inp)>100):
                fun_inp = np.array(tmp_inp[1:])
                fun_inp = fun_inp.reshape(1,-1)
                fun_inp = fun_inp.reshape((1, n_steps, 1))
                yhat = model.predict(fun_inp, verbose = 0)
                tmp_inp.extend(yhat[0].tolist())
                tmp_inp = tmp_inp[1:]
                lst_out.extend(yhat.tolist())
                i = i+1
            else:
                fun_inp = fun_inp.reshape((1, n_steps, 1))
                yhat = model.predict(fun_inp, verbose = 0)
                tmp_inp.extend(yhat[0].tolist())
                lst_out.extend(yhat.tolist())
                i = i+1
                
        plot_new = np.arange(1, 101)
        plot_pred = np.arange(101, 101 + n)
        # plt.plot(plot_new, scaler.inverse_transform(inp))
        # plt.plot(plot_pred, scaler.inverse_transform(lst_out))

        inp = inp.tolist()
        inp.extend(lst_out)
        # plt.plot(inp)

        final = scaler.inverse_transform(inp).tolist()
        # plt.plot(final)
        plt.ylabel("Price")
        plt.xlabel("Time")
        plt.title("AAPL prediction ")
        # plt.axhline(y=final[len(final)-1], color = 'red', linestyle = ':', label = 'NEXT %d Days: %f'%(n, (round(float(*final[len(final)-1]),2))))
        plt.legend()    
        st.subheader('Prediction of Stock price:')                                                                                                            
        st.line_chart(final)
        final_price = final[-1]
        final_price = final_price[-1]
        predicted_date=datetime.date.today() + datetime.timedelta(n)
      
        final_data = 'Prediction of %s till %s is %f.'%(stock_ticker,predicted_date.strftime('%Y-%m-%d'), final_price)
        st.subheader(final_data)
    except Exception as e:
        warning = '<p style="color:Red; font-size: 30px; font-weight: bold;">Enter Details First!!</p>'
        st.markdown(warning, unsafe_allow_html=True)
