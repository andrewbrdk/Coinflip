import numpy as np
import pandas as pd
import scipy.stats as stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

#from numpy.random import default_rng
#rng = default_rng(17)
np.random.seed(7)   

def init_conv_session_values():
    if 'conv_a_exact' not in st.session_state:
        st.session_state['conv_a_exact'] = 15.0
    if 'conv_b_exact' not in st.session_state:
        st.session_state['conv_b_exact'] = 16.0
    if 'conv_daily_users' not in st.session_state:
        st.session_state['conv_daily_users'] = 3000
    if 'conv_n_days' not in st.session_state:
        st.session_state['conv_n_days'] = 15
    if 'conv_b_split' not in st.session_state:
        st.session_state['conv_b_split'] = 50.0
        
init_conv_session_values()
st.session_state

st.title('Conversions Comparison')

st.subheader("Simulate Data")

col1, col2 = st.columns(2)

with col1:
    st.number_input(label='p_A Exact, %',
                    min_value=0.0,
                    max_value=100.0,
                    value=st.session_state['conv_a_exact'],
                    step=0.1,
                    format='%f',
                    key='conv_a_exact')

with col2:
    st.number_input(label='p_B Exact, %',
                    min_value=0.0,
                    max_value=100.0,
                    value=st.session_state['conv_b_exact'],
                    step=0.1,
                    format='%f',
                    key='conv_b_exact')

st.number_input(label='Daily Users',
                min_value=0,
                value=st.session_state['conv_daily_users'],
                step=100,
                format='%d',
                key='conv_daily_users')
st.number_input(label='N Days',
                min_value=0,
                value=st.session_state['conv_n_days'],
                step=1,
                format='%d',
                key='conv_n_days')            
st.number_input(label='B Group Traffic Part, %',
                min_value=0.0,
                max_value=100.0,
                value=st.session_state['conv_b_split'],
                step=0.1,
                format='%f',
                key='conv_b_split')


trials = np.full(fill_value=st.session_state['conv_daily_users'],
                 shape=st.session_state['conv_n_days'])
conv_b_split = st.session_state['conv_b_split'] / 100
a_trials = np.rint(trials * (1 - conv_b_split)).astype(int)
b_trials = np.rint(trials * conv_b_split).astype(int)

conv_a_exact = st.session_state['conv_a_exact'] / 100
conv_b_exact = st.session_state['conv_b_exact'] / 100
a_trials_conv = stats.binom.rvs(n=a_trials, p=conv_a_exact)
b_trials_conv = stats.binom.rvs(n=b_trials, p=conv_b_exact)

df_exp = pd.concat([
    pd.DataFrame({'group': np.full(fill_value='A', shape=len(a_trials)),
                  'day': np.arange(len(a_trials)),
                  'n_users': a_trials,
                  'conv': a_trials_conv}),
    pd.DataFrame({'group': np.full(fill_value='B', shape=len(b_trials)),
                  'day': np.arange(len(b_trials)),
                  'n_users': b_trials,
                  'conv': b_trials_conv})
])
st.write(df_exp)

df = df_exp.copy()

st.subheader("Results")

st.write("Add Results Summary")


st.subheader("Traffic Split")

fig = px.line(df, x='day', y='n_users', color='group', markers=True)
fig.update_layout(yaxis_rangemode='tozero')
st.plotly_chart(fig)

df_accum2 = df.groupby(['group'], as_index=False)[['n_users', 'conv']].sum()
fig = px.bar(df_accum2, x='group', y='n_users', color='group')
st.plotly_chart(fig)

st.subheader("Daily Conversions")

df['p_daily'] = df['conv'] / df['n_users']
fig = px.line(df, x='day', y='p_daily', color='group', markers=True)
fig.update_layout(yaxis_rangemode='tozero')
st.plotly_chart(fig)


df_accum = df.groupby('group')[['n_users', 'conv']].cumsum().rename(columns={'n_users': 'n_users_accum', 'conv':'conv_accum'})
df = pd.concat([df, df_accum], axis=1)
df['p_accum'] = df['conv_accum'] / df['n_users_accum']
fig = px.line(df, x='day', y='p_accum', color='group', markers=True)
fig.update_layout(yaxis_rangemode='tozero')
st.plotly_chart(fig)


st.subheader("Conversions Prob Density Estimates")




st.markdown('---')
st.write("Get the sources at https://github.com/noooway/Coinflip")
st.write("See theory explanation at https://github.com/noooway/Bayesian_ab_testing")
