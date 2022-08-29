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

def posterior_for_binom_and_uniform_prior(p, n_heads, n_trials):
    alpha_prior = 1
    beta_prior = 1
    alpha_post = alpha_prior + n_heads
    beta_post = beta_prior + (n_trials - n_heads)
    return stats.beta.pdf(p, alpha_post, beta_post)

def hpdi_for_binom_and_uniform_prior(hpdi, n_heads, n_trials):
    #todo: switch to built-in quantile functions?
    p_grid = np.linspace(start=0, stop=1, num=3001)
    p_posterior = np.array([posterior_for_binom_and_uniform_prior(p, n_heads, n_trials) for p in p_grid])
    norm = np.sum(p_posterior)
    n_start = np.argmax(p_posterior)
    n_left = n_start
    n_right = n_start
    s = p_posterior[n_start]
    while s < hpdi * norm:
        next_left = p_posterior[n_left - 1]
        next_right = p_posterior[n_right + 1]
        if next_left > next_right:
            n_left = n_left - 1
            s = s + next_left
        elif next_left < next_right:
            n_right = n_right + 1
            s = s + next_right
        else:
            n_left = n_left - 1
            n_right = n_right + 1
            s = s + next_left + next_right
    return(p_grid[n_left], p_grid[n_right])

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
                    #value=st.session_state['conv_a_exact'],
                    step=0.1,
                    format='%f',
                    key='conv_a_exact')

with col2:
    st.number_input(label='p_B Exact, %',
                    min_value=0.0,
                    max_value=100.0,
                    #value=st.session_state['conv_b_exact'],
                    step=0.1,
                    format='%f',
                    key='conv_b_exact')

st.number_input(label='Daily Users',
                min_value=0,
                #value=st.session_state['conv_daily_users'],
                step=100,
                format='%d',
                key='conv_daily_users')
st.number_input(label='N Days',
                min_value=0,
                #value=st.session_state['conv_n_days'],
                step=1,
                format='%d',
                key='conv_n_days')            
st.number_input(label='B Group Traffic Part, %',
                min_value=0.0,
                max_value=100.0,
                #value=st.session_state['conv_b_split'],
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


with st.spinner(text=f'Computing Conversions Interval Estimates ...'):
    hpdi = 0.9
    df[['p_hpdi_lower','p_hpdi_higher']] = df.apply(lambda row: pd.Series(hpdi_for_binom_and_uniform_prior(hpdi, row['conv_accum'], row['n_users_accum'])), axis=1)
    df['error_lower'] = df['p_accum'] - df['p_hpdi_lower']
    df['error_higher'] = df['p_hpdi_higher'] - df['p_accum']
    #display(df.head())

    #fig = px.line(df, x='day', y='p_accum', color='group', markers=True)
    #fig.update_layout(yaxis_rangemode='tozero')
    fig = go.Figure()
    col = 'blue'
    fig.add_trace(go.Scatter(x=df[df['group'] == 'A']['day'], 
                            y=df[df['group'] == 'A']['p_accum'],
                            mode='lines+markers', name='A', line_color=col))
    fig.add_trace(go.Scatter(x=pd.concat([df[df['group'] == 'A']['day'], df[df['group'] == 'A']['day'][::-1], df[df['group'] == 'A']['day'][0:1]]), 
                            y=pd.concat([df[df['group'] == 'A']['p_hpdi_higher'], df[df['group'] == 'A']['p_hpdi_lower'][::-1], df[df['group'] == 'A']['p_hpdi_higher'][0:1]]),
                            fill='toself', name=f'{hpdi:.0%} HPDI A',
                            hoveron = 'points+fills',
                            hoverinfo = 'text+x+y',
                            line_color=col, fillcolor=col, opacity=0.4))
    col = 'red'
    fig.add_trace(go.Scatter(x=df[df['group'] == 'B']['day'], 
                            y=df[df['group'] == 'B']['p_accum'],
                            mode='lines+markers', name='B', line_color=col))
    fig.add_trace(go.Scatter(x=pd.concat([df[df['group'] == 'B']['day'], df[df['group'] == 'B']['day'][::-1], df[df['group'] == 'B']['day'][0:1]]), 
                            y=pd.concat([df[df['group'] == 'B']['p_hpdi_higher'], df[df['group'] == 'B']['p_hpdi_lower'][::-1], df[df['group'] == 'B']['p_hpdi_higher'][0:1]]),
                            fill='toself', name=f'{hpdi:.0%} HPDI B',
                            hoveron = 'points+fills',
                            hoverinfo = 'text+x+y',
                            line_color=col, fillcolor=col, opacity=0.4))
    fig.update_layout(
        title='Accumulated Conversions',
        xaxis_title='Days',
        yaxis_title='P',
        yaxis_rangemode='tozero',
        hovermode="x")
    fig.update_layout(height=470)
    st.plotly_chart(fig)



st.subheader("Conversions Prob Density Estimates")

p_grid = np.linspace(start=0, stop=1, num=3001)
p_posterior_a = np.array([posterior_for_binom_and_uniform_prior(p, df[df['group'] == 'A']['conv_accum'].iloc[-1], df[df['group'] == 'A']['n_users_accum'].iloc[-1]) for p in p_grid])
p_posterior_b = np.array([posterior_for_binom_and_uniform_prior(p, df[df['group'] == 'B']['conv_accum'].iloc[-1], df[df['group'] == 'B']['n_users_accum'].iloc[-1]) for p in p_grid])

fig = go.Figure()
fig.add_trace(go.Scatter(x=p_grid, y=p_posterior_a, mode='lines', name='A', line_color='blue'))
fig.add_trace(go.Scatter(x=p_grid, y=p_posterior_b, mode='lines', name='B', line_color='red'))
fig.update_layout(title='Posterior',
                  xaxis_title='p',
                  yaxis_title='Prob Density',
                  hovermode="x")
#fig.update_layout(xaxis_range=[0, 0.1])
st.plotly_chart(fig)


st.markdown('---')
st.write("Get the sources at https://github.com/noooway/Coinflip")
st.write("See theory explanation at https://github.com/noooway/Bayesian_ab_testing")
