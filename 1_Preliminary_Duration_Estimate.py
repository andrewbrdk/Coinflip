import numpy as np
import pandas as pd
import scipy.stats as stats
import plotly.graph_objects as go
import streamlit as st

#from numpy.random import default_rng
#rng = default_rng(17)
np.random.seed(7)

#todo: move common functions to separate module
def pb_ge_pa(a_alpha, a_beta, b_alpha, b_beta, n_cmp=30000):
    pa = stats.beta.rvs(a_alpha, a_beta, size=n_cmp)
    pb = stats.beta.rvs(b_alpha, b_beta, size=n_cmp)
    return np.sum(pb >= pa) / n_cmp

def pb_ge_pa_sims(s_a, s_b, n_cmp=30000):
    n_pars = len(s_a['alpha_post'])
    pa = stats.beta.rvs(s_a['alpha_post'], s_a['beta_post'], size=(n_cmp, n_pars))
    pb = stats.beta.rvs(s_b['alpha_post'], s_b['beta_post'], size=(n_cmp, n_pars))
    return np.sum(pb >= pa, axis=0) / n_cmp   

def expected_conv_after_choice(pa_mean, pb_mean, pb_ge_pa, n_rest):
    return int(pa_mean * (1 - pb_ge_pa) * n_rest + pb_mean * pb_ge_pa * n_rest)

def simulate(p, trials, alpha, beta):
    trials_conv = stats.binom.rvs(n=trials, p=p)
    trials_accum = np.cumsum(trials)
    trials_conv_accum = np.cumsum(trials_conv)
    alpha_post = trials_conv_accum + alpha
    beta_post = (trials_accum - trials_conv_accum) + beta
    s = {
        'p': p,
        'trials_accum': trials_accum,
        'trials_conv_accum': trials_conv_accum,
        'alpha_post': alpha_post,
        'beta_post': beta_post
    }
    return s 

def min_n_to_reach_certainty_level(probs_pb_ge_pa, trials_accum, required_pb_ge_pa):
    prob_gt_required = (probs_pb_ge_pa > required_pb_ge_pa) | (probs_pb_ge_pa < 1 - required_pb_ge_pa)
    reached = trials_accum[prob_gt_required]
    min_reached = np.min(reached) if prob_gt_required[-1] else np.max(trials_accum)
    return min_reached

def conversions_on_choice(a_sim, b_sim, probs_pb_ge_pa, trials_accum, n_total_affected):
    pa_mean = stats.beta(a_sim['alpha_post'], a_sim['beta_post']).mean()
    pb_mean = stats.beta(b_sim['alpha_post'], b_sim['beta_post']).mean()
    after_choice = np.array([expected_conv_after_choice(pa_m, pb_m, pb_ge_pa, n_total_affected - n_exp) 
                              for pa_m, pb_m, pb_ge_pa, n_exp
                              in zip(pa_mean, pb_mean, probs_pb_ge_pa, trials_accum)])
    sim_and_expected_convs = after_choice + a_sim['trials_conv_accum'] + b_sim['trials_conv_accum']
    return sim_and_expected_convs    

def beta_dist_mean_std_to_alpha_beta(mean, std):
    var = std**2
    nu = mean * (1 - mean) / var - 1
    alpha = mean * nu
    beta = (1 - mean) * nu
    return alpha, beta

def init_session_values():
    if 'a_mean' not in st.session_state:
        st.session_state['a_mean'] = 15.0
    if 'a_std' not in st.session_state:
        st.session_state['a_std'] = 0.1
    if 'b_mean' not in st.session_state:
        st.session_state['b_mean'] = 16.0
    if 'b_std' not in st.session_state:
        st.session_state['b_std'] = 1.0
    if 'b_split' not in st.session_state:
        st.session_state['b_split'] = 50.0
    if 'pb_gt_pa_required' not in st.session_state:
        st.session_state['pb_gt_pa_required'] = 95.0
    if 'n_total_affected' not in st.session_state:
        st.session_state['n_total_affected'] = 10**6
    if 'sim_max' not in st.session_state:
        st.session_state['sim_max'] = 100000
    if 'sim_step' not in st.session_state:
        st.session_state['sim_step'] = 5000
    if 'n_simulations' not in st.session_state:
        st.session_state['n_simulations'] = 100
        
init_session_values()
#st.session_state

st.title('Preliminary Duration Estimate')

#todo: choose what to test: conversions, means, etc'
summary_container = st.container()
summary_container.write(f"""
    Base conversion: {st.session_state['a_mean']} +- {st.session_state['a_std']}%  
    Expected experimental conversion: {st.session_state['b_mean']} +- {st.session_state['b_std']}%    
    Experimental group traffic: {st.session_state['b_split']}%   
    Required P(p_experiment > p_base): {st.session_state['pb_gt_pa_required']}%
""")
summary_bar = summary_container.progress(0)

st.subheader("A Priori Conversions")
#todo: choose parametrization

col1, col2 = st.columns(2)

with col1:
    st.number_input(label='A Mean, %',
                    min_value=0.0,
                    max_value=100.0,
                    step=0.1,
                    format='%f',
                    key='a_mean')
    st.number_input(label='A Std, %',
                    min_value=0.01,
                    step=0.01,
                    format='%f',
                    key='a_std')

with col2:
    st.number_input(label='B Mean, %',
                    min_value=0.0,
                    max_value=100.0,
                    step=0.1,
                    format='%f',
                    key='b_mean')
    st.number_input(label='B Std, %',
                    min_value=0.01,
                    step=0.01,
                    format='%f',
                    key='b_std')

a_mean, a_std = st.session_state['a_mean']/100, st.session_state['a_std']/100
b_mean, b_std = st.session_state['b_mean']/100, st.session_state['b_std']/100
a_alpha, a_beta = beta_dist_mean_std_to_alpha_beta(a_mean, a_std)
b_alpha, b_beta = beta_dist_mean_std_to_alpha_beta(b_mean, b_std)

x = np.linspace(0, 1, 3001)
fig = go.Figure()
fig.add_trace(go.Scatter(x=x * 100, y=stats.beta.pdf(x, a_alpha, a_beta),
                         mode='lines',
                         name=f"A: mean = {a_mean * 100} %, std = {(a_std * 100):.2f} %"))
fig.add_trace(go.Scatter(x=x * 100, y=stats.beta.pdf(x, b_alpha, b_beta),
                         mode='lines',
                         name=f"B: mean = {b_mean * 100} %, std = {(b_std * 100):.2f} %"))
fig.update_layout(title='A Priori Conversions',
                  xaxis_title='Conversions, %',
                  yaxis_title='Prob Density',
                  hovermode="x",
                  height=550)
xrange_min = np.floor(np.min([a_mean - 5 * a_std, b_mean - 5 * b_std]) * 100)
xrange_max = np.ceil(np.max([a_mean + 5 * a_std, b_mean + 5 * b_std]) * 100)
fig.update_xaxes(range=[xrange_min, xrange_max])
st.plotly_chart(fig)


st.subheader("Duration Estimates")

st.number_input(label='B Group Traffic Part, %',
                min_value=0.0,
                step=1.0,
                format='%f',
                key='b_split')
b_split = st.session_state['b_split'] / 100

col1, col2 = st.columns(2)
with col1: 
    st.number_input(label='Max N in Simulations',
                    min_value=5000,
                    #value=st.session_state['sim_max'],
                    step=5000,
                    format='%d',
                    key='sim_max')
with col2: 
    st.number_input(label='N Step in Simulations',
                    min_value=1000,
                    #value=st.session_state['sim_step'],
                    step=1000,
                    format='%d',
                    key='sim_step')

n_simulations = st.number_input(label='Simulations',
                                min_value=1,
                                #value=st.session_state['n_simulations'],
                                step=1,
                                format='%d',
                                key='n_simulations')
n_sim_steps = st.session_state['sim_max'] // st.session_state['sim_step']

a_prior = stats.beta(a_alpha, a_beta)
b_prior = stats.beta(b_alpha, b_beta)
a_p_sim = a_prior.rvs(n_simulations)
b_p_sim = b_prior.rvs(n_simulations)

trials = np.append(0, np.full(fill_value=st.session_state['sim_step'], shape=n_sim_steps))
a_trials = np.rint(trials * (1 - b_split)).astype(int)
b_trials = np.rint(trials * b_split).astype(int)

i = 0
with st.spinner(text=f'Running {n_simulations} simulations ...'):
    my_bar = st.progress(0)
    sims = []
    for a_p, b_p in zip(a_p_sim, b_p_sim):
        s = {}
        s['A'] = simulate(a_p, a_trials, a_alpha, a_beta)
        s['B'] = simulate(b_p, b_trials, b_alpha, b_beta)
        s['pb_ge_pa'] = pb_ge_pa_sims(s['A'], s['B'], n_cmp=10000)
        s['N'] = s['A']['trials_accum'] + s['B']['trials_accum']
        s['pb_gt_pa_required'] = st.session_state['pb_gt_pa_required'] / 100
        s['min_n_to_reach_certainty_lvl'] = min_n_to_reach_certainty_level(s['pb_ge_pa'], s['N'], s['pb_gt_pa_required'])
        s['n_total_affected'] = st.session_state['n_total_affected']
        s['expected_conversions_on_choice'] = conversions_on_choice(s['A'], s['B'], s['pb_ge_pa'], s['N'], s['n_total_affected'])
        s['n_for_max_expected_conv'] = s['N'][np.argmax(s['expected_conversions_on_choice'])]
        sims.append(s)
        i = i + 1
        my_bar.progress(i / n_simulations)
        summary_bar.progress(i / n_simulations)
        #todo: update spinner text if i % 10 == 0: st.spinner(text=f'finished {i} of {n_simulations} simulations')
my_bar.empty()
summary_bar.empty()

n_reached_hist = [s['min_n_to_reach_certainty_lvl'] for s in sims]
n_max_hist = [s['n_for_max_expected_conv'] for s in sims]
summary_container.write(f"""
    Estimated experiment duration to reach P(p_experiment > p_base) = {st.session_state['pb_gt_pa_required']}: {np.mean(n_reached_hist)}  
    Duration for maximum expected conversions: {np.mean(n_max_hist)}  
""")

st.subheader("Duration Estimate for $P(p_B \ge p_A) = x$")

pb_gt_pa_required = st.number_input(label='Required P(p_B > p_A)',
                                    min_value=0.0,
                                    #value=st.session_state['pb_gt_pa_required'],
                                    step=1.0,
                                    format='%f',
                                    key='pb_gt_pa_required')
pb_gt_pa_required = pb_gt_pa_required / 100

st.write(f'Expected $N$ to reach $P(p_B \ge p_A)$ or $P(p_A \ge p_B) = {pb_gt_pa_required *100}$%: ${np.mean(n_reached_hist)}$')

fig = go.Figure()
fig.add_trace(go.Histogram(x=n_reached_hist, histnorm='probability', 
                           name='N to required P(p_b >= p_a) certainty', marker_color='red',
                           opacity=0.6))
fig.add_vline(x=np.mean(n_reached_hist), line_dash='dash')
fig.update_layout(title=f'N to reach P(p_b >= p_a) = {pb_gt_pa_required * 100} or P(p_a >= p_b) = {pb_gt_pa_required * 100} %',
                  xaxis_title='N',
                  yaxis_title='% from total simulations',
                  barmode='overlay')
st.plotly_chart(fig)

fig = go.Figure()
for s in sims:
    col = 'red' if (s['pb_ge_pa'][-1] > pb_gt_pa_required) or (s['pb_ge_pa'][-1] < 1 - pb_gt_pa_required) else 'blue'
    fig.add_trace(go.Scatter(x=s['N'], y=s['pb_ge_pa'],
                             mode='lines', line_color=col, opacity=0.2,
                             hovertemplate=f"a_p: {s['A']['p'] * 100:.1f} %, b_p = {s['B']['p'] * 100:.1f} %"))
fig.add_hline(y=pb_gt_pa_required, line_dash='dash')
fig.add_hline(y=1 - pb_gt_pa_required, line_dash='dash')
fig.update_layout(title='Simulations',
                  xaxis_title='N',
                  yaxis_title='P(p_b >= p_a)',
                  showlegend=False,
                  height=550)
st.plotly_chart(fig)

st.subheader("Duration Estimate for Maximum Expected Conversions")

st.number_input(label='Total Affected Users',
                min_value=1000,
                #value=st.session_state['n_total_affected'],
                step=1000,
                format='%d',
                key='n_total_affected')

st.write(f'Expected N to reach max ExpConv: {np.mean(n_max_hist)}')
#probs_at_nmax = [s['pb_ge_pa'][np.argmax(s['sim_and_expected_convs'])] for s in conv_sims]

fig = go.Figure()
fig.add_trace(go.Histogram(x=n_max_hist, histnorm='probability', 
                           name='N to required max ExpConv', marker_color='red',
                           opacity=0.6))
fig.add_vline(x=np.mean(n_max_hist), line_dash='dash')
fig.update_layout(title=f'N to reach max ExpConv',
                  xaxis_title='N',
                  yaxis_title='% from total simulations',
                  barmode='overlay')
st.plotly_chart(fig)


fig = go.Figure()
for s in sims:
    col = 'red'
    fig.add_trace(go.Scatter(x=s['N'], y=s['expected_conversions_on_choice'],
                             mode='lines', line_color=col, opacity=0.2,
                             hovertemplate=f"a_p: {s['A']['p'] * 100:.1f} %, b_p = {s['B']['p'] * 100:.1f} %"))
fig.update_layout(title='Simulations',
                  xaxis_title='N',
                  yaxis_title='Expected Conversions',
                  showlegend=False,
                  height=550)
st.plotly_chart(fig)

#todo: make simulations plots readable or remove them
st.markdown('---')
st.write("Get the sources at https://github.com/noooway/Coinflip")
st.write("See theory explanation at https://github.com/noooway/Bayesian_ab_testing")