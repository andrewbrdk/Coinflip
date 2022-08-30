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

def posterior_sample_for_binom_and_uniform_prior(ns, ntotal, n_sample):
    alpha_prior = 1
    beta_prior = 1
    a = alpha_prior + ns
    b = beta_prior + (ntotal - ns) 
    return np.random.beta(a, b, n_sample)

def alpha_beta_post(alpha, beta, n_conv, n_total):
    alpha_post = alpha + n_conv
    beta_post = beta + (n_total - n_conv)
    return alpha_post, beta_post 

def prob_pb_gt_pa(p_a, p_b, N_a, N_b = None):
    N_b = N_a if N_b is None else N_b
    n_sample = 30000
    post_sample_a = posterior_sample_for_binom_and_uniform_prior(p_a * N_a, N_a, n_sample)
    post_sample_b = posterior_sample_for_binom_and_uniform_prior(p_b * N_b, N_b, n_sample)
    post_sample_diff = post_sample_b - post_sample_a
    prob_b_gt_a = len(post_sample_diff[post_sample_diff > 0]) / len(post_sample_diff)
    return prob_b_gt_a

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
    if 'conv_pb_gt_pa_required' not in st.session_state:
        st.session_state['conv_pb_gt_pa_required'] = 95.0
    if 'conv_n_total_affected' not in st.session_state:
        st.session_state['conv_n_total_affected'] = 10**6
    if 'conv_sim_max' not in st.session_state:
        st.session_state['conv_sim_max'] = 100000
    if 'conv_sim_step' not in st.session_state:
        st.session_state['conv_sim_step'] = 5000
    if 'conv_n_simulations' not in st.session_state:
        st.session_state['conv_n_simulations'] = 100
        
init_conv_session_values()
st.session_state

st.title('Conversions Comparison')

st.subheader("Simulate Data")

col1, col2 = st.columns(2)

with col1:
    st.number_input(label='p_A Exact, %',
                    min_value=0.0,
                    max_value=100.0,
                    step=0.1,
                    format='%f',
                    key='conv_a_exact')

with col2:
    st.number_input(label='p_B Exact, %',
                    min_value=0.0,
                    max_value=100.0,
                    step=0.1,
                    format='%f',
                    key='conv_b_exact')

st.number_input(label='Daily Users',
                min_value=0,
                step=100,
                format='%d',
                key='conv_daily_users')
st.number_input(label='N Days',
                min_value=0,
                step=1,
                format='%d',
                key='conv_n_days')            
st.number_input(label='B Group Traffic Part, %',
                min_value=0.0,
                max_value=100.0,
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

summary_container = st.container()

st.subheader("Details")

df_summary = df_exp.groupby(['group'])[['n_users', 'conv']].sum()
df_summary['p'] = df_summary['conv'] / df_summary['n_users']
df_summary['col'] = pd.Series({'A': 'red', 'B':'blue'})
summary_container.write(df_summary)

df_plot = df_exp.set_index('group')

fig = make_subplots(rows=1, cols=2, 
                    column_widths=[0.75, 0.25],
                    subplot_titles=("Daily", "Total"))
for gr in df_plot.index.unique():
    fig.add_trace(
        go.Scatter(x=df_plot['day'][gr], y=df_plot['n_users'][gr], 
                   line_color=df_summary['col'][gr],
                   name=gr),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(x=[gr], y=[df_summary['n_users'][gr]],
               marker_color=df_summary['col'][gr],
               name=gr),
        row=1, col=2
    )
fig.update_layout(title_text='Total Users')
fig.update_xaxes(title_text="Days", row=1, col=1)
fig.update_yaxes(title_text="N Users", row=1, col=1)
fig.update_xaxes(title_text="Groups", row=1, col=2)
fig.update_layout(yaxis_rangemode='tozero')
st.plotly_chart(fig)


fig = make_subplots(rows=1, cols=2, 
                    column_widths=[0.75, 0.25],
                    subplot_titles=("Daily", "Total"))
for gr in df_plot.index.unique():
    fig.add_trace(
        go.Scatter(x=df_plot['day'][gr], y=df_plot['conv'][gr], 
                   line_color=df_summary['col'][gr],
                   name=gr),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(x=[gr], y=[df_summary['conv'][gr]],
               marker_color=df_summary['col'][gr],
               name=gr),
        row=1, col=2
    )
fig.update_layout(title_text='Converted Users')
fig.update_xaxes(title_text="Days", row=1, col=1)
fig.update_yaxes(title_text="N Converted", row=1, col=1)
fig.update_xaxes(title_text="Groups", row=1, col=2)
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




n_sample = 100000

post_sample_a = posterior_sample_for_binom_and_uniform_prior(df_summary['conv']['A'], df_summary['n_users']['A'], n_sample)
post_sample_b = posterior_sample_for_binom_and_uniform_prior(df_summary['conv']['B'], df_summary['n_users']['B'], n_sample)
post_sample_rel = post_sample_b / post_sample_a

fig = go.Figure()
fig.add_trace(go.Histogram(x=post_sample_rel, histnorm='probability density', 
                           name='B/A', marker_color='red',
                           opacity=0.6))
fig.add_vline(x=1, line_dash="dash")

fig.update_layout(title='B/A',
                  xaxis_title='B/A',
                  yaxis_title='Prob Density',
                  barmode='overlay')
st.plotly_chart(fig)

st.write(f"Expected(B/A) = {np.mean(post_sample_rel):.2f}")
summary_container.write(f"Expected(B/A) = {np.mean(post_sample_rel):.2f}")


pb_gt_pa = len(post_sample_rel[post_sample_rel > 1]) / len(post_sample_rel)
pa_gte_pb = len(post_sample_rel[post_sample_rel <= 1]) / len(post_sample_rel)

x = ['p_B <= p_A', 'p_B > p_A', ]
y = [pa_gte_pb, pb_gt_pa]
colors = ['red', 'green']

fig = go.Figure()
fig.add_trace(go.Bar(x=x, y=y, marker_color=colors, width=0.3))
fig.update_layout(yaxis_range=[0,1])
fig.update_layout(
     autosize=False,
     width=800,
     height=500)
st.plotly_chart(fig)

pb_gt_pa = len(post_sample_rel[post_sample_rel > 1]) / len(post_sample_rel)
st.write(f'P(p_B/p_A > 1): {pb_gt_pa}')
summary_container.write(f'P(p_B/p_A > 1): {pb_gt_pa}')


st.subheader("Daily P(p_B > p_A)")

pb_gt_pa_required = st.number_input(label='Required P(p_B > p_A)',
                                    min_value=0.0,
                                    step=1.0,
                                    format='%f',
                                    key='conv_pb_gt_pa_required')
pb_gt_pa_required = pb_gt_pa_required / 100

widedf = df.set_index(['group', 'day']).unstack(level=0)
#display(widedf.head())

widedf['pb_gt_pa'] = widedf.apply(lambda row: prob_pb_gt_pa(
    p_a=row['p_accum']['A'],
    p_b=row['p_accum']['B'],
    N_a=row['n_users_accum']['A'], 
    N_b=row['n_users_accum']['B']), axis=1)
widedf = widedf.reset_index()
#display(widedf.head())

fig = px.line(widedf, x='day', y='pb_gt_pa', markers=True,
              title="P(p_b >= p_a)",
              labels={"day": "Day", "pb_gt_pa": ""})
fig.add_hline(y=0.5, line_dash="dash")
fig.add_hline(y=pb_gt_pa_required, line_dash="dash")
fig.add_hline(y=1 - pb_gt_pa_required, line_dash="dash")
fig.update_layout(
    yaxis_range=[0, 1],
    yaxis_tickformat = ',.0%')
st.plotly_chart(fig)

st.subheader("Simulations")


col1, col2 = st.columns(2)
with col1: 
    st.number_input(label='Max N in Simulations',
                    min_value=5000,
                    step=5000,
                    format='%d',
                    key='conv_sim_max')
with col2: 
    st.number_input(label='N Step in Simulations',
                    min_value=1000,
                    step=1000,
                    format='%d',
                    key='conv_sim_step')

n_simulations = st.number_input(label='Simulations',
                                min_value=1,
                                step=1,
                                format='%d',
                                key='conv_n_simulations')
n_sim_steps = st.session_state['conv_sim_max'] // st.session_state['conv_sim_step']


a_alpha_post, a_beta_post = alpha_beta_post(alpha=1, beta=1,
                                            n_conv=df_summary['conv']['A'], 
                                            n_total=df_summary['n_users']['A'])
b_alpha_post, b_beta_post = alpha_beta_post(alpha=1, beta=1,
                                            n_conv=df_summary['conv']['B'], 
                                            n_total=df_summary['n_users']['B'])
a_post = stats.beta(a_alpha_post, a_beta_post)
b_post = stats.beta(b_alpha_post, b_beta_post)
a_p_sim = a_post.rvs(n_simulations)
b_p_sim = b_post.rvs(n_simulations)

trials = np.append(0, np.full(fill_value=st.session_state['conv_sim_step'], shape=n_sim_steps))
a_trials = np.rint(trials * (1 - conv_b_split)).astype(int)
b_trials = np.rint(trials * conv_b_split).astype(int)

i = 0
with st.spinner(text=f'Running {n_simulations} simulations ...'):
    my_bar = st.progress(0)
    sims = []
    for a_p, b_p in zip(a_p_sim, b_p_sim):
        s = {}
        s['A'] = simulate(a_p, a_trials, a_alpha_post, a_beta_post)
        s['B'] = simulate(b_p, b_trials, b_alpha_post, b_beta_post)
        s['pb_ge_pa'] = pb_ge_pa_sims(s['A'], s['B'], n_cmp=10000)
        s['N'] = s['A']['trials_accum'] + s['B']['trials_accum']
        s['pb_gt_pa_required'] = st.session_state['conv_pb_gt_pa_required'] / 100
        s['min_n_to_reach_certainty_lvl'] = min_n_to_reach_certainty_level(s['pb_ge_pa'], s['N'], s['pb_gt_pa_required'])
        s['n_total_affected'] = st.session_state['conv_n_total_affected']
        s['expected_conversions_on_choice'] = conversions_on_choice(s['A'], s['B'], s['pb_ge_pa'], s['N'], s['n_total_affected'])
        s['n_for_max_expected_conv'] = s['N'][np.argmax(s['expected_conversions_on_choice'])]
        sims.append(s)
        i = i + 1
        my_bar.progress(i / n_simulations)
        #todo: update spinner text if i % 10 == 0: st.spinner(text=f'finished {i} of {n_simulations} simulations')
my_bar.empty()

n_reached_hist = [s['min_n_to_reach_certainty_lvl'] for s in sims]
n_max_hist = [s['n_for_max_expected_conv'] for s in sims]
summary_container.write(f"""
    Estimated experiment duration to reach P(p_experiment > p_base) = {st.session_state['conv_pb_gt_pa_required']}: {np.mean(n_reached_hist)}  
    Duration for maximum expected conversions: {np.mean(n_max_hist)}  
""")

st.subheader("Duration Estimate for $P(p_B \ge p_A) = x$")

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
                step=1000,
                format='%d',
                key='conv_n_total_affected')

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



st.markdown('---')
st.write("Get the sources at https://github.com/noooway/Coinflip")
st.write("See theory explanation at https://github.com/noooway/Bayesian_ab_testing")
