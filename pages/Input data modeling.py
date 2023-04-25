import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os.path
import gdown


st.set_page_config(
    page_title="PD Curves from input data",
    page_icon="📥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "# Modeling the Survival Embeddings and Default Probability with input data"
    }
)

DATA_PATH = "./data/lending_club_dataset_for_xgbse.csv"
MODEL_PATH = "./data/estimator.pkl"

@st.cache_data
def load_data(): 
    with st.spinner("Loading data..."):  

        if os.path.isfile(DATA_PATH) is False:
            with st.spinner("Downloading data from drive..."):
                url = "https://drive.google.com/uc?id=1j76gcHyVH3_Mwizlo6zzRXTIYXdTXAwF"
                output = DATA_PATH
                gdown.download(url, output, quiet = False)
                st.success("Data downloaded!")
                st.balloons()

        data = pd.read_csv(DATA_PATH, low_memory = False)
        data.drop('Unnamed: 0', axis = 1, inplace = True)
        return data
    
@st.cache_resource
def load_model():
    with st.spinner("Loading model..."):  

        if os.path.isfile(MODEL_PATH) is False:
            with st.spinner("Downloading model from drive..."):
                url = "https://drive.google.com/uc?id=1ZcnAxXa-lXoa5TpQpadWgSQFxTbV33Tj"
                output = MODEL_PATH
                gdown.download(url, output, quiet = False)
                st.success("Model downloaded!")
                st.balloons()

        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        return model
    

def plot_LSC(mean, upper_ci, lower_ci):
    fig, ax = plt.subplots()
    i = 0

    ax.plot(mean.columns, mean.iloc[i])
    ax.fill_between(mean.columns, lower_ci.iloc[i], upper_ci.iloc[i], alpha = 0.2)

    ax.set_title(f'Lifetime survival curve for entered credit')
    ax.set_ylabel('Survival probability')
    ax.set_xlabel('Month')
    return fig

def plot_LPD(mean, upper_ci, lower_ci):
    fig, ax = plt.subplots()

    i = 0
    mean = 1 - mean
    upper_ci = 1 - upper_ci
    lower_ci = 1 - lower_ci

    ax.plot(mean.columns, mean.iloc[i])
    ax.fill_between(mean.columns, lower_ci.iloc[i], upper_ci.iloc[i], alpha = 0.2)

    ax.set_title(f'Lifetime probability of default for entered credit')
    ax.set_ylabel('Probability of default')
    ax.set_xlabel('Month')
    return fig


def plot_LSC_with_streamlit(mean):
    df = pd.DataFrame([list(mean.iloc[0]), list(mean.columns)])
    df.reset_index()
    df_display = df.T.rename(columns={0:'Survival probability', 1:'Month'})
    st.line_chart(df_display, x = 'Month', y = 'Survival probability')

def plot_LPD_with_streamlit(mean):
    mean = 1 - mean
    df = pd.DataFrame([list(mean.iloc[0]), list(mean.columns)])
    df.reset_index()
    df_display = df.T.rename(columns={0:'Probability of default', 1:'Month'})
    st.line_chart(df_display, x = 'Month', y = 'Probability of default')


if 'data' not in st.session_state:
    data = load_data()
    N = data.shape[0]
    st.session_state.data = 1
    
if 'model' not in st.session_state:
    model = load_model()
    st.session_state.model = 1


st.title('Plot survival embeddings with input data')
st.divider()




st.header('Enter loan data for modeling')
with st.form("input_form"):
    st.subheader('General loan information')
    col1, col2, col3 = st.columns(3)
    with col1:
        loan_amnt = st.number_input("loan_amnt", help = "The listed amount of the loan applied for by the borrower", value = data['loan_amnt'].mean())
    with col2:
        annual_inc = st.number_input("annual_inc", help = "The self-reported annual income provided by the borrower during registration", value = data['annual_inc'].mean())
    with col3:
        recoveries = st.number_input("recoveries", help = "Post charge off gross recovery", value = data['recoveries'].mean())


    st.divider()
    st.subheader('Percent and ratio features')
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        dti = st.number_input("dti", help = "A ratio calculated using the borrower's total monthly debt payments on the total debt obligations, excluding mortgage and the requested LC loan, divided by the borrower's self-reported monthly income", value = data['dti'].mean())
    with col2:
        pct_tl_nvr_dlq = st.number_input("pct_tl_nvr_dlq", help = "Percent of trades never delinquent", value = data['pct_tl_nvr_dlq'].mean())
    with col3:
        percent_bc_gt_75 = st.number_input("percent_bc_gt_75", help = "Percentage of all bankcard accounts > 75% of limit", value = data['percent_bc_gt_75'].mean())
    with col4:
        revol_util = st.number_input("revol_util", help = "Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit", value = data['revol_util'].mean())


    st.divider()
    st.subheader('Past historical features')

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        pub_rec = st.number_input("pub_rec", help = "Number of derogatory public records", value = data['pub_rec'].mean())
    with col2:
        pub_rec_bankruptcies = st.number_input("pub_rec_bankruptcies", help = "Number of public record bankruptcies", value = data['pub_rec_bankruptcies'].mean())
    with col3:
        delinq_2yrs = st.number_input("delinq_2yrs", help = "The number of 30+ days past-due incidences of delinquency in the borrower's credit file for the past 2 years", value = data['delinq_2yrs'].mean())
    with col4:
        inq_last_6mths = st.number_input("inq_last_6mths", help = "The number of inquiries in past 6 months (excluding auto and mortgage inquiries)", value = data['inq_last_6mths'].mean())
    with col5:
        acc_open_past_24mths = st.number_input("acc_open_past_24mths", help = "Number of trades opened in past 24 months", value = data['acc_open_past_24mths'].mean())
        

    st.divider()
    st.subheader('Total features')

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        tot_coll_amt = st.number_input("tot_coll_amt", help = "Total collection amounts ever owed", value = data['tot_coll_amt'].mean())
    with col2:
        total_rev_hi_lim = st.number_input("total_rev_hi_lim", help = "Total revolving high credit limit", value = data['total_rev_hi_lim'].mean())
    with col3:
        tot_hi_cred_lim = st.number_input("tot_hi_cred_lim", help = "Total high credit/credit limit", value = data['tot_hi_cred_lim'].mean())
    with col4:
        total_bal_ex_mort = st.number_input("total_bal_ex_mort", help = "Total credit balance excluding mortgage", value = data['total_bal_ex_mort'].mean())
    with col5:
        total_bc_limit = st.number_input("total_bc_limit", help = "Total bankcard high credit/credit limit", value = data['total_bc_limit'].mean())
            

    st.divider()
    st.subheader('Accounts and trades features')

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        avg_cur_bal = st.number_input("avg_cur_bal", help = "Average current balance of all accounts", value = data['avg_cur_bal'].mean())
    with col2:
        mo_sin_old_il_acct = st.number_input("mo_sin_old_il_acct", help = "Months since oldest bank installment account opened", value = data['mo_sin_old_il_acct'].mean())
    with col3:
        mort_acc = st.number_input("mort_acc", help = "Number of mortgage accounts", value = data['mort_acc'].mean())
    with col4:
        num_il_tl = st.number_input("num_il_tl", help = "Number of installment accounts", value = data['num_il_tl'].mean())
    with col5:
        num_rev_tl_bal_gt_0 = st.number_input("num_rev_tl_bal_gt_0", help = "Number of revolving trades with balance >0", value = data['num_rev_tl_bal_gt_0'].mean())
    with col6:
        num_tl_90g_dpd_24m = st.number_input("num_tl_90g_dpd_24m", help = "Number of accounts 90 or more days past due in last 24 months", value = data['num_tl_90g_dpd_24m'].mean())

                    
    st.divider()
    st.subheader('Diff and avg features')

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        lines_acc_diff = st.number_input("lines_acc_diff", help = "Difference between total number of credit lines and number of open credit lines from the borrower", value = data['lines_acc_diff'].mean())
    with col2:
        num_rev_acc_diff = st.number_input("num_rev_acc_diff", help = "Difference between number of renewable accounts and number of open renewable accounts", value = data['num_rev_acc_diff'].mean())
    with col3:
        mo_tl_op_diff = st.number_input("mo_tl_op_diff", help = "Difference between number of bank card accounts and number of active bank card accounts", value = data['mo_tl_op_diff'].mean())
    with col4:
        mths_since_recent_diff = st.number_input("mths_since_recent_diff", help = "Difference between months from the opening of the earliest account and months from the opening of the last account", value = data['mths_since_recent_diff'].mean())
    with col5:
        num_sats_diff = st.number_input("num_sats_diff", help = "Difference between months since the last bank card account was opened and months since the last request", value = data['num_sats_diff'].mean())
    with col6:
        fico_range_avg = st.number_input("fico_range_avg", help = "Average FICO score", value = data['fico_range_avg'].mean())


    st.divider()
    st.subheader('Binary categorical features')

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        application_type = st.number_input("application_type", 0, 1, help = "Indicates whether the loan is an individual application or a joint application with two co-borrowers (1 = Joint App, 0 = Individual)")
    with col2:
        disbursement_method = st.number_input("disbursement_method", 0, 1, help = "The method by which the borrower receives their loan (1 = DIRECT_PAY, 0 = CASH)")
    with col3:
        debt_settlement_flag = st.number_input("debt_settlement_flag", 0, 1, help = "Flags whether or not the borrower, who has charged-off, is working with a debt-settlement company (1 = Y, 0 = N)")
    with col4:
        initial_list_status = st.number_input("initial_list_status", 0, 1, help = "The initial listing status of the loan. Possible values are – W, F (1 = F, 0 = W)")


    st.divider()
    st.subheader('Categorical features')

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        statuses = ('Not verified', 'Verified', 'Source verified')
        verification_status = st.selectbox("Verification status",
                                            statuses,
                                            help = "Indicates if income was verified by LC, not verified, or if the income source was verified")
        
    with col2:
        grade = st.selectbox("Grade",
                                ('A', 'B', 'C', 'D', 'E', 'F', 'G'),
                                help = "LC assigned loan grade")
    with col3:
        homes = ('OWN', 'RENT')
        home_ownership = st.selectbox("Home ownership",
                                        homes,
                                        help = "Client home ownership")
    with col4:
        purposes = ('Debt consolidation', 'Home improvement', 'Major purchase', 'Other')
        purpose = st.selectbox('Purpose', 
                                    purposes,
                                    help = "Credit purpose")
    with col5:
        states = ('North East', 'South East', 'South West', 'West')
        addr_state = st.selectbox("Geoposition (addr_state)",
                                    states,
                                    help = "Client state in the USA")

    plot_button = st.form_submit_button("Plot survival curve")
    
    grade_map = {
        "A": 7,
        "B": 6,
        "C": 5,
        "D": 4,
        "E": 3,
        "F": 2,
        "G": 1
    }

    if plot_button:
        loan_result = 0; duration = 0; grade = grade_map[grade]
        verification = statuses.index(verification_status)
        home_ownerships_list = list(map(lambda x: 1 if x == home_ownership else 0, homes))
        purposes_list = list(map(lambda x: 1 if x == purpose else 0, purposes))
        states_list = list(map(lambda x: 1 if x == addr_state else 0, states))

        

        df = pd.DataFrame(columns =
            ['loan_amnt', 'grade', 'annual_inc', 'verification_status', 'dti',
            'delinq_2yrs', 'inq_last_6mths', 'pub_rec', 'revol_util',
            'initial_list_status', 'recoveries', 'application_type', 'tot_coll_amt',
            'total_rev_hi_lim', 'acc_open_past_24mths', 'avg_cur_bal',
            'mo_sin_old_il_acct', 'mort_acc', 'num_il_tl', 'num_rev_tl_bal_gt_0',
            'num_tl_90g_dpd_24m', 'pct_tl_nvr_dlq', 'percent_bc_gt_75',
            'pub_rec_bankruptcies', 'tot_hi_cred_lim', 'total_bal_ex_mort',
            'total_bc_limit', 'disbursement_method', 'debt_settlement_flag',
            'loan_result', 'lines_acc_diff', 'num_rev_acc_diff', 'mo_tl_op_diff',
            'mths_since_recent_diff', 'num_sats_diff', 'fico_range_avg', 'duration',
            'home_ownership_OWN', 'home_ownership_RENT',
            'purpose_debt_consolidation', 'purpose_home_improvement',
            'purpose_major_purchase', 'purpose_other', 'addr_state_NorthEast',
            'addr_state_SouthEast', 'addr_state_SouthWest', 'addr_state_West'])
            
            
        df.loc[0] = [loan_amnt, grade, annual_inc, verification, dti,
            delinq_2yrs, inq_last_6mths, pub_rec, revol_util,
            initial_list_status, recoveries, application_type, tot_coll_amt,
            total_rev_hi_lim, acc_open_past_24mths, avg_cur_bal,
            mo_sin_old_il_acct, mort_acc, num_il_tl, num_rev_tl_bal_gt_0,
            num_tl_90g_dpd_24m, pct_tl_nvr_dlq, percent_bc_gt_75,
            pub_rec_bankruptcies, tot_hi_cred_lim, total_bal_ex_mort,
            total_bc_limit, disbursement_method, debt_settlement_flag,
            loan_result, lines_acc_diff, num_rev_acc_diff, mo_tl_op_diff,
            mths_since_recent_diff, num_sats_diff, fico_range_avg, duration,
            home_ownerships_list[0], home_ownerships_list[1],
            purposes_list[0], purposes_list[1], purposes_list[2], purposes_list[3], 
            states_list[0], states_list[1], states_list[2], states_list[3]]
        
        
        st.subheader('Modeling survival curves for entered credit data')
        st.dataframe(df)
        df = df.loc[[0]]
        with st.spinner("Modeling survival curve..."):  
            X = df.drop(['duration', 'loan_result'], axis = 1)
            mean, upper_ci, lower_ci = model.predict(X, return_ci = True)

            col1, col2 = st.columns(2)
            with col1:
                st.write(plot_LSC(mean, upper_ci, lower_ci))
                plot_LSC_with_streamlit(mean)
            with col2:
                st.write(plot_LPD(mean, upper_ci, lower_ci))
                plot_LPD_with_streamlit(mean)
