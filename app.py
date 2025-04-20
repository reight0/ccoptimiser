import streamlit as st
import pandas as pd
import numpy as np # For numerical operations, potentially handling NaN/inf
import pulp        # Import PuLP for optimization

# --- Page Config (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(layout="wide", page_title="Credit Card Optimiser")

# --- Configuration ---
DATA_FILE = 'CardDatabase.csv' # Your CSV file name
# Ensure these EXACTLY match your UI inputs and CSV categories
SPENDING_CATEGORIES = [
    "Dining", "Shopping", "Entertainment", "Groceries",
    "Transport", "Travel", "Petrol", "General/Other"
]
# Define the maximum number of bonus category pairs the code will check
MAX_BONUS_CATEGORIES = 8
# Define Fixed Max Cards Limit for Strategy
MAX_CARDS_STRATEGY_LIMIT = 8
BIG_M = 1000000 # A large number for Big M constraints
EPSILON = 0.01 # A small number for Big M constraints

# --- Data Loading ---
@st.cache_data # Cache data loading for performance
def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        # Basic data cleaning/type conversion
        bonus_cols = []
        for i in range(1, MAX_BONUS_CATEGORIES + 1):
            bonus_cols.append(f'BonusCategory{i}')
            bonus_cols.append(f'BonusRate{i} (%)')

        expected_cols = [
            'CardName', 'Issuer', 'RewardType', 'IncomeRequirement', 'RewardModelType',
            'BaseRate (%)'] + bonus_cols + [
            'BonusSpendCap (Monthly $)', 'BonusRewardCap (Monthly $)',
            'MinTotalSpendReq (Monthly $)', 'SpendThresholdX ($)', 'BaseRateBelowX (%)',
            'IsChoiceCard', 'ChoiceCategoryOptions', 'ChoiceBonusRate (%)',
            'MinSpendCategoryRequired', 'MinSpendCategoryValue ($)', 'Notes'
        ]

        actual_columns = df.columns.tolist()
        core_required_cols = ['CardName', 'Issuer', 'RewardType', 'RewardModelType', 'BaseRate (%)']
        missing_core_cols = [col for col in core_required_cols if col not in actual_columns]
        if missing_core_cols:
             print(f"CRITICAL ERROR: Missing essential columns in {filepath}: {', '.join(missing_core_cols)}. Cannot proceed.")
             return None

        extra_cols = [col for col in actual_columns if col not in expected_cols and col != 'IncomeRequirement']
        # if extra_cols:
        #      print(f"WARNING: Found potentially unexpected columns in {filepath}: {', '.join(extra_cols)}. Ensure required columns exist and are correct.")


        if 'IncomeRequirement' in df.columns:
             df['IncomeRequirement'] = pd.to_numeric(df['IncomeRequirement'], errors='coerce').fillna(0)
        df['RewardModelType'] = pd.to_numeric(df['RewardModelType'], errors='coerce')
        df = df.dropna(subset=['RewardModelType'])
        df['RewardModelType'] = df['RewardModelType'].astype(int)

        rate_cols = [col for col in df.columns if '(%)' in col]
        for col in rate_cols:
             if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

        numeric_cols = ['BonusSpendCap (Monthly $)', 'BonusRewardCap (Monthly $)',
                        'MinTotalSpendReq (Monthly $)', 'SpendThresholdX ($)',
                        'MinSpendCategoryValue ($)']
        for col in numeric_cols:
             if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        bool_cols = ['IsChoiceCard']
        for col in bool_cols:
             if col in df.columns:
                df[col] = df[col].apply(lambda x: str(x).strip().upper() == 'TRUE' if pd.notna(x) else False)

        df = df.dropna(subset=['CardName', 'BaseRate (%)', 'RewardType'])

        return df
    except FileNotFoundError:
        print(f"CRITICAL ERROR: {filepath} not found. Please make sure it's in the same folder as app.py.")
        return None
    except Exception as e:
        print(f"CRITICAL ERROR: Error loading or processing data: {e}")
        return None

# --- Core Individual Card Reward Calculation ---
def calculate_potential_reward(card_details, user_spending, total_monthly_spend):
    """ Calculates potential monthly reward for ranking. """
    card_name_debug = card_details.get('CardName', 'Unknown') if isinstance(card_details, (pd.Series, dict)) else 'InvalidInputType'
    # print(f"-- Processing Potential for Card: '{card_name_debug}'") # Keep commented out unless needed
    cards_to_debug = ["Lady's Card", "Live+ Card"] # Use the exact names from YOUR CSV

    if isinstance(card_details, pd.DataFrame): card_details = card_details.iloc[0] if not card_details.empty else None
    if card_details is None: return 0.0

    # Keep detailed prints conditional - enable if needed again
    # if card_name_debug in cards_to_debug: ...

    model_type = card_details.get('RewardModelType', 0); base_rate = card_details.get('BaseRate (%)', 0.0)
    min_total_spend_req = card_details.get('MinTotalSpendReq (Monthly $)', 0); min_total_spend_met = (total_monthly_spend >= min_total_spend_req)
    min_cat_req = card_details.get('MinSpendCategoryRequired', ''); min_cat_val = card_details.get('MinSpendCategoryValue ($)', 0); min_cat_spend_met = True
    if pd.notna(min_cat_req) and min_cat_req != '' and min_cat_req in user_spending: min_cat_spend_met = (user_spending.get(min_cat_req, 0) >= min_cat_val)
    applicable_base_rate = base_rate
    if model_type == 4:
        threshold = card_details.get('SpendThresholdX ($)', 0)
        if total_monthly_spend < threshold: applicable_base_rate = card_details.get('BaseRateBelowX (%)', 0.0)
    bonus_conditions_met = False
    if model_type in [2, 3, 6]: bonus_conditions_met = True;
    if min_total_spend_req > 0 and not min_total_spend_met: bonus_conditions_met = False
    elif model_type == 5: bonus_conditions_met = min_total_spend_met
    elif model_type == 7: bonus_conditions_met = min_cat_spend_met and min_total_spend_met

    potential_bonus_reward = 0.0; potential_base_reward = 0.0; spend_potentially_earning_bonus = 0.0; processed_bonus_categories = set()
    if bonus_conditions_met:
        if card_details.get('IsChoiceCard', False):
            potential_choices = [c.strip() for c in str(card_details.get('ChoiceCategoryOptions','')).split(';') if c.strip()]
            best_spend_in_choice = -1.0; chosen_cat_for_calc = None
            for choice in potential_choices:
                 choice = choice.strip()
                 if choice and choice in user_spending: spend = user_spending[choice];
                 if spend > best_spend_in_choice: best_spend_in_choice = spend; chosen_cat_for_calc = choice
            if chosen_cat_for_calc:
                 choice_rate = card_details.get('ChoiceBonusRate (%)', 0.0); potential_bonus_reward = best_spend_in_choice * choice_rate
                 spend_potentially_earning_bonus = best_spend_in_choice; processed_bonus_categories = {chosen_cat_for_calc}
        else:
            for i in range(1, MAX_BONUS_CATEGORIES + 1):
                cat_col = f'BonusCategory{i}'; rate_col = f'BonusRate{i} (%)'; category = card_details.get(cat_col, '')
                if pd.notna(category) and category != '' and category in user_spending:
                    rate = card_details.get(rate_col, 0.0); spend = user_spending[category]
                    if spend > 0.01: potential_bonus_reward += spend * rate; spend_potentially_earning_bonus += spend; processed_bonus_categories.add(category)
        bonus_spend_cap = card_details.get('BonusSpendCap (Monthly $)', float('inf')); bonus_reward_cap = card_details.get('BonusRewardCap (Monthly $)', float('inf'))
        bonus_spend_cap = float('inf') if bonus_spend_cap == 0 else bonus_spend_cap; bonus_reward_cap = float('inf') if bonus_reward_cap == 0 else bonus_reward_cap
        eligible_bonus_spend = min(spend_potentially_earning_bonus, bonus_spend_cap); potential_bonus_reward = min(potential_bonus_reward, bonus_reward_cap)
    if model_type == 1 or model_type == 4: potential_base_reward = total_monthly_spend * applicable_base_rate; potential_bonus_reward = 0
    else:
        if not bonus_conditions_met:
             if applicable_base_rate > 0: potential_base_reward = total_monthly_spend * applicable_base_rate; potential_bonus_reward = 0
        else:
            base_spend = 0
            for category, spend in user_spending.items():
                if category not in processed_bonus_categories: base_spend += spend
            potential_base_reward = base_spend * applicable_base_rate
    monthly_reward = potential_bonus_reward + potential_base_reward
    return monthly_reward

# --- Optimal Allocation Logic using PuLP ---
# [solve_optimal_allocation function remains the same - Snipped for brevity]
# --- Optimal Allocation Logic using PuLP ---
def solve_optimal_allocation(top_n_cards_df, user_spending_dict, total_monthly_spend):
    """ Solves for optimal spend allocation maximizing calculated reward using MIP. """
    card_details_dict = top_n_cards_df.set_index('CardName').to_dict('index')
    card_names = top_n_cards_df['CardName'].tolist()
    categories = list(user_spending_dict.keys())

    # 1. Create the MIP Problem
    prob = pulp.LpProblem("CreditCardOptimisationMIP", pulp.LpMaximize)

    # 2. Define Decision Variables
    spend_vars = pulp.LpVariable.dicts("Spend", (card_names, categories), lowBound=0, cat='Continuous')
    min_spend_met_vars = pulp.LpVariable.dicts("MinSpendMet", card_names, cat='Binary')
    choice_vars = pulp.LpVariable.dicts("ChoiceSelected", (card_names, categories), cat='Binary')
    reward_vars = pulp.LpVariable.dicts("Reward", (card_names, categories), lowBound=0, cat='Continuous')

    # 3. Define Objective Function: Maximize total reward
    prob += pulp.lpSum(reward_vars[k][c] for k in card_names for c in categories), "TotalReward"

    # 4. Define Constraints
    # Constraint 1: Allocate all spending
    for c in categories:
        prob += pulp.lpSum(spend_vars[k][c] for k in card_names) == user_spending_dict.get(c, 0), f"AllocateSpend_{c}"

    # Constraint 2 & 3: Minimum Spend Link (Big M)
    card_bonus_categories = {k: set() for k in card_names} # Define this early
    for k in card_names:
        details = card_details_dict[k]; min_req = details.get('MinTotalSpendReq (Monthly $)', 0)
        model_type = details.get('RewardModelType', 0); relies_on_min = model_type in [5, 7] # Check if bonus depends on min spend
        if relies_on_min and min_req > 0:
            total_spend_on_card_k = pulp.lpSum(spend_vars[k][c] for c in categories)
            prob += total_spend_on_card_k >= min_req * min_spend_met_vars[k], f"MinSpendMetLower_{k}"
            # Ensure y=0 if spend < min_req. M must be larger than max possible spend.
            M_val = total_monthly_spend + EPSILON # Use total spend as Big M here
            prob += total_spend_on_card_k <= (min_req - EPSILON) + M_val * min_spend_met_vars[k], f"MinSpendMetUpper_{k}"
        else: prob += min_spend_met_vars[k] == 1, f"MinSpendMetFixed_{k}" # Min spend effectively met

        # Also populate the bonus categories set while iterating
        if details.get('IsChoiceCard', False):
             choices = [c.strip() for c in str(details.get('ChoiceCategoryOptions','')).split(';') if c.strip()]
             card_bonus_categories[k].update(choices)
        else:
             for i in range(1, MAX_BONUS_CATEGORIES + 1):
                 cat = details.get(f'BonusCategory{i}', '')
                 if pd.notna(cat) and cat != '': card_bonus_categories[k].add(cat)

    # Constraint 4: Link Spend to Reward
    for k in card_names:
        details = card_details_dict[k]; model_type = details.get('RewardModelType', 0)
        # *** Calculate applicable_base_rate ONCE per card k (Handles Model 4) ***
        applicable_base_rate = details.get('BaseRate (%)', 0.0)
        if model_type == 4:
            # *** CORRECTED INDENTATION/SCOPE VERIFIED ***
            threshold = details.get('SpendThresholdX ($)', 0) # Define threshold safely here
            if total_monthly_spend < threshold: # Use threshold correctly indented
                applicable_base_rate = details.get('BaseRateBelowX (%)', 0.0)
        # *** End base rate calculation for card k ***

        for c in categories:
            is_potential_bonus = False; potential_bonus_rate = 0.0; is_choice_option = False
            if details.get('IsChoiceCard', False):
                potential_choices = [ch.strip() for ch in str(details.get('ChoiceCategoryOptions','')).split(';') if ch.strip()]
                if c in potential_choices:
                    potential_bonus_rate = details.get('ChoiceBonusRate (%)', 0.0); is_potential_bonus = True; is_choice_option = True;
                    # Constraints linking Reward to Choice Selection (w=choice_vars) and Min Spend (y=min_spend_met_vars)
                    prob += reward_vars[k][c] <= spend_vars[k][c] * potential_bonus_rate + BIG_M * (1 - choice_vars[k][c]) + BIG_M * (1- min_spend_met_vars[k])
                    # Removed LB constraint: prob += reward_vars[k][c] >= spend_vars[k][c] * potential_bonus_rate - BIG_M * (1 - choice_vars[k][c]) - BIG_M * (1- min_spend_met_vars[k])
                    prob += reward_vars[k][c] <= spend_vars[k][c] * applicable_base_rate + BIG_M * choice_vars[k][c]
                    # Removed LB constraint: prob += reward_vars[k][c] >= spend_vars[k][c] * applicable_base_rate - BIG_M * choice_vars[k][c]
                    # Keep LB >= Base always
                    prob += reward_vars[k][c] >= spend_vars[k][c] * applicable_base_rate, f"RewardLB_Base_Choice_{k}_{c}"


            if not is_choice_option:
                for i in range(1, MAX_BONUS_CATEGORIES + 1):
                    if details.get(f'BonusCategory{i}') == c: potential_bonus_rate = details.get(f'BonusRate{i} (%)', 0.0); is_potential_bonus = True; break
            if is_potential_bonus and not is_choice_option:
                # Constraints for Standard Bonus Categories (depend on min_spend_met_vars[k])
                prob += reward_vars[k][c] <= spend_vars[k][c] * potential_bonus_rate, f"RewardUB_Bonus_{k}_{c}"
                prob += reward_vars[k][c] <= spend_vars[k][c] * applicable_base_rate + BIG_M * min_spend_met_vars[k], f"RewardUB_BaseIfMinFail_{k}_{c}"
                prob += reward_vars[k][c] >= spend_vars[k][c] * applicable_base_rate, f"RewardLB_Base_{k}_{c}"
                # Removed LB constraint: prob += reward_vars[k][c] >= spend_vars[k][c] * potential_bonus_rate - BIG_M * (1 - min_spend_met_vars[k]), f"RewardLB_BonusIfMinMet_{k}_{c}"
            elif not is_potential_bonus and not is_choice_option:
                prob += reward_vars[k][c] == spend_vars[k][c] * applicable_base_rate, f"RewardEQ_Base_{k}_{c}"

    # Constraint 5: Bonus Spend Cap (Skip for Model 6)
    for k in card_names:
        details = card_details_dict[k]; spend_cap = details.get('BonusSpendCap (Monthly $)', 0)
        if spend_cap > 0:
            if not details.get('IsChoiceCard', False): # Apply only if NOT choice card
                bonus_spend_terms = []; effective_bonus_cats = set()
                for i in range(1, MAX_BONUS_CATEGORIES + 1): cat = details.get(f'BonusCategory{i}', '');
                if pd.notna(cat) and cat != '' and cat in categories and cat in card_bonus_categories[k]: effective_bonus_cats.add(cat)
                if effective_bonus_cats: bonus_spend_terms.extend(spend_vars[k][c] for c in effective_bonus_cats)
                if bonus_spend_terms: prob += pulp.lpSum(bonus_spend_terms) <= spend_cap, f"BonusSpendCap_{k}"

    # Constraint 6: Bonus Reward Cap
    for k in card_names:
        details = card_details_dict[k]; reward_cap = details.get('BonusRewardCap (Monthly $)', 0)
        if reward_cap > 0:
            effective_bonus_cats = card_bonus_categories[k] # Use set populated earlier
            if effective_bonus_cats: prob += pulp.lpSum(reward_vars[k][c] for c in effective_bonus_cats if c in categories) <= reward_cap, f"BonusRewardCap_{k}" # Ensure c is in categories

    # Constraint 7: Model 6 Choice Limit
    for k in card_names:
        details = card_details_dict[k]
        if details.get('IsChoiceCard', False):
             potential_choices = [c.strip() for c in str(details.get('ChoiceCategoryOptions','')).split(';') if c.strip()]
             if potential_choices: prob += pulp.lpSum(choice_vars[k][c] for c in potential_choices if c in categories) <= 1, f"ChoiceLimit_{k}"
             # Link spend var to choice var using Big M to ensure spend is only allocated if chosen? Not needed if reward handles it.
             # Ensure non-choice vars are 0
             for c in categories:
                 if c not in potential_choices: prob += choice_vars[k][c] == 0, f"ChoiceZero_{k}_{c}"
        else:
             for c in categories: prob += choice_vars[k][c] == 0, f"ChoiceZero_{k}_{c}"


    # 5. Solve the Problem
    try: solver = pulp.PULP_CBC_CMD(msg=0); prob.solve(solver)
    except pulp.PulpSolverError: st.error("PuLP Solver Error..."); return {}, {}, card_details_dict, {}
    except Exception as e: st.error(f"Optimization error: {e}"); st.exception(e); return {}, {}, card_details_dict, {}

    # 6. Process Results
    allocation = {k: {} for k in card_names}; card_monthly_rewards = {k: 0.0 for k in card_names}
    min_spend_met_final = {k: False for k in card_names}
    if pulp.LpStatus[prob.status] == 'Optimal':
        for k in card_names:
            min_spend_met_final[k] = (min_spend_met_vars[k].varValue > 0.5) if min_spend_met_vars[k] is not None else True
            card_total_monthly_reward = 0.0
            for c in categories:
                allocated_spend = spend_vars[k][c].varValue if spend_vars[k][c] is not None else 0.0
                calculated_reward = reward_vars[k][c].varValue if reward_vars[k][c] is not None else 0.0
                if allocated_spend > 0.01: allocation[k][c] = {'spend': allocated_spend, 'reward': max(0, calculated_reward)}; card_total_monthly_reward += max(0, calculated_reward)
            card_monthly_rewards[k] = card_total_monthly_reward
        return allocation, card_monthly_rewards, card_details_dict, min_spend_met_final
    else: st.error(f"Could not find optimal solution. Solver status: {pulp.LpStatus[prob.status]}"); return {}, {}, card_details_dict, {}
# --- END OF solve_optimal_allocation FUNCTION ---

# --- Load Data ---
card_data = load_data(DATA_FILE)
# --- Add DEBUG print for loaded data ---
if card_data is not None:
    print("\nDEBUG: Loaded Card Data (Relevant Rows):")
    cards_to_print = ["UOB Lady's Card", "Live+ Card", "Lady's Card"] # Add variations if needed
    relevant_cards_df = card_data[card_data['CardName'].isin(cards_to_print)]
    if not relevant_cards_df.empty: print(relevant_cards_df.to_markdown(index=False))
    else: print("Relevant cards not found in loaded data by specified names.")
else: print("\nDEBUG: card_data is None after loading.")
# --- End DEBUG print ---

# --- Initialize Session State ---
if 'page' not in st.session_state: st.session_state.page = 'caveats'

# --- Page Routing ---
if st.session_state.page == 'caveats':
    # [Caveats page code remains the same - Snipped]
    st.header("Dear Renee, your credit card strategy thing...."); st.markdown("strategy optimised based on hassle preference (number of cards), reward type preference, and spending habits.")
    st.write("---"); st.subheader("Some caveats and assumptions:")
    st.markdown("""1. Draft / V1 version - please pay if you want me to further develop this.\n2. Considers only a list* of entry level (30k income requirement) CCs - please pay if you want me to further expand.\n3. Does not consider potential advantages from using certain bank accounts with CCs.\n4. Assumes you are spending with the relevant merchants required by the CCs for bonus rates under the categories.\n5. Does not consider annual fees, specific welcome offers/sign-up bonuses, points transfer bonuses to partners, conversion fees, ease of reward redemption, specific merchant discounts/perks, card design, or customer service quality…..also domestic spending only.\n6. Information based on quick and dirty desktop scan - could be wrong.""") # Removed '\*'
    st.markdown("---"); st.subheader("*List of Credit Cards Considered in Database:")
    if card_data is not None and not card_data.empty:
        card_issuer_names = [];
        if 'Issuer' in card_data.columns and 'CardName' in card_data.columns:
             sorted_cards = card_data.sort_values(by=['Issuer', 'CardName']); card_issuer_names = [f"{row['Issuer']} - {row['CardName']}" for index, row in sorted_cards.iterrows()]
             st.caption("; ".join(card_issuer_names))
        else: st.error("Could not generate card list: 'Issuer' or 'CardName' column missing.")
    elif card_data is None: st.error("Could not load card list from database file.")
    else: st.warning("Card database loaded but appears empty.")
    st.markdown("---")
    if st.button("Proceed to Optimiser"): st.session_state.page = 'optimiser'; st.rerun()

elif st.session_state.page == 'optimiser':
    if card_data is None: st.title("Credit Card Optimiser - Error"); st.error("Card data failed to load."); st.stop()
    st.title("Credit Card Optimiser")
    st.sidebar.header("Your Constraints & Spending")
    max_cards_willingness = st.sidebar.number_input(f"Max Cards Willing to Manage (up to {MAX_CARDS_STRATEGY_LIMIT}):", min_value=1, max_value=MAX_CARDS_STRATEGY_LIMIT, value=2, step=1, key="max_cards_input")
    reward_preference = st.sidebar.selectbox("Reward Preference:", ["Best Overall Value (incl. Miles)", "Cashback Only"], index=0, key="reward_pref_input")
    st.sidebar.subheader("Estimated Monthly Spending ($):")
    user_spending_input = {}; total_monthly_spend_input = 0
    for category in SPENDING_CATEGORIES:
        default_val = 0.0; spend = st.sidebar.number_input(f"{category}:", min_value=0.0, step=10.0, value=float(default_val), key=f"spend_input_{category}", format="%.2f")
        user_spending_input[category] = spend; total_monthly_spend_input += spend
    st.sidebar.write(f"**Total Estimated Monthly Spend: ${total_monthly_spend_input:,.2f}**")
    st.header("Analysis & Recommendations")
    if total_monthly_spend_input <= 0.01: st.info("Enter spending in the sidebar.")
    else:
        if reward_preference == "Cashback Only":
            if 'RewardType' in card_data.columns: eligible_cards_df = card_data[card_data['RewardType'].astype(str).str.strip().str.upper() == 'CASHBACK'].copy()
            else: st.error("'RewardType' column missing..."); eligible_cards_df = card_data.copy();
        else: eligible_cards_df = card_data.copy()
        if eligible_cards_df.empty: st.error(f"No cards found matching criteria.")
        else:
            try:
                eligible_cards_df['PotentialMonthlyReward'] = eligible_cards_df.apply(lambda row: calculate_potential_reward(row, user_spending_input, total_monthly_spend_input), axis=1)
                eligible_cards_df['PotentialAnnualReward'] = eligible_cards_df['PotentialMonthlyReward'] * 12
            except Exception as e: st.error("Error during potential reward calculation."); st.exception(e); st.stop()
            ranked_cards_df = eligible_cards_df.sort_values(by='PotentialAnnualReward', ascending=False).reset_index(drop=True)
            num_cards_to_consider = min(max_cards_willingness, MAX_CARDS_STRATEGY_LIMIT, len(ranked_cards_df))
            if num_cards_to_consider <= 0: st.warning("No eligible cards found.")
            else:
                top_n_cards = ranked_cards_df.head(num_cards_to_consider).copy()
                if top_n_cards.empty or top_n_cards['PotentialAnnualReward'].sum() <= 0.01 : st.warning("No suitable cards found.")
                else:
                    st.subheader(f"Recommended Spending Allocation (Using up to {num_cards_to_consider} card(s)):")
                    try:
                        if not top_n_cards.empty:
                            allocation_details, card_monthly_rewards, card_details_dict_used, min_spend_met_final = solve_optimal_allocation( top_n_cards, user_spending_input, total_monthly_spend_input )
                            if not card_details_dict_used: pass # Error handled inside function
                            else:
                                total_annual_reward_allocated = sum(card_monthly_rewards.values()) * 12
                                st.success(f"**Estimated Total Annual Reward from Strategy: ${total_annual_reward_allocated:,.2f}**")
                                cards_used_in_strategy = [ name for name, allocations in allocation_details.items() if sum(data['spend'] for data in allocations.values()) > 0.01]
                                num_cards_actually_used = len(cards_used_in_strategy)
                                if num_cards_actually_used < num_cards_to_consider and num_cards_actually_used > 0 : st.info(f"Note: Optimal strategy found using {num_cards_actually_used} card(s), although up to {num_cards_to_consider} were considered.")
                                elif num_cards_actually_used == 0: st.warning("Solver ran, but allocated no spending.")
                                st.write("---")
                                card_rank = 0
                                for i in range(num_cards_to_consider):
                                    card_name = top_n_cards.iloc[i]['CardName']
                                    if card_name in cards_used_in_strategy:
                                        card_rank += 1; card_allocations = allocation_details.get(card_name, {}); card_annual_reward = card_monthly_rewards.get(card_name, 0) * 12; card_info = card_details_dict_used.get(card_name, {})
                                        st.markdown(f"**{card_rank}. 💳 {card_name}** (Issuer: {card_info.get('Issuer', 'N/A')})"); st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;Est. Annual Reward from this card: **${card_annual_reward:,.2f}**")
                                        min_req_final = card_info.get('MinTotalSpendReq (Monthly $)', 0); model_type_final = card_info.get('RewardModelType', 0); relies_on_min_final = model_type_final in [5, 7]
                                        if relies_on_min_final and min_req_final > 0 and not min_spend_met_final.get(card_name, False) :
                                            alloc_spend_val = sum(data['spend'] for data in card_allocations.values())
                                            st.warning(f"&nbsp;&nbsp;&nbsp;&nbsp;**Note:** Optimal allocation resulted in spend (${alloc_spend_val:,.2f}/mo) below the ${min_req_final:,.0f} minimum. Bonus rates did not apply; rewards reflect base rate.", icon="ℹ️")
                                        notes = card_info.get('Notes', '')
                                        if notes and pd.notna(notes): st.caption(f"&nbsp;&nbsp;&nbsp;&nbsp;Notes: {notes}")
                                        if card_allocations:
                                            for category, data in sorted(card_allocations.items()):
                                                if data['spend'] > 0.01:
                                                    reward_str = f"{data['reward']:.2f}" if data['reward'] >= 0.01 else f"{data['reward']:.4f}"
                                                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;**{category}:** Spend per month - ${data['spend']:,.2f} (Return per month - ${reward_str})", unsafe_allow_html=True)
                                        else: st.markdown("&nbsp;&nbsp;&nbsp;&nbsp;*(No specific spend allocated to this card)*", unsafe_allow_html=True)
                                        st.write("---")
                        else: st.warning("Top N cards list was empty.")
                    except pulp.PulpSolverError as plpe: st.error(f"PuLP Solver Error: {plpe}. Ensure PuLP/solver installed."); st.exception(plpe)
                    except Exception as e: st.error("An error occurred during optimal allocation."); st.exception(e)

elif 'page' in st.session_state and st.session_state.page != 'caveats':
     st.error("Something went wrong with page navigation.");
     if st.button("Go back to Start"): st.session_state.page = 'caveats'; st.rerun()