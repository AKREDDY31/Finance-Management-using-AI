import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
import numpy as np
from datetime import datetime

# Page Config
st.set_page_config(page_title="AI Personal Finance & Budget Coach", layout="wide")

# Load Data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("finance_data.csv")
    except FileNotFoundError:
        df = pd.DataFrame(columns=['Date', 'Type', 'Category', 'Amount'])
    return df

# Load existing data
df = load_data()

st.title("💰 AI Personal Finance & Budget Coach")

# Add New Entry
st.subheader("➕ Add Income or Expense")
with st.form("entry_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        entry_type = st.selectbox("Type", ["Income", "Expense"])
    with col2:
        category = st.text_input("Category", "Food")
    with col3:
        amount = st.number_input("Amount (₹)", min_value=0.0, format="%.2f")

    note = st.text_area("Optional Note")
    submitted = st.form_submit_button("Add Entry")

if submitted:
    new_entry = {"Date": datetime.now().strftime('%Y-%m-%d'), "Type": entry_type, "Category": category, "Amount": amount}
    df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
    df.to_csv("finance_data.csv", index=False)
    st.success("✅ Entry added successfully!")

# Summary Stats
income_df = df[df['Type'] == 'Income']
expense_df = df[df['Type'] == 'Expense']

total_income = income_df['Amount'].sum()
total_expense = expense_df['Amount'].sum()
current_balance = total_income - total_expense

col1, col2, col3 = st.columns(3)
col1.metric("Total Income", f"₹ {total_income:.2f}")
col2.metric("Total Expenses", f"₹ {total_expense:.2f}")
col3.metric("Current Balance", f"₹ {current_balance:.2f}")

# AI Saving Tips
st.subheader("🤖 AI Saving Tips & Financial Insights")

tips = []
if total_expense > total_income:
    tips.append("⚠️ Expenses exceed income. Consider reducing non-essentials.")

top_category = expense_df.groupby('Category')['Amount'].sum().idxmax() if not expense_df.empty else None
if top_category:
    top_amount = expense_df.groupby('Category')['Amount'].sum().max()
    if top_amount > (0.3 * total_expense):
        tips.append(f"🔍 High spending detected on **{top_category}**.")

savings_rate = (total_income - total_expense) / total_income if total_income > 0 else 0
if savings_rate < 0.2 and total_income > 0:
    tips.append("💡 Savings rate below 20%. Aim higher!")

tips.append("✅ Track expenses weekly for better control.")

for tip in tips:
    st.write(tip)

# Budget Planner
st.subheader("📝 Budget Planner")
monthly_budget = st.number_input("Set Your Monthly Budget (₹)", min_value=0, value=20000)

df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
current_month = datetime.now().month
monthly_expense = df[df['Date'].dt.month == current_month]['Amount'].sum()

st.write(f"🧾 This Month's Expenses: ₹ {monthly_expense:.2f}")
if monthly_expense > monthly_budget:
    st.error("🚨 Budget exceeded!")
elif monthly_expense > 0.8 * monthly_budget:
    st.warning("⚠️ Nearing budget limit.")
else:
    st.success("✅ Within budget.")

# Future Expense Prediction
st.subheader("📈 Future Expense Prediction")
df['Month'] = df['Date'].dt.to_period('M')
monthly_summary = df[df['Type'] == 'Expense'].groupby('Month')['Amount'].sum().reset_index()
monthly_summary['Month_Num'] = np.arange(len(monthly_summary))

if len(monthly_summary) > 1:
    model = LinearRegression()
    X = monthly_summary[['Month_Num']]
    y = monthly_summary['Amount']
    model.fit(X, y)

    next_month_num = [[monthly_summary['Month_Num'].max() + 1]]
    predicted_expense = model.predict(next_month_num)[0]

    st.write(f"🔮 Next Month's Predicted Expense: ₹ {predicted_expense:.2f}")
    fig = px.line(monthly_summary, x='Month', y='Amount', title="🗓 Monthly Expense Trend")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Need more data for prediction.")

# Savings Goal
st.subheader("🎯 Personalized Savings Goal")
goal_amount = st.number_input("Set Savings Goal (₹)", min_value=0, value=50000)
current_savings = current_balance
savings_progress = (current_savings / goal_amount) * 100 if goal_amount > 0 else 0

st.write(f"💰 Current Savings: ₹ {current_savings:.2f}")
st.progress(min(savings_progress / 100, 1.0))

if current_savings >= goal_amount:
    st.success("🎉 Goal Achieved!")
else:
    st.write(f"📊 {savings_progress:.2f}% of your goal completed.")

# AI Financial Chat
st.subheader("🤖 AI Financial Wellness Chat")
user_question = st.text_input("Ask about savings, investments, or expenses:")

if user_question:
    if "save" in user_question.lower():
        st.write("💡 Tip: Try the 50/30/20 rule for saving.")
    elif "investment" in user_question.lower():
        st.write("📈 Tip: Consider SIPs or Mutual Funds for stable growth.")
    elif "loan" in user_question.lower():
        st.write("🔍 Tip: Keep EMIs below 30% of income.")
    else:
        st.write("🤖 Sorry, I can't answer that yet. Try another question.")

# Display Data
st.subheader("📊 All Transactions")
st.dataframe(df.tail(10))
