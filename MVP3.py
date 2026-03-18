import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm

# ==========================================
# 1. Data Layer (模擬市場價格與存貨數據)
# ==========================================

@st.cache_data
def generate_inventory_market_data():
    np.random.seed(42)
    days = 100
    dates = pd.date_range(end=pd.Timestamp.today(), periods=days)
    
    # 模擬三種不同波動特性的原物料價格 (使用幾何布朗運動概念簡化版)
    # 1. 銅導線 (Copper Wire) - 中等波動
    copper_returns = np.random.normal(0.0005, 0.015, days)
    copper_price = 100 * np.cumprod(1 + copper_returns)
    
    # 2. 半導體晶片 (Semiconductor Chips) - 高波動
    chip_returns = np.random.normal(0.001, 0.035, days)
    chip_price = 50 * np.cumprod(1 + chip_returns)
    
    # 3. 工業包裝紙箱 (Industrial Packaging) - 低波動
    paper_returns = np.random.normal(0.0001, 0.005, days)
    paper_price = 10 * np.cumprod(1 + paper_returns)
    
    market_data = pd.DataFrame({
        'Date': dates,
        'Copper_Wire': copper_price,
        'Semiconductor_Chips': chip_price,
        'Industrial_Packaging': paper_price
    })
    
    # 假設該 SME 倉庫裡的存貨數量
    inventory_quantities = {
        'Copper_Wire': 5000,          # 5000 單位
        'Semiconductor_Chips': 10000, # 10000 單位
        'Industrial_Packaging': 20000 # 20000 單位
    }
    
    return market_data, inventory_quantities

# ==========================================
# 2. Logic Layer (計量金融：VaR 與動態融資成數)
# ==========================================

def calculate_var_and_haircut(price_history, quantity, confidence_level=0.99, holding_period=10):
    # 計算每日報酬率
    returns = price_history.pct_change().dropna()
    
    # 統計學：計算日波動率 (Standard Deviation)
    daily_volatility = np.std(returns)
    
    # 計算最新存貨總市值 (Current Portfolio Value)
    current_price = price_history.iloc[-1]
    portfolio_value = current_price * quantity
    
    # 計量金融：參數法計算風險值 (Parametric VaR)
    # VaR = Portfolio_Value * Z * Volatility * sqrt(Holding_Period)
    z_score = norm.ppf(confidence_level)
    var_value = portfolio_value * z_score * daily_volatility * np.sqrt(holding_period)
    
    # 動態融資成數 (Dynamic Haircut) 邏輯
    # 基礎 Haircut 設為 20%，若 VaR 佔總價值的比例過高，則依比例增加 Haircut
    var_pct = var_value / portfolio_value
    base_haircut = 0.20
    dynamic_haircut = base_haircut + (var_pct * 1.5) # 風險懲罰係數 1.5
    
    # 限制 Haircut 最大不超過 80%
    dynamic_haircut = min(dynamic_haircut, 0.80)
    
    # 銀行願意核准的最大融資金額
    max_loan_amount = portfolio_value * (1 - dynamic_haircut)
    
    # 預警系統 (Early Warning Trigger)
    status = "Healthy (Low Risk)"
    status_color = "green"
    if dynamic_haircut > 0.50:
        status = "Margin Call / Reduce Exposure"
        status_color = "red"
    elif dynamic_haircut > 0.35:
        status = "Watchlist (Increasing Volatility)"
        status_color = "orange"
        
    return {
        'Current_Price': current_price,
        'Portfolio_Value': portfolio_value,
        'Daily_Volatility': daily_volatility,
        'VaR_Value': var_value,
        'Haircut': dynamic_haircut,
        'Max_Loan_Amount': max_loan_amount,
        'Status': status,
        'Status_Color': status_color
    }

# ==========================================
# 3. Presentation Layer (Streamlit 網頁介面)
# ==========================================

st.set_page_config(page_title="Inventory Financing MVP", layout="wide")
st.title("📦 Dynamic Inventory Valuation & Early Warning System")
st.markdown("Utilize **Value at Risk (VaR)** to dynamically adjust financing ratios for inventory-backed loans.")

# 載入資料
df_market, inventory_qty = generate_inventory_market_data()

# 佈局：左側參數與存貨選擇，右側視覺化與風險指標
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("1. Inventory Selection")
    
    # 選擇存貨種類
    asset_options = ['Copper_Wire', 'Semiconductor_Chips', 'Industrial_Packaging']
    selected_asset = st.selectbox("Select Pledged Inventory Asset:", asset_options)
    
    # 取得對應數值
    qty = inventory_qty[selected_asset]
    price_series = df_market[selected_asset]
    
    st.markdown("---")
    st.subheader("2. Real-Time Asset Status")
    
    # 呼叫計量風險引擎
    risk_metrics = calculate_var_and_haircut(price_series, qty)
    
    st.metric("Total Pledged Units", f"{qty:,} units")
    st.metric("Current Market Price (Per Unit)", f"${risk_metrics['Current_Price']:.2f}")
    st.metric("Total Inventory Value (Market to Market)", f"${risk_metrics['Portfolio_Value']:,.0f}")

with col2:
    st.subheader("3. Market Volatility & Price Trend (Last 100 Days)")
    
    # 繪製歷史價格走勢圖
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_market['Date'], y=price_series, mode='lines', name='Asset Price', line=dict(color='royalblue', width=2)))
    fig.update_layout(height=250, margin=dict(l=0, r=0, t=30, b=0), xaxis_title="Date", yaxis_title="Price (USD)")
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.subheader("4. Quant Risk Engine & Loan Decision")
    
    # 顯示狀態預警
    st.markdown(f"### System Alert Status: :{risk_metrics['Status_Color']}[{risk_metrics['Status']}]")
    
    r1, r2, r3 = st.columns(3)
    # 顯示年化波動率 (日波動率 * sqrt(252))
    ann_vol = risk_metrics['Daily_Volatility'] * np.sqrt(252)
    r1.metric("Annualized Volatility", f"{ann_vol * 100:.2f}%")
    
    # 顯示 10 天期 99% VaR
    r2.metric("10-Day VaR (99% Confidence)", f"${risk_metrics['VaR_Value']:,.0f}")
    
    # 顯示動態融資成數
    r3.metric("Dynamic Haircut Ratio", f"{risk_metrics['Haircut'] * 100:.1f}%")
    
    st.info(f"💡 **Financing Decision:** Based on the VaR risk assessment, the maximum allowable loan amount for this inventory is **${risk_metrics['Max_Loan_Amount']:,.0f}**.")