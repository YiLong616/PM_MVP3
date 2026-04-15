import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from datetime import timedelta

# ==========================================
# 0. Page Configuration (頁面基礎設定)
# ==========================================
st.set_page_config(page_title="Dynamic Inventory Financing", layout="wide", page_icon="📦")

# ==========================================
# 1. Data Layer (資料層：模擬市場價格與存貨數據)
# ==========================================
@st.cache_data
def generate_inventory_market_data():
    """產生模擬的市場價格與存貨數量資料"""
    np.random.seed(42)
    days = 100
    dates = pd.date_range(end=pd.Timestamp.today(), periods=days)
    
    # 模擬三種不同波動特性的原物料價格
    # 1. Copper Wire (中等波動)
    copper_returns = np.random.normal(0.0005, 0.015, days)
    copper_price = 100 * np.cumprod(1 + copper_returns)
    
    # 2. Semiconductor Chips (高波動)
    chip_returns = np.random.normal(0.001, 0.035, days)
    chip_price = 50 * np.cumprod(1 + chip_returns)
    
    # 3. Industrial Packaging (低波動)
    paper_returns = np.random.normal(0.0001, 0.005, days)
    paper_price = 10 * np.cumprod(1 + paper_returns)
    
    market_data = pd.DataFrame({
        'Date': dates,
        'Copper_Wire': copper_price,
        'Semiconductor_Chips': chip_price,
        'Industrial_Packaging': paper_price
    })
    
    # 模擬倉庫存貨數量
    inventory_quantities = {
        'Copper_Wire': 5000,
        'Semiconductor_Chips': 10000,
        'Industrial_Packaging': 20000
    }
    
    return market_data, inventory_quantities

# ==========================================
# 2. Logic Layer (邏輯層：計量風險引擎與業務邏輯)
# ==========================================
def calculate_risk_engine(price_history, quantity, outstanding_loan, stress_drop_pct, conf_level=0.99, hold_days=10):
    """計算 VaR, CVaR, 動態融資成數與 Margin Call 狀態"""
    returns = price_history.pct_change().dropna()
    daily_vol = np.std(returns)
    
    # 目前價格與壓力測試後的價格
    current_price = price_history.iloc[-1]
    stressed_price = current_price * (1 - stress_drop_pct)
    portfolio_value = stressed_price * quantity
    
    # 計量模型：VaR (Value at Risk) 與 CVaR (Expected Shortfall)
    z_score = norm.ppf(conf_level)
    # 常態分配下的 VaR
    var_value = portfolio_value * z_score * daily_vol * np.sqrt(hold_days)
    # 常態分配下的 CVaR (比 VaR 更嚴格的尾部風險衡量)
    cvar_value = portfolio_value * (norm.pdf(z_score) / (1 - conf_level)) * daily_vol * np.sqrt(hold_days)
    
    # 業務邏輯：動態融資成數 (Dynamic Haircut)
    # 使用 CVaR 來決定風險懲罰，使銀行保護力更強
    risk_ratio = cvar_value / portfolio_value if portfolio_value > 0 else 0
    base_haircut = 0.20
    dynamic_haircut = base_haircut + (risk_ratio * 1.5)
    dynamic_haircut = min(dynamic_haircut, 0.85) # 最高扣減率 85%
    
    # 計算最大可貸金額
    max_loan_amount = portfolio_value * (1 - dynamic_haircut)
    
    # 判定 Margin Call 與狀態
    margin_call_amount = 0
    if outstanding_loan > max_loan_amount:
        status = "MARGIN CALL"
        status_color = "red"
        margin_call_amount = outstanding_loan - max_loan_amount
    elif dynamic_haircut > 0.40:
        status = "WATCHLIST"
        status_color = "orange"
    else:
        status = "HEALTHY"
        status_color = "green"
        
    return {
        'Current_Price': current_price,
        'Stressed_Price': stressed_price,
        'Portfolio_Value': portfolio_value,
        'Daily_Vol': daily_vol,
        'VaR': var_value,
        'CVaR': cvar_value,
        'Haircut': dynamic_haircut,
        'Max_Loan': max_loan_amount,
        'Margin_Call': margin_call_amount,
        'Status': status,
        'Status_Color': status_color
    }

# ==========================================
# 3. Presentation Layer (展示層：網頁介面)
# ==========================================

st.title("📦 Dynamic Inventory Valuation & Risk Engine")
st.markdown("Powered by **VaR/CVaR Quant Models** for Real-Time Supply Chain Finance.")

# 載入資料
df_market, inventory_qty = generate_inventory_market_data()

# --- 平台下拉式介紹與操作說明 ---
with st.expander("ℹ️ Platform Introduction & How to Use (Click to expand)", expanded=False):
    st.markdown("""
    ### Welcome to the Dynamic Inventory Risk Engine
    This platform helps banks and lenders dynamically assess the real-time value and risk of pledged inventory (e.g., raw materials, electronic components) using Wall Street-grade quantitative models.
    
    **How to Use:**
    1. **Select Asset:** Choose the type of inventory pledged by the SME in the sidebar.
    2. **Input Loan Info:** Enter the SME's current outstanding loan amount.
    3. **Stress Testing:** Use the slider to simulate extreme market crashes (e.g., Black Swan events) to see if the collateral can withstand the shock.
    4. **Monitor Dashboard:** - Check the **Gauge Chart** for the Dynamic Haircut.
       - Watch the **System Alert Status**; if it turns RED, a Margin Call is triggered.
       - Analyze the **Price Chart**, which includes a 10-day forward risk projection cone based on historical volatility.
    """)

st.markdown("---")

# --- 版面切割：左側輸入區，右側儀表板 ---
col_input, col_dash1, col_dash2 = st.columns([1, 1.5, 1.5])

with col_input:
    st.subheader("⚙️ Scenario Inputs")
    
    # 選擇資產
    asset_options = ['Copper_Wire', 'Semiconductor_Chips', 'Industrial_Packaging']
    selected_asset = st.selectbox("1. Select Pledged Asset", asset_options)
    
    qty = inventory_qty[selected_asset]
    price_series = df_market[selected_asset]
    current_val_est = price_series.iloc[-1] * qty
    
    # 輸入已借款金額 (預設為總價值的一半)
    st.markdown("<br>", unsafe_allow_html=True)
    outstanding_loan = st.number_input(
        "2. Current Outstanding Loan ($)", 
        min_value=0, 
        value=int(current_val_est * 0.5), 
        step=10000
    )
    
    # 壓力測試滑桿
    st.markdown("<br>", unsafe_allow_html=True)
    stress_drop = st.slider(
        "3. Market Stress Test (Price Drop %)", 
        min_value=0.0, max_value=50.0, value=0.0, step=1.0,
        help="Simulate an instant market crash to test collateral resilience."
    ) / 100.0

# 執行核心運算
metrics = calculate_risk_engine(price_series, qty, outstanding_loan, stress_drop)

with col_dash1:
    st.subheader("📊 Collateral Health")
    
    # 狀態與 Margin Call 警告特效
    if metrics['Status'] == 'MARGIN CALL':
        st.error(f"🚨 **{metrics['Status']} TRIGGERED!** Shortfall: **${metrics['Margin_Call']:,.0f}**")
    elif metrics['Status'] == 'WATCHLIST':
        st.warning(f"⚠️ **{metrics['Status']}**: High volatility detected. Monitor closely.")
    else:
        st.success(f"✅ **{metrics['Status']}**: Collateral value is sufficient.")
        
    # 核心財務指標
    st.metric("Total Pledged Units", f"{qty:,}")
    st.metric("Stressed Portfolio Value", f"${metrics['Portfolio_Value']:,.0f}", 
              delta=f"-{stress_drop*100}% Stress Applied" if stress_drop > 0 else None, 
              delta_color="inverse")
    st.metric("Max Allowable Loan", f"${metrics['Max_Loan']:,.0f}")

with col_dash2:
    st.subheader("⚖️ Risk Metrics")
    
    # 繪製動態融資成數 Gauge Chart
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = metrics['Haircut'] * 100,
        number = {'suffix': "%", 'valueformat': ".1f"},
        title = {'text': "Dynamic Haircut", 'font': {'size': 18}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "black"},
            'steps': [
                {'range': [0, 35], 'color': "#A3E4D7"}, # 淺綠 (安全)
                {'range': [35, 55], 'color': "#F9E79F"}, # 淺黃 (警戒)
                {'range': [55, 100], 'color': "#F5B7B1"} # 淺紅 (危險)
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': metrics['Haircut'] * 100
            }
        }
    ))
    fig_gauge.update_layout(height=220, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_gauge, use_container_width=True)
    
    # 顯示進階計量指標
    c1, c2 = st.columns(2)
    c1.metric("10-Day VaR (99%)", f"${metrics['VaR']:,.0f}")
    c2.metric("10-Day CVaR (99%)", f"${metrics['CVaR']:,.0f}")

st.markdown("---")

# --- 底部寬版區域：價格走勢與未來風險預測區間 ---
st.subheader("📈 Historical Trend & 10-Day Forward Risk Projection")

fig_price = go.Figure()

# 1. 歷史價格線
fig_price.add_trace(go.Scatter(
    x=df_market['Date'], y=price_series, 
    mode='lines', name='Historical Price', 
    line=dict(color='#2E86C1', width=2)
))

# 2. 建立未來 10 天的日期 (VaR Projection)
last_date = df_market['Date'].iloc[-1]
future_dates = [last_date + timedelta(days=i) for i in range(11)]

# 計算未來 10 天的預期價格下限 (使用 VaR 換算回單位價格)
unit_var_drop = metrics['VaR'] / qty
current_p = metrics['Current_Price']
# 為了視覺化，畫一個平滑的扇形擴散 (依照時間平方根遞增風險)
upper_bound = [current_p] * 11
lower_bound = [current_p - (unit_var_drop * np.sqrt(i/10)) for i in range(11)]

# 3. 畫出未來 10 天的 VaR 信心區間 (陰影)
fig_price.add_trace(go.Scatter(
    x=future_dates, y=upper_bound, 
    mode='lines', line=dict(width=0), 
    showlegend=False, hoverinfo='skip'
))
fig_price.add_trace(go.Scatter(
    x=future_dates, y=lower_bound, 
    mode='lines', name='99% Risk Floor (VaR)', 
    fill='tonexty', fillcolor='rgba(231, 76, 60, 0.2)', # 半透明紅色
    line=dict(color='#E74C3C', width=2, dash='dash')
))

fig_price.update_layout(
    height=350, 
    margin=dict(l=0, r=0, t=30, b=0),
    xaxis_title="Date", 
    yaxis_title="Price per Unit (USD)",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig_price, use_container_width=True)
