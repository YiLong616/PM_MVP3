import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from datetime import timedelta

# ==========================================
# 0. Page Configuration (頁面基礎設定)
# ==========================================
st.set_page_config(page_title="Strategic Inventory Finance Terminal", layout="wide", page_icon="🏢")

# ==========================================
# 1. Data Layer (資料層：模擬市場價格、存貨與營運數據)
# ==========================================
@st.cache_data
def generate_advanced_data():
    """產生模擬的市場價格、存貨數量與營運指標"""
    np.random.seed(42)
    days = 100
    dates = pd.date_range(end=pd.Timestamp.today(), periods=days)
    
    # 模擬三種不同波動特性的原物料價格
    assets = {
        'Copper_Wire': {'mu': 0.0005, 'sigma': 0.015, 'base': 100, 'qty': 5000},
        'Semiconductor_Chips': {'mu': 0.001, 'sigma': 0.035, 'base': 50, 'qty': 10000},
        'Industrial_Packaging': {'mu': 0.0001, 'sigma': 0.005, 'base': 10, 'qty': 20000}
    }
    
    data_dict = {'Date': dates}
    qty_dict = {}
    for name, param in assets.items():
        # 幾何布朗運動產生價格
        returns = np.random.normal(param['mu'], param['sigma'], days)
        data_dict[name] = param['base'] * np.cumprod(1 + returns)
        qty_dict[name] = param['qty']
        
    market_df = pd.DataFrame(data_dict)
    
    # 模擬營運細節數據 (Inventory Aging - 庫齡結構比重)
    # 分別代表: [0-30天, 31-90天, 90天以上]
    aging_data = {
        'Copper_Wire': [0.75, 0.20, 0.05], 
        'Semiconductor_Chips': [0.60, 0.30, 0.10],
        'Industrial_Packaging': [0.40, 0.40, 0.20]
    }
    
    return market_df, qty_dict, aging_data

# ==========================================
# 2. Logic Layer (邏輯層：營運調和風險引擎)
# ==========================================
def calculate_integrated_risk(
    price_history, qty, outstanding_loan, 
    po_coverage_pct, dsi_days, aging_weights,
    stress_drop, conf_level=0.99, hold_days=10
):
    """計算整合 VaR/CVaR 與營運指標的動態融資成數與狀態"""
    
    # --- Part A: Quant Market Risk (市場風險計量) ---
    returns = price_history.pct_change().dropna()
    daily_vol = np.std(returns)
    
    current_price = price_history.iloc[-1]
    stressed_price = current_price * (1 - stress_drop)
    portfolio_value = stressed_price * qty
    
    # 常態分配下的 VaR 與 CVaR
    z_score = norm.ppf(conf_level)
    var_value = portfolio_value * z_score * daily_vol * np.sqrt(hold_days)
    cvar_value = portfolio_value * (norm.pdf(z_score) / (1 - conf_level)) * daily_vol * np.sqrt(hold_days)
    
    # 基礎融資扣減率 (Base Haircut) - 使用嚴格的 CVaR 佔比來懲罰高波動資產
    risk_ratio = cvar_value / portfolio_value if portfolio_value > 0 else 0
    base_haircut = 0.20 + (risk_ratio * 1.5)
    
    # --- Part B: Operational Tuning (營運指標調和) ---
    # 1. PO Coverage Bonus (訂單覆蓋獎勵): 最高可減少 15% Haircut
    po_bonus = po_coverage_pct * 0.15 
    
    # 2. DSI Bonus (周轉獎勵): 假設行業基準為 60 天，低於 60 天給予獎勵，最高 10%
    dsi_bonus = 0
    if dsi_days < 60:
        dsi_bonus = ((60 - dsi_days) / 60) * 0.10 
        
    # 3. Aging Penalty (庫齡懲罰): 90天以上的老舊庫存佔比越高，懲罰越重，最高增加 30%
    old_stock_ratio = aging_weights[2] 
    aging_penalty = old_stock_ratio * 0.30 
    
    # 計算最終動態融資成數 (限制在 10% ~ 85% 之間)
    final_haircut = base_haircut - po_bonus - dsi_bonus + aging_penalty
    final_haircut = max(0.10, min(final_haircut, 0.85)) 
    
    # 計算最大可貸金額與 Margin Call
    max_loan = portfolio_value * (1 - final_haircut)
    margin_call_amount = max(0, outstanding_loan - max_loan)
    
    # 狀態判定
    if margin_call_amount > 0:
        status = "MARGIN CALL"
        status_color = "red"
    elif final_haircut > 0.45:
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
        'Base_Haircut': base_haircut,
        'Final_Haircut': final_haircut,
        'Max_Loan': max_loan,
        'Margin_Call': margin_call_amount,
        'PO_Bonus': po_bonus,
        'DSI_Bonus': dsi_bonus,
        'Aging_Penalty': aging_penalty,
        'DSI_Status': "Excellent" if dsi_days <= 45 else "Fair" if dsi_days <= 75 else "Poor",
        'Status': status,
        'Status_Color': status_color
    }

# ==========================================
# 3. Presentation Layer (展示層：網頁介面)
# ==========================================

# 載入資料
df_market, inventory_qty, aging_dict = generate_advanced_data()

st.title("🛡️ Strategic Inventory Valuation & Risk Engine")
st.markdown("Integrates **Market Volatility (VaR/CVaR)** with **Operational Health (DSI, PO Coverage, Aging)** for intelligent supply chain financing.")

# --- 平台下拉式介紹與操作說明 ---
with st.expander("ℹ️ Platform Introduction & Quick Start Guide", expanded=False):
    st.markdown("""
    ### 🎯 Core Concept
    This system calculates **Dynamic Financing Haircuts** by combining Wall Street risk models with your client's supply chain health.
    
    * **Market Risk (Baseline):** Uses VaR/CVaR to project worst-case price drops over 10 days.
    * **Business Health (Adjustments):**
        * ✅ **PO Coverage & Fast Turnover (DSI):** Reduces the haircut (High Liquidity).
        * ⚠️ **Stale Inventory (>90 days):** Increases the haircut (Obsolescence Risk).

    ---
    ### 🚀 Quick Start Guide
    
    **1. Setup Asset & Loan** (Sidebar)
    * Select the pledged asset and enter the current outstanding loan amount.
    
    **2. Input Business Health** (Sidebar)
    * **PO Coverage:** Slide to set confirmed orders. (Higher coverage = Lower risk)
    * **DSI (Turnover):** Enter average days to sell stock. (Fewer days = Lower risk)
    
    **3. Run Stress Test** (Sidebar)
    * Simulate a sudden market crash (e.g., 20% drop) to test if the collateral can withstand the shock.
    
    **4. Read the Dashboard** (Main Screen)
    * 🚨 **Status Alert:** Instantly checks if a Margin Call is triggered.
    * ⚖️ **Gauge Chart:** Shows the final algorithmically adjusted haircut.
    * 📉 **Risk Projection Chart:** The red cone visualizes the 99% VaR worst-case price floor for the next 10 days.
    """)

st.markdown("---")

# --- 側邊欄：情境輸入區 ---
st.sidebar.header("1. Asset & Loan Input")
asset_options = list(aging_dict.keys())
selected_asset = st.sidebar.selectbox("Select Pledged Asset", asset_options)

qty = inventory_qty[selected_asset]
price_series = df_market[selected_asset]
est_current_value = price_series.iloc[-1] * qty

outstanding_loan = st.sidebar.number_input(
    "Existing Outstanding Loan ($)", 
    min_value=0, 
    value=int(est_current_value * 0.4), # 預設借 4 成
    step=10000
)

st.sidebar.markdown("---")
st.sidebar.header("2. Business Operations Data")
po_cov = st.sidebar.slider("Confirmed PO Coverage (%)", 0, 100, 50) / 100.0
dsi = st.sidebar.number_input("Days Sales of Inventory (DSI)", value=40, help="Average days required to clear current inventory levels.")

st.sidebar.markdown("---")
st.sidebar.header("3. Extreme Stress Testing")
stress_drop = st.sidebar.slider(
    "Instant Market Crash (Price Drop %)", 
    0.0, 50.0, 0.0, 1.0,
    help="Simulate an instant Black Swan crash to test collateral resilience."
) / 100.0

# 執行核心運算
res = calculate_integrated_risk(
    price_series, qty, outstanding_loan, 
    po_cov, dsi, aging_dict[selected_asset], stress_drop
)

# --- 主畫面版面切割 ---
st.subheader("📊 Collateral Health & Financing Intelligence")

# 狀態警告區塊
if res['Status'] == 'MARGIN CALL':
    st.error(f"🚨 **MARGIN CALL TRIGGERED!** Shortfall to cover: **${res['Margin_Call']:,.0f}**")
elif res['Status'] == 'WATCHLIST':
    st.warning(f"⚠️ **WATCHLIST**: Elevated risk detected. Monitor collateral closely.")
else:
    st.success(f"✅ **HEALTHY**: Collateral value and operational health are sufficient.")

# 第一排：核心財務與營運指標
col1, col2, col3, col4 = st.columns(4)
col1.metric("Stressed Portfolio Value", f"${res['Portfolio_Value']:,.0f}", 
          delta=f"-{stress_drop*100:.0f}% Stress" if stress_drop > 0 else None, delta_color="inverse")
col2.metric("Max Allowable Loan", f"${res['Max_Loan']:,.0f}")
col3.metric("Inventory Turnover (DSI)", f"{dsi} Days", f"Liquidity: {res['DSI_Status']}")
col4.metric("PO Coverage Ratio", f"{po_cov*100:.0f}%", "Sales Guaranteed")

st.write("") # 增加適當間距

# 第二排：視覺化圖表與邏輯拆解
col_gauge, col_pie, col_logic = st.columns([1.2, 1, 1.2])

with col_gauge:
    # 動態融資成數 Gauge Chart
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = res['Final_Haircut'] * 100,
        number = {'suffix': "%", 'valueformat': ".1f"},
        title = {'text': "Final Dynamic Haircut", 'font': {'size': 16}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "black"},
            'steps': [
                {'range': [0, 35], 'color': "#A3E4D7"}, # 淺綠
                {'range': [35, 55], 'color': "#F9E79F"}, # 淺黃
                {'range': [55, 100], 'color': "#F5B7B1"} # 淺紅
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': res['Final_Haircut'] * 100
            }
        }
    ))
    fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=10))
    st.plotly_chart(fig_gauge, use_container_width=True)

with col_pie:
    # 庫齡結構 Pie Chart
    labels = ['0-30 Days (Fresh)', '31-90 Days (Normal)', '91+ Days (Stale)']
    values = aging_dict[selected_asset]
    fig_pie = go.Figure(data=[go.Pie(
        labels=labels, values=values, hole=.4, 
        marker_colors=['#27AE60', '#F1C40F', '#E74C3C']
    )])
    fig_pie.update_layout(
        title={'text': "Inventory Aging Profile", 'font': {'size': 16}, 'x': 0.5},
        height=250, margin=dict(l=10, r=10, t=40, b=10), showlegend=False
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col_logic:
    # 風險調整邏輯說明
    st.markdown("##### 📝 Haircut Adjustment Logic")
    st.info(f"""
    * **Base Market Risk (VaR):** `{res['Base_Haircut']*100:.1f}%`
    * **PO Coverage Reward:** `-{(res['PO_Bonus'])*100:.1f}%` (Reduces risk)
    * **High Turnover Reward:** `-{(res['DSI_Bonus'])*100:.1f}%` (Reduces risk)
    * **Stale Inventory Penalty:** `+{(res['Aging_Penalty'])*100:.1f}%` (Increases risk)
    """)
    st.markdown(f"**10-Day 99% CVaR Exposure:** `${res['CVaR']:,.0f}`")

st.markdown("---")

# --- 底部寬版區域：價格走勢與未來風險預測區間 ---
st.subheader("📈 Market Trend & 10-Day Forward Risk Projection")

fig_price = go.Figure()

# 1. 歷史價格線
fig_price.add_trace(go.Scatter(
    x=df_market['Date'], y=price_series, 
    mode='lines', name='Historical Price', 
    line=dict(color='#2E86C1', width=2)
))

# 2. 未來 10 天的日期 (VaR Projection Cone)
last_date = df_market['Date'].iloc[-1]
future_dates = [last_date + timedelta(days=i) for i in range(11)]

# 計算未來 10 天的預期價格下限 (使用 VaR 換算回單位價格)
unit_var_drop = res['VaR'] / qty
current_p = res['Current_Price']
upper_bound = [current_p] * 11
lower_bound = [current_p - (unit_var_drop * np.sqrt(i/10)) for i in range(11)]

# 3. 畫出未來 10 天的 VaR 信心區間 (紅色半透明陰影)
fig_price.add_trace(go.Scatter(
    x=future_dates, y=upper_bound, 
    mode='lines', line=dict(width=0), 
    showlegend=False, hoverinfo='skip'
))
fig_price.add_trace(go.Scatter(
    x=future_dates, y=lower_bound, 
    mode='lines', name='99% Risk Floor (VaR Cone)', 
    fill='tonexty', fillcolor='rgba(231, 76, 60, 0.2)',
    line=dict(color='#E74C3C', width=2, dash='dash')
))

# 4. 畫出壓力測試水平線
if stress_drop > 0:
    fig_price.add_hline(
        y=res['Stressed_Price'], line_dash="dot", line_color="purple", 
        annotation_text=f"Stress Floor (-{stress_drop*100:.0f}%)", 
        annotation_position="bottom right"
    )

fig_price.update_layout(
    height=350, 
    margin=dict(l=0, r=0, t=30, b=0),
    xaxis_title="Date", 
    yaxis_title="Price per Unit (USD)",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig_price, use_container_width=True)
