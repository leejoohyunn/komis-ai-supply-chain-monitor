# main.py - 메인 대시보드 (1페이지)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time

# 페이지 설정
st.set_page_config(
    page_title="RAG 기반 수급 리스크 예측 시스템",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #f0f8ff, #e6f3ff);
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    .alert-critical {
        background: linear-gradient(135deg, #ff6b6b, #ee5a52);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        animation: pulse 2s infinite;
    }
    
    .alert-warning {
        background: linear-gradient(135deg, #feca57, #ff9ff3);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    
    .success-box {
        background: linear-gradient(135deg, #5cb85c, #449d44);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .navigation-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
        transition: transform 0.3s ease;
    }
    
    .navigation-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
</style>
""", unsafe_allow_html=True)

# 🔄 세션 상태 초기화
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.current_risk_level = 70
    st.session_state.last_update = datetime.now()
    st.session_state.alert_count = 3

# 📊 샘플 데이터 생성 함수
@st.cache_data(ttl=3600)
def load_dashboard_data():
    """대시보드용 핵심 데이터 로드"""
    dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='ME')
    
    # 수급안정화지수 시뮬레이션
    supply_index = 50 + 15 * np.sin(np.arange(len(dates)) * 0.3) + np.random.normal(0, 5, len(dates))
    supply_index = np.clip(supply_index, 10, 90)
    
    # AI 리스크 지수 (수급안정화지수에 선행)
    ai_risk_index = supply_index + np.random.normal(0, 8, len(dates))
    ai_risk_index = np.roll(ai_risk_index, -2)  # 2개월 선행
    ai_risk_index = np.clip(ai_risk_index, 15, 95)
    
    return pd.DataFrame({
        'date': dates,
        'supply_stability_index': supply_index,
        'ai_risk_index': ai_risk_index,
        'price_volatility': np.random.uniform(0.1, 0.8, len(dates)),
        'geopolitical_tension': np.random.uniform(0.2, 0.9, len(dates))
    })

# 🎯 메인 헤더
st.markdown('<div class="main-header"> RAG 기반 생성형 AI를 활용한 수급 리스크 예측 모델 개발</div>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">산업통상자원부 제13회 공공데이터 활용 아이디어 공모전 출품작 </p>', unsafe_allow_html=True)
# st.markdown('<p style="text-align: center; font-size: 1.0rem; font-weight: bold; color: #666;"> AI기반 핵심광물 수급리스크 진단평가 모델 개발(데이터 분석 부문 과제 2) </p>', unsafe_allow_html=True)

# 📈 데이터 로드
data = load_dashboard_data()
current_risk = st.session_state.current_risk_level

# # 🚨 상단 알림 시스템
# col_alert1, col_alert2 = st.columns(2)

# with col_alert1:
#     if current_risk > 70:
#         st.markdown(f'''
#         <div class="alert-critical">
#             🚨 <strong>위험 단계</strong><br>
#             니켈 공급 위기 확률: {current_risk}%<br>
#             즉시 대응 방안 검토 필요
#         </div>
#         ''', unsafe_allow_html=True)
#     elif current_risk > 50:
#         st.markdown(f'''
#         <div class="alert-warning">
#             ⚠️ <strong>주의 단계</strong><br>
#             니켈 공급 위기 확률: {current_risk}%<br>
#             모니터링 강화 권장
#         </div>
#         ''', unsafe_allow_html=True)
#     else:
#         st.markdown(f'''
#         <div class="success-box">
#             ✅ <strong>안전 단계</strong><br>
#             니켈 공급 위기 확률: {current_risk}%<br>
#             정상 수준 유지 중
#         </div>
#         ''', unsafe_allow_html=True)

# with col_alert2:
#     st.markdown(f'''
#     <div style="background: #f8f9fa; padding: 1rem; border-radius: 10px; border-left: 4px solid #17a2b8;">
#         📊 <strong>시스템 현황</strong><br>
#         • 마지막 업데이트: {st.session_state.last_update.strftime("%Y-%m-%d %H:%M")}<br>
#         • AI 모델 정확도: 87.3%<br>
#         • 활성 알림: {st.session_state.alert_count}건<br>
#         • 데이터 동기화: 정상
#     </div>
#     ''', unsafe_allow_html=True)

# 📊 핵심 KPI 대시보드
#🔹🔸🧩▪️▫️☑️♦️✔️✴️✳️
# st.markdown("### 🔸 핵심 지표 대시보드")

# col1, col2, col3 = st.columns(3)

# with col1:
#     current_index = 52
#     delta_index = current_index - data['supply_stability_index'].iloc[-2]
#     st.metric(
#         "24년 5월 실제 수급안정화지수",
#         f"{current_index:.1f}",
#         delta=f"{delta_index:+.1f}",
#         help="KOMIS 공식 지수 (낮을수록 위험)"
#     )

# with col2:
#     ai_risk = 70
#     delta_ai = ai_risk - data['ai_risk_index'].iloc[-2]
#     st.metric(
#         "24년 5월 예측 수급안정화지수",
#         f"{ai_risk:.1f}",
#         delta=f"{delta_ai:+.1f}",
#         delta_color="inverse",
#         help="RAG 기반 지정학적 리스크 (높을수록 위험)"
#     )

# with col3:
#     prediction_accuracy = 55
#     st.metric(
#         "24년 5월 AI 리스크 점수",
#         f"{prediction_accuracy:.1f}",
#         delta="+2.1%p",
#         help="KAN 모델 MAPE 기준"
#     )

# with col4:
#     supply_disruption_prob = current_risk
#     st.metric(
#         "공급중단 확률",
#         f"{supply_disruption_prob}%",
#         delta="+12%p" if supply_disruption_prob > 50 else "-5%p",
#         delta_color="inverse",
#         help="향후 3개월 내 공급 중단 가능성"
#     )

# 📈 메인 차트 섹션# st.markdown("### 🔸 실시간 수급 모니터링")

# KAN 모델 예측 결과 데이터 (전체 시계열)
@st.cache_data
def get_kan_prediction_data():
    """KAN 모델의 전체 시계열 데이터"""
    
    # 실제 수급안정화지수 데이터 (파란색)
    actual_data = {
        '2018-01-01': 66.26, '2018-02-01': 52.38, '2018-03-01': 47.69, '2018-04-01': 48.57,
        '2018-05-01': 44.42, '2018-06-01': 41.73, '2018-07-01': 36.27, '2018-08-01': 46.92,
        '2018-09-01': 49.69, '2018-10-01': 56.57, '2018-11-01': 59.53, '2018-12-01': 69.12,
        '2019-01-01': 72.46, '2019-02-01': 65.48, '2019-03-01': 55.47, '2019-04-01': 52.35,
        '2019-05-01': 54.25, '2019-06-01': 60.78, '2019-07-01': 60.04, '2019-08-01': 48.04,
        '2019-09-01': 34.51, '2019-10-01': 25.11, '2019-11-01': 26.65, '2019-12-01': 37.26,
        '2020-01-01': 47.78, '2020-02-01': 60.25, '2020-03-01': 60.75, '2020-04-01': 68.00,
        '2020-05-01': 68.40, '2020-06-01': 62.49, '2020-07-01': 56.98, '2020-08-01': 49.71,
        '2020-09-01': 39.97, '2020-10-01': 37.75, '2020-11-01': 35.01, '2020-12-01': 29.33,
        '2021-01-01': 25.78, '2021-02-01': 17.27, '2021-03-01': 13.16, '2021-04-01': 21.46,
        '2021-05-01': 18.41, '2021-06-01': 12.08, '2021-07-01': 11.30, '2021-08-01': 9.07,
        '2021-09-01': 9.08, '2021-10-01': 8.79, '2021-11-01': 7.90, '2021-12-01': 10.16,
        '2022-01-01': 8.81, '2022-02-01': 7.40, '2022-03-01': 6.24, '2022-04-01': 6.34,
        '2022-05-01': 8.12, '2022-06-01': 7.59, '2022-07-01': 9.82, '2022-08-01': 16.42,
        '2022-09-01': 16.37, '2022-10-01': 16.64, '2022-11-01': 17.82, '2022-12-01': 11.87,
        '2023-01-01': 9.53, '2023-02-01': 9.59, '2023-03-01': 13.99, '2023-04-01': 19.88,
        '2023-05-01': 18.54, '2023-06-01': 25.90, '2023-07-01': 29.55, '2023-08-01': 28.00,
        '2023-09-01': 30.97, '2023-10-01': 31.98, '2023-11-01': 41.74, '2023-12-01': 47.40,
        '2024-01-01': 48.14, '2024-02-01': 52.14, '2024-03-01': 51.21, '2024-04-01': 44.00,
        '2024-05-01': 36.40, '2024-06-01': 27.85, '2024-07-01': 38.68, '2024-08-01': 44.04,
        '2024-09-01': 43.48, '2024-10-01': 43.79, '2024-11-01': 43.81
    }
    
    # AI 예측 데이터 (초록색 - 2025년 4-9월만)
    ai_predicted_data = {
        '2024-06-01': 31.01,
        '2024-07-01': 30.37,
        '2024-08-01': 34.89,
        '2024-09-01': 43.14,
        '2024-10-01': 49.83,
        '2024-11-01': 56.09
    }
    
    # 날짜를 pandas datetime으로 변환
    dates = pd.to_datetime(list(actual_data.keys()))
    actual_values = list(actual_data.values())
    
    # AI 예측 데이터의 날짜와 값 분리
    ai_dates = pd.to_datetime(list(ai_predicted_data.keys()))
    predicted_values = list(ai_predicted_data.values())
    
    return dates, actual_values, ai_dates, predicted_values

# 데이터 로드
dates, actual_values, ai_dates, predicted_values = get_kan_prediction_data()

# 전체 시계열 그래프
st.markdown("### 🔸 니켈 수급 안정화 지수 시계열 분석")

fig_timeseries = go.Figure()

# 실제 수급안정화지수 (파란색)
fig_timeseries.add_trace(go.Scatter(
    x=dates,
    y=actual_values,
    mode='lines+markers',
    name='실제 수급 안정화 지수',
    line=dict(color='blue', width=2),
    marker=dict(size=4, color='blue'),
    hovertemplate='<b>실제값</b><br>날짜: %{x}<br>지수: %{y:.2f}<extra></extra>'
))

# AI 예측 수급안정화지수 (초록색 - 2025년 4-9월만)
fig_timeseries.add_trace(go.Scatter(
    x=ai_dates,
    y=predicted_values,
    mode='lines+markers',
    name='AI 예측 수급 안정화 지수',
    line=dict(color='green', width=2, dash='dot'),
    marker=dict(size=4, color='green', symbol='circle'),
    hovertemplate='<b>AI 예측값</b><br>날짜: %{x}<br>지수: %{y:.2f}<extra></extra>'
))

# 최근 6개월 예측 구간 표시 (빨간색 X)
# recent_dates = dates[-6:]
# recent_predictions = [56.2, 49.8, 31.5, 30.2, 28.1, 44.2]  # 임시 예측값

# fig_timeseries.add_trace(go.Scatter(
#     x=recent_dates,
#     y=recent_predictions,
#     mode='markers',
#     name='최근 6개월 예측',
#     marker=dict(size=8, color='red', symbol='x'),
#     hovertemplate='<b>최근 예측</b><br>날짜: %{x}<br>지수: %{y:.2f}<extra></extra>'
# ))

fig_timeseries.update_layout(
    title="니켈 수급 안정화 지수 6개월 예측 및 분석 결과",
    xaxis_title="년",
    yaxis_title="수급안정화지수",
    height=500,
    showlegend=True,
    hovermode='x unified',
    plot_bgcolor='white',
    paper_bgcolor='white',
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="right",
        x=0.99
    )
)

# 그리드 추가
fig_timeseries.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
fig_timeseries.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

# 확대/축소 가능하도록 설정
fig_timeseries.update_layout(
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1년", step="year", stepmode="backward"),
                dict(count=2, label="2년", step="year", stepmode="backward"),
                dict(count=3, label="3년", step="year", stepmode="backward"),
                dict(step="all", label="전체")
            ])
        ),
        type="date"
    )
)

st.plotly_chart(fig_timeseries, use_container_width=True)

# 🎯 스마트 네비게이션
st.markdown("### 🔸 지능형 분석 도구")
st.markdown("분석 목적에 맞는 전문 도구를 선택하여 구체적인 분석을 확인하세요!")

col_nav1, col_nav2, col_nav3 = st.columns(3)

with col_nav1:
    st.markdown('''
    <div class="navigation-card">
        <h4>4차원 데이터 명세서</h4>
        <p>모든 데이터 소스를 한 곳에서 탐색</p>
        <ul>
            <li>사용한 데이터 분석</li>
            <li>4차원 수급 리스크 분석 체계</li>
            <li>4차원 데이터 상세 현황</li>
        </ul>
    </div>
    ''', unsafe_allow_html=True)
    
    if st.button("통합 데이터 센터", use_container_width=True):
        st.switch_page("pages/3_total_data_hub.py")

with col_nav2:
    st.markdown('''
    <div class="navigation-card">
        <h4>  RAG 시스템 체험</h4>
        <p>RAG 기반 지정학적 리스크 분석을 직접 체험해보세요</p>
        <ul>
            <li>월별 지수 분석</li>
            <li>RAG 상세 분석 결과</li>
            <li>JSON 출력 형식 분석</li>
        </ul>
    </div>
    ''', unsafe_allow_html=True)
    
    if st.button("RAG 시스템 체험하기", use_container_width=True):
        st.switch_page("pages/2_RAG_system_demo.py")

with col_nav3:
    st.markdown('''
    <div class="navigation-card">
        <h4>고급 상관관계 분석실</h4>
        <p>AI 변수 간의 숨겨진 관계를 발견하고 인과성을 분석</p>
        <ul>
            <li>그레인저 인과성 검정 결과</li>
            <li>주요 검정 결과</li>
            <li>해석 주의사항</li>
        </ul>
    </div>
    ''', unsafe_allow_html=True)
    
    if st.button("인과성 분석", use_container_width=True):
        st.switch_page("pages/5_advanced_correlation.py")

# 📊 사이드바 - 실시간 상태
# with st.sidebar:
#     st.markdown("### 📊 실시간 시스템 상태")
    
#     # 빠른 액션
#     st.markdown("**⚡ 빠른 액션**")
#     if st.button("📊 전체 데이터 새로고침", use_container_width=True):
#         st.cache_data.clear()
#         st.rerun()
    
#     if st.button("📋 보고서 생성", use_container_width=True):
#         with st.spinner("보고서 생성 중..."):
#             time.sleep(2)
#         st.success("보고서가 생성되었습니다!")
    
#     if st.button("🔔 알림 설정", use_container_width=True):
#         st.info("알림 설정 페이지로 이동합니다.")



# 푸터
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p><strong>RAG 기반 생성형 AI를 활용한 수급 리스크 예측 모델</strong></p>
    <p>🏆 산업통상자원부 제13회 공공데이터 활용 아이디어 공모전 출품작</p>
</div>
""", unsafe_allow_html=True)
