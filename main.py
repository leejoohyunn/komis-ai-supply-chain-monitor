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
    dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='M')
    
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

# 📈 메인 차트 섹션
st.markdown("### 🔸 실시간 수급 모니터링")

# KAN 모델 예측 결과 데이터 (첨부된 이미지 기반)
@st.cache_data
def get_kan_prediction_data():
    """KAN 모델의 백테스팅 결과 데이터"""
    
    # 2021년 데이터
    data_2021 = {
        'dates': ['2021-01', '2021-02', '2021-03', '2021-04', '2021-05', '2021-06'],
        'actual': [25.8, 17.3, 13.2, 21.5, 18.4, 12.1],
        'predicted': [26.8, 24.4, 21.6, 19.5, 16.6, 12.8]
    }
    
    # 2022년 데이터
    data_2022 = {
        'dates': ['2022-01', '2022-02', '2022-03', '2022-04', '2022-05', '2022-06'],
        'actual': [8.8, 7.4, 6.2, 6.3, 8.1, 7.6],
        'predicted': [9.4, 7.5, 6.1, 5.7, 6.9, 6.8]
    }
    
    # 2023년 데이터
    data_2023 = {
        'dates': ['2023-01', '2023-02', '2023-03', '2023-04', '2023-05', '2023-06'],
        'actual': [9.5, 9.6, 14.0, 19.9, 18.5, 25.9],
        'predicted': [8.5, 7.4, 9.6, 12.5, 13.5, 18.6]
    }
    
    return data_2021, data_2022, data_2023

# 데이터 로드
data_2021, data_2022, data_2023 = get_kan_prediction_data()

# 연도 선택 탭
tab1, tab2, tab3 = st.tabs(["2021년 상반기", "2022년 상반기", "2023년 상반기"])

with tab1:
    #st.markdown("#### 2021년 상반기 6개월 예측 결과 비교")
    fig_2021 = go.Figure()
    
    fig_2021.add_trace(go.Scatter(
        x=data_2021['dates'],
        y=data_2021['actual'],
        mode='lines+markers+text',
        name='실제 값 (Actual)',
        line=dict(color='blue', width=3),
        marker=dict(size=8, color='blue'),
        text=[f'{val:.1f}' for val in data_2021['actual']],
        textposition="top center",
        textfont=dict(size=10)
    ))
    
    fig_2021.add_trace(go.Scatter(
        x=data_2021['dates'],
        y=data_2021['predicted'],
        mode='lines+markers+text',
        name='예측 값 (Predicted)',
        line=dict(color='red', width=3, dash='dash'),
        marker=dict(size=8, color='red', symbol='x'),
        text=[f'{val:.1f}' for val in data_2021['predicted']],
        textposition="bottom center",
        textfont=dict(size=10)
    ))
    
    fig_2021.update_layout(
        title="2021년 상반기 6개월 예측 결과 비교",
        xaxis_title="예측 대상 연월",
        yaxis_title="수급안정화지수",
        height=400,
        showlegend=True,
        hovermode='x unified',
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    fig_2021.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig_2021.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    st.plotly_chart(fig_2021, use_container_width=True)

with tab2:
    #st.markdown("#### 2022년 상반기 6개월 예측 결과 비교")
    fig_2022 = go.Figure()
    
    fig_2022.add_trace(go.Scatter(
        x=data_2022['dates'],
        y=data_2022['actual'],
        mode='lines+markers+text',
        name='실제 값 (Actual)',
        line=dict(color='blue', width=3),
        marker=dict(size=8, color='blue'),
        text=[f'{val:.1f}' for val in data_2022['actual']],
        textposition="top center",
        textfont=dict(size=10)
    ))
    
    fig_2022.add_trace(go.Scatter(
        x=data_2022['dates'],
        y=data_2022['predicted'],
        mode='lines+markers+text',
        name='예측 값 (Predicted)',
        line=dict(color='red', width=3, dash='dash'),
        marker=dict(size=8, color='red', symbol='x'),
        text=[f'{val:.1f}' for val in data_2022['predicted']],
        textposition="bottom center",
        textfont=dict(size=10)
    ))
    
    fig_2022.update_layout(
        title="2022년 상반기 6개월 예측 결과 비교",
        xaxis_title="예측 대상 연월",
        yaxis_title="수급안정화지수",
        height=400,
        showlegend=True,
        hovermode='x unified',
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    fig_2022.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig_2022.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    st.plotly_chart(fig_2022, use_container_width=True)

with tab3:
    #st.markdown("#### 2023년 상반기 6개월 예측 결과 비교")
    fig_2023 = go.Figure()
    
    fig_2023.add_trace(go.Scatter(
        x=data_2023['dates'],
        y=data_2023['actual'],
        mode='lines+markers+text',
        name='실제 값 (Actual)',
        line=dict(color='blue', width=3),
        marker=dict(size=8, color='blue'),
        text=[f'{val:.1f}' for val in data_2023['actual']],
        textposition="top center",
        textfont=dict(size=10)
    ))
    
    fig_2023.add_trace(go.Scatter(
        x=data_2023['dates'],
        y=data_2023['predicted'],
        mode='lines+markers+text',
        name='예측 값 (Predicted)',
        line=dict(color='red', width=3, dash='dash'),
        marker=dict(size=8, color='red', symbol='x'),
        text=[f'{val:.1f}' for val in data_2023['predicted']],
        textposition="bottom center",
        textfont=dict(size=10)
    ))
    
    fig_2023.update_layout(
        title="2023년 상반기 6개월 예측 결과 비교",
        xaxis_title="예측 대상 연월",
        yaxis_title="수급안정화지수",
        height=400,
        showlegend=True,
        hovermode='x unified',
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    fig_2023.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig_2023.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    st.plotly_chart(fig_2023, use_container_width=True)

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
with st.sidebar:
    st.markdown("### 📊 실시간 시스템 상태")
    
    # 빠른 액션
    st.markdown("**⚡ 빠른 액션**")
    if st.button("📊 전체 데이터 새로고침", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    if st.button("📋 보고서 생성", use_container_width=True):
        with st.spinner("보고서 생성 중..."):
            time.sleep(2)
        st.success("보고서가 생성되었습니다!")
    
    if st.button("🔔 알림 설정", use_container_width=True):
        st.info("알림 설정 페이지로 이동합니다.")



# 푸터
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p><strong>RAG 기반 생성형 AI를 활용한 수급 리스크 예측 모델</strong></p>
    <p>🏆 산업통상자원부 제13회 공공데이터 활용 아이디어 공모전 출품작</p>
</div>
""", unsafe_allow_html=True)