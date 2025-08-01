# pages/5_📈_그레인저_인과성_분석.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
from datetime import datetime, timedelta

st.set_page_config(
    page_title="그레인저 인과성 분석",
    layout="wide"
)

# 커스텀 CSS
st.markdown("""
<style>
    .analysis-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    
    .insight-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .result-box {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("# 그레인저 인과성 분석")
st.markdown("AI 수급 리스크 지수와 KOMIS 수급안정화지수 간의 인과관계 분석")

# # 연구 배경 설명
# st.markdown("""
# <div class="analysis-card">
#     <h4>🔸 연구 배경</h4>
#     <p>본 분석에서는 RAG 기반 생성형 AI를 활용하여 개발한 <strong>'AI 수급 리스크 지수'</strong>가 
#     기존의 KOMIS 수급안정화지수와 어떤 관계를 가지는지 통계적으로 검정하였습니다.</p>
    
#     <p><strong>그레인저 인과성 검정</strong>은 한 시계열 자료가 다른 시계열 자료의 미래값 예측에 
#     도움이 되는지를 통계적으로 확인하는 방법입니다.</p>
# </div>
# """, unsafe_allow_html=True)

# # 분석용 데이터 생성 (보고서 결과 기반)
# @st.cache_data
# def generate_granger_data():
#     """그레인저 검정 결과 데이터 생성 (보고서 기반)"""
#     # 보고서의 그레인저 검정 결과를 반영한 시뮬레이션 데이터
#     dates = pd.date_range(start='2018-01-01', end='2024-04-30', freq='M')
#     np.random.seed(42)
    
#     # AI 수급 리스크 지수 (선행지표)
#     ai_risk_index = 50 + 20 * np.sin(np.arange(len(dates)) * 0.1) + np.random.normal(0, 8, len(dates))
    
#     # KOMIS 수급안정화지수 (후행지표 - AI지수에 의해 영향받음)
#     komis_index = np.zeros(len(dates))
#     komis_index[0] = 50
    
#     for i in range(1, len(dates)):
#         # AI 지수의 영향을 3-6개월 시차로 반영
#         lag_effect = 0
#         for lag in range(1, min(7, i+1)):
#             if i-lag >= 0:
#                 lag_effect += 0.15 * (ai_risk_index[i-lag] - 50) * (0.8 ** (lag-1))
        
#         komis_index[i] = 0.7 * komis_index[i-1] + 0.3 * (50 + lag_effect) + np.random.normal(0, 5)
    
#     return pd.DataFrame({
#         'date': dates,
#         'ai_risk_index': ai_risk_index,
#         'komis_index': komis_index
#     })

# # 데이터 로드
# granger_data = generate_granger_data()

# # 시계열 데이터 시각화
# st.markdown("## 🔸 시계열 데이터 비교")

# fig_timeseries = go.Figure()

# fig_timeseries.add_trace(go.Scatter(
#     x=granger_data['date'],
#     y=granger_data['ai_risk_index'],
#     name='AI 수급 리스크 지수',
#     line=dict(color='blue', width=2)
# ))

# fig_timeseries.add_trace(go.Scatter(
#     x=granger_data['date'],
#     y=granger_data['komis_index'],
#     name='KOMIS 수급안정화지수',
#     line=dict(color='orange', width=2)
# ))

# fig_timeseries.update_layout(
#     title="AI 수급 리스크 지수 vs KOMIS 수급안정화지수",
#     xaxis_title="날짜",
#     yaxis_title="지수 값",
#     height=500,
#     legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
# )

# st.plotly_chart(fig_timeseries, use_container_width=True)

# 그레인저 인과성 검정 결과
st.markdown("### 🔸 그레인저 인과성 검정 결과")

# 이미지 중앙 정렬을 위한 컬럼 사용
col_left, col_center, col_right = st.columns([1, 2, 1])

with col_center:
    try:
        st.image("../images/image3.png", caption="그레인저 인과성 검정 결과", use_container_width=True)
    except:
        st.error("image3.png 파일을 찾을 수 없습니다.")


# col1, col2 = st.columns([2, 1])

# with col1:
#     # 보고서의 그레인저 검정 결과를 반영한 그래프
#     lags = list(range(1, 13))  # 1개월부터 12개월까지
    
#     # 보고서에 따르면 12개월까지 모든 시차에서 p < 0.05
#     p_values = [0.01, 0.008, 0.012, 0.015, 0.009, 0.007, 
#                 0.018, 0.011, 0.014, 0.016, 0.013, 0.019]
    
#     fig_granger = go.Figure()
    
#     fig_granger.add_trace(go.Scatter(
#         x=lags,
#         y=p_values,
#         mode='lines+markers',
#         name='p-값',
#         line=dict(color='blue', width=3),
#         marker=dict(size=8, color='blue')
#     ))
    
#     # 유의수준 선 추가
#     fig_granger.add_hline(
#         y=0.05, 
#         line_dash="dash", 
#         line_color="red",
#         annotation_text="유의수준 (α=0.05)"
#     )
    
#     fig_granger.update_layout(
#         title="그레인저 인과성 검정: AI 수급 리스크 지수 → KOMIS 수급안정화지수",
#         xaxis_title="시차 (개월)",
#         yaxis_title="p-값",
#         height=400,
#         showlegend=False
#     )
    
#     st.plotly_chart(fig_granger, use_container_width=True)

# with col2:
#     st.markdown("### 검정 결과 요약")
    
#     # 시차별 결과 테이블
#     granger_results = pd.DataFrame({
#         '시차(개월)': lags,
#         'p-값': [f"{p:.3f}" for p in p_values],
#         '유의성': ['유의함' if p < 0.05 else '유의하지 않음' for p in p_values]
#     })
    
#     st.dataframe(granger_results, height=400)

# 검정 방법론 및 결과 해석
# st.markdown("### 🔸 그레인저 인과성 검정 방법론")

# st.markdown("""
# #####  검정의 개념과 한계

# 그레인저 인과검정은 한 시계열 자료가 다른 시계열 자료의 **미래값 예측에 도움이 되는지** 
# 통계적으로 확인하는 방법입니다. 이를 기반으로 인과관계를 추론하는 것일 뿐, 
# **실제 인과관계를 반드시 따른다는 것을 의미하지 않는다**는 점에 유의해야 합니다.

# #####  분석 절차:
# - AI 지수(AI 기반 지정학적 리스크 지수)와 수급안정화지수를 각각 차분하여 변화량 지표 생성
# - 시계열의 정상성 확보를 위한 데이터 변환
# - 1개월부터 12개월까지 시차별 순차 검정
# """)

# 검정 결과 및 해석
st.markdown("### 🔸 검정 결과 및 통계적 해석")

col_result1, col_result2 = st.columns(2)

with col_result1:
    st.success("""
    ####  주요 검정 결과
    - **검정 범위:** 1~12개월 시차
    - **결과:** 모든 시차에서 p < 0.05
    - **결론:** AI 수급 리스크 지수가 수급안정화지수의 변동을 예측하는 유의미한 선행지표
    - **방향성:** 역방향은 성립하지 않음
    """)

with col_result2:
    st.info("""
    ####  해석 시 주의사항
    - **통계적 인과성:** 예측력 향상을 의미하며, 진정한 인과관계와는 구별
    - **방향성:** AI 지수 → 수급안정화지수 (단방향)
    - **시차 효과:** 최대 12개월까지 지속적 영향
    - **예측 의미:** AI 지수 변동성 심화 시 향후 수급지수 급변 가능성
    """)

# 변동성 기반 조기 경보 시스템
# st.markdown("### 🔸 변동성 기반 조기 경보 메커니즘")

# st.markdown("""
# ####  AI 지수 변동성 모니터링

# 검정 결과를 바탕으로, **AI 지수의 변동성이 심화되는 구간에서는 향후 수급안정화지수가 
# 급격한 변화를 보일 가능성**이 있다는 점을 파악할 수 있습니다.
# """)

# st.warning("""
# ####  조기 경보 지표
# - **변동성 임계값:** AI 지수의 표준편차가 과거 평균의 1.5배 초과 시
# - **지속성 조건:** 높은 변동성이 3개월 이상 지속 시
# - **예상 영향:** 향후 3~6개월 내 수급안정화지수 급변 가능성
# - **모니터링 주기:** 월별 rolling volatility 추적
# """)

# # 기술적 상세 정보
# with st.expander("🔧 기술적 상세 정보"):
#     st.markdown("""
#     ### 그레인저 인과성 검정 방법론
    
#     **1. 데이터 전처리**
#     - AI 수급 리스크 지수와 KOMIS 수급안정화지수를 각각 차분하여 변화량 지표 생성
#     - 시계열의 정상성 확보를 위한 데이터 변환
    
#     **2. 검정 절차**
#     - 귀무가설: AI 지수는 KOMIS 지수의 예측에 도움이 되지 않음
#     - 대립가설: AI 지수는 KOMIS 지수의 예측에 유의미한 도움이 됨
#     - 시차 1개월부터 12개월까지 순차적 검정
    
#     **3. 판정 기준**
#     - 유의수준: α = 0.05
#     - 모든 시차에서 p < 0.05로 귀무가설 기각
#     - AI 지수의 선행지표로서의 유의성 확인
#     """)


# 돌아가기 버튼
st.markdown("---")
if st.button("🏠 메인 대시보드로 돌아가기", use_container_width=True):
    st.switch_page("main.py")

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p><strong>RAG 기반 생성형 AI를 활용한 수급 리스크 예측 모델</strong></p>
    <p>🏆 산업통상자원부 제13회 공공데이터 활용 아이디어 공모전 출품작</p>
</div>
""", unsafe_allow_html=True)
