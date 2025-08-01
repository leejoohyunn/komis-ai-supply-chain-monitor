# pages/3_🗄️_통합_데이터_허브.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import io
import os

st.set_page_config(
    page_title="통합 데이터 허브",
    page_icon="🗄️",
    layout="wide"
)

# 커스텀 CSS
st.markdown("""
<style>
    .data-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    
    .dimension-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .resource-card { border-left-color: #28a745; }
    .market-card { border-left-color: #007bff; }
    .tech-card { border-left-color: #ffc107; }
    .geopolitical-card { border-left-color: #dc3545; }
    
    .quality-excellent { color: #28a745; }
    .quality-good { color: #ffc107; }
    .quality-poor { color: #dc3545; }
    
    .status-online { color: #28a745; }
    .status-offline { color: #dc3545; }
    .status-warning { color: #ffc107; }
</style>
""", unsafe_allow_html=True)

def add_paper_citation():
    """논문 인용을 추가하는 간단한 함수"""
    # 컬럼을 사용해서 좌측에는 이미지, 우측에는 논문 인용 배치
    col_image, col_citation = st.columns([1, 2])
    
    with col_image:
        try:
            # 현재 파일 위치에서 ../images/image2.png 절대 경로 계산
            current_dir = os.path.dirname(os.path.abspath(__file__))
            image_path = os.path.join(current_dir, "..", "images", "image2.png")
    
            # 이미지 표시
            st.image(image_path, caption="Jia, S.(2025)의 핵심 광물 공급리스크 평가 지표 체계", width=300)
    
        except Exception as e:
            # 이미지가 없을 경우 마크다운 플레이스홀더 표시
            st.markdown("""
            <div style="background-color: #f5f5f5; padding: 20px; border-radius: 10px; text-align: center; border: 2px dashed #ccc; margin-bottom: 5px;">
                <p style="margin: 0;">📊 image2.png</p>
                <p style="margin: 5px 0 0 0;"><small>이미지 파일을 업로드해주세요</small></p>
            </div>
            """, unsafe_allow_html=True)
        
    with col_citation:
        st.markdown("""
        <div style="margin-top: 10px;">
        본 시스템은 <strong>Jia, S.(2025)의 연구</strong>를 기반으로 니켈 수급 리스크를 4가지 핵심 차원에서 분석합니다
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        - **자원 차원**: 광물 공급 기반의 구조적 안정성
        - **시장 차원**: 공급/수요 균형과 가격 신호
        - **기술 차원**: 공급 압력에 대한 기술적 대응력
        - **국제관계 차원**: 지정학적 리스크 (RAG 기반 AI 분석)
        """)
        
        # DOI 링크 추가
        st.markdown("""
        <div style="margin-top: 15px; padding: 10px; background-color: #f8f9fa; border-radius: 8px; border-left: 3px solid #007bff;">
            <a href="https://www.nature.com/articles/s41598-025-94848-8" target="_blank" style="text-decoration: none; color: #007bff;">
                <span style="font-size: 1.2em;">🔗</span> 
                <strong>원문 논문 보기 (DOI)</strong>
                <br>
                <small style="color: #6c757d;">Scientific Reports | Nature</small>
            </a>
        </div>
        """, unsafe_allow_html=True)

st.markdown("# 4차원 데이터 명세서")
st.markdown("RAG 기반 생성형 AI를 활용한 4차원 데이터 분석 시스템")
st.markdown("### 🔸사용한 데이터 목록")

# 정량/정성 데이터를 Streamlit 컬럼으로 분리
data_col1, data_col2 = st.columns(2)

with data_col1:
    st.markdown("""
    <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; border-left: 4px solid #6c757d;">
        <h6 style="color: #495057; margin-bottom: 10px;">📈 정량 데이터</h6>
        <ul style="font-size: 0.9em; line-height: 1.6; margin: 0;">
            <li>전세계 광산 생산량</li>
            <li>전세계 매장량</li>
            <li>자원대체율</li>
            <li>선물가격</li>
            <li>시장전망지표</li>
            <li>전세계 소비량</li>
            <li>생산집중도(HHI)</li>
            <li>대한민국 수출입 규모</li>
            <li>주요 기업 주가</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with data_col2:
    st.markdown("""
    <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; border-left: 4px solid #6c757d;">
        <h6 style="color: #495057; margin-bottom: 10px;">📝 정성 데이터</h6>
        <ul style="font-size: 0.9em; line-height: 1.6; margin: 0;">
            <li>전략광종 월간동향</li>
            <li>희소금속 월간동향</li>
            <li>주간자원뉴스</li>
            <li>일일자원뉴스</li>
        </ul>
        <div style="background-color: #fff3cd; padding: 10px; border-radius: 8px; margin-top: 10px; border-left: 3px solid #ffc107;">
            <small style="color: #856404;">
                 정성 데이터를 RAG 기반 LLM으로 정량화하여 예측 모델에 통합
            </small>
        </div>
    </div>
    """, unsafe_allow_html=True)

# 메인 컨텐츠
st.markdown("### 🔸 4차원 수급 리스크 분석 체계")
    
# 논문 인용과 이미지 추가
add_paper_citation()
    
# 4차원 데이터를 2x2 그리드로 배치
st.markdown("### 🔸 4차원 데이터 상세 현황")

# 좌측과 우측 컬럼 생성
col_left, col_right = st.columns(2)

# 좌측 컬럼: 자원 차원 + 시장 차원
with col_left:
        # 자원 차원 데이터
        
        
        resource_data = {
            "데이터명": ["전세계 광산 생산량", "전세계 매장량"],
            "설명": ["연간 전세계 광산에서 채굴된 니켈의 총량", "현재 채굴 가능한 니켈의 총량"],
            "출처": ["미국지질조사국(USGS)", "미국지질조사국(USGS)"],
            "업데이트": ["연 1회", "연 1회"],
            "활용목적": ["실제 공급되는 자원의 규모와 안정성 평가", "장기적 수급 안정성의 근간 판단"]
        }
        
        st.markdown("""
        <div class="data-card resource-card">
            <h5>자원 차원 - 공급 기반 안정성</h5>
            <p><strong>목적:</strong> 광물 공급 기반의 구조적 안정성을 평가</p>
            <p><strong>핵심 지표:</strong> 전세계 생산량 · 매장량</p>
            <p><strong>한계:</strong> USGS 데이터는 연간 단위로만 제공되어 월별 변동성 반영 한계</p>
        </div>
        """, unsafe_allow_html=True)
        
        resource_df = pd.DataFrame(resource_data)
        st.dataframe(resource_df, use_container_width=True)
        
        st.markdown("---")
        
        # 시장 차원 데이터
    
        
        market_data = {
            "데이터명": ["선물가격", "시장전망지표", "전세계 소비량", "생산집중도(HHI)", "대한민국 수출입 규모", "주요 기업 주가"],
            "설명": ["미래 특정 시점 약속된 니켈 가격", "중장기 투자 리스크 평가 척도", "연간 전세계 산업에서 사용된 니켈 총량", "주요 생산국의 시장 지배력 측정", "국내 니켈 수출입 물량/금액 추이", "Vale, Glencore 등 주요 기업 시장 데이터"],
            "출처": ["Investing.com", "KOMIS", "KOMIS", "USGS", "관세청", "Yahoo Finance"],
            "업데이트": ["실시간", "월 1회", "년 1회", "년 1회", "월 1회", "실시간"],
            "활용목적": ["미래 수급 기대 반영", "전문가 종합 판단", "수요 측면 규모 파악", "지정학적 리스크 측정", "국내 수급 취약성 진단", "공급망 핵심 기업 동향"]
        }
        
        st.markdown("""
        <div class="data-card market-card">
            <h5>시장 차원 - 공급/수요 균형</h5>
            <p><strong>목적:</strong> 광물 시장의 공급/수요 균형과 가격 신호를 반영</p>
            <p><strong>핵심 지표:</strong> 선물가격 · HHI · 수출입 통계 · 주요 기업 주가</p>
            <p><strong>특징:</strong> 선물가격 외 대부분 지표가 분기/연간 단위로 시점 불일치 존재</p>
        </div>
        """, unsafe_allow_html=True)
        
        market_df = pd.DataFrame(market_data)
        st.dataframe(market_df, use_container_width=True)
    
# 우측 컬럼: 기술 차원 + 국제관계 차원
with col_right:
        # 기술 차원 데이터
        
        
        tech_data = {
            "데이터명": ["자원대체율"],
            "설명": ["기존 니켈을 성능 저하나 비용 증가 없이 대체할 수 있는 가능성"],
            "출처": ["EU CRMs (Critical Raw Materials)"],
            "업데이트": ["년 1회"],
            "활용목적": ["광물 부족 시 다른 소재로의 대체 가능성을 통한 공급 위험 완화 평가"]
        }
        
        st.markdown("""
        <div class="data-card tech-card">
            <h5>기술 차원 - 기술적 대응력</h5>
            <p><strong>목적:</strong> 공급 압력에 대한 기술적 대응력을 평가</p>
            <p><strong>핵심 지표:</strong> 자원대체율</p>
            <p><strong>한계:</strong> 신뢰도 있는 '자원회수율' 데이터 확보 불가로 대체율만 활용</p>
        </div>
        """, unsafe_allow_html=True)
        
        tech_df = pd.DataFrame(tech_data)
        st.dataframe(tech_df, use_container_width=True)
        
        st.markdown("---")
        
        # 국제관계 차원 데이터 (AI 기반)
        
        
        geopolitical_data = {
            "데이터명": ["전략광종 월간동향", "희소금속 월간동향", "주간자원뉴스", "일일자원뉴스", "AI 지정학적 리스크 지수"],
            "설명": ["월간 니켈 시장 주요 이슈 및 동향", "월간 희소금속 시장 동향", "주간 광물 가격 및 시장 동향", "일일 광물 시장 동향", "RAG 기반 LLM으로 생성된 리스크 지수"],
            "출처": ["KOMIS", "KOMIS", "KOMIS", "KOMIS", "자체 개발 (Gemini 2.5 Flash)"],
            "업데이트": ["월 1회", "월 1회", "주 1회", "일 1회", "실시간"],
            "활용목적": ["중장기 구조적 변화", "희소금속 시장 분석", "단기 이슈 포착", "실시간 동향 반영", "비정형 데이터 정량화"]
        }
        
        st.markdown("""
        <div class="data-card geopolitical-card">
            <h5>국제관계 차원 - 지정학적 리스크 (혁신적 접근)</h5>
            <p><strong>목적:</strong> 기존 정량 데이터로 포착 불가능한 지정학적 리스크를 AI로 분석</p>
            <p><strong>핵심 기술:</strong> RAG (검색 증강 생성) + LLM (Gemini 2.5 Flash)</p>
            <p><strong>혁신점:</strong> 비정형 텍스트 데이터를 실시간으로 분석하여 정량적 지수로 변환</p>
            <p><strong>분석 과정:</strong> 보고서 크롤링 → RAG 데이터베이스 구축 → 의미 단위 분해 → 긍/부정 분류 → 지수 산출</p>
        </div>
        """, unsafe_allow_html=True)
        
        geopolitical_df = pd.DataFrame(geopolitical_data)
        st.dataframe(geopolitical_df, use_container_width=True)
    
# # 데이터 통합 체계
# st.markdown("### 🔗 4차원 데이터 통합 체계")

# col1, col2 = st.columns(2)

# with col1:
#     st.markdown("""
#     **📋 데이터 수집 전략:**
#     - **정량 데이터**: 기존 통계 기반 (자원, 시장, 기술)
#     - **정성 데이터**: AI 기반 비정형 데이터 분석 (국제관계)
#     - **시점 조정**: 연간/분기 데이터를 월별로 보간
#     - **실시간 업데이트**: RAG 시스템을 통한 최신 정보 반영
#     """)

# with col2:
#     st.markdown("""
#     **🎯 예측 모델링:**
#     - **KAN (Kolmogorov-Arnold Network)** 적용
#     - **4차원 변수** 통합 분석
#     - **확장 윈도우 검증** + **전진 검증**
#     - **MAPE 23%** 달성 (목표 지수 예측)
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
