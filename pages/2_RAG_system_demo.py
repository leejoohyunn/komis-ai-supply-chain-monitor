import os
os.environ["CHROMA_SERVER"] = "false"

# SQLite 버전 문제 해결
import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3

# 그 다음에 chromadb import
import chromadb
from langchain_community.vectorstores import Chroma

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
from datetime import datetime
import time

# Langchain imports
from chromadb.config import Settings  # 이 줄을 파일 상단에 추가
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings  # 이 줄 제거
from langchain_huggingface import HuggingFaceEmbeddings  # 새로운 import
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# Google API 키 설정 - Secrets 우선, fallback으로 하드코딩
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
st.sidebar.success("✅ API 키 로드 성공")


# Streamlit 페이지 설정
st.set_page_config(
    page_title="AI 지정학적 리스크 지수 분석",
    page_icon="📊",
    layout="wide"
)

@st.cache_data
def load_data(file_path):
    """데이터 로딩 함수"""
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='cp949')
    return df

def safe_json_loads(response_str: str, default_value=None):
    """안전한 JSON 파싱을 위한 헬퍼 함수"""
    if default_value is None:
        default_value = {}
    try:
        if '```json' in response_str:
            response_str = response_str.split('```json')[1].split('```')[0]
        return json.loads(response_str.strip())
    except (json.JSONDecodeError, IndexError):
        return default_value

def create_improved_analysis_chain():
    """'생각의 사슬', '자기 비판', '강도' 평가를 포함하여 정확도를 높인 분석 체인"""
    prompt_template = ChatPromptTemplate.from_template(
        """
[AI 작업 지침 - 수입자 리스크 분석 (고도화 버전)]

당신은 '비용'과 '공급 안정성' 관점에서만 평가하는 대한민국 원자재 구매 담당자입니다.
가격 상승이나 공급망 불안정은 원인과 관계없이 부정적 신호입니다.

[분석 절차 (4단계)]

### 1단계: 사실 식별 (Fact Identification)
- 문장에서 {target_item} 수입과 관련된 객관적인 사실(event)을 먼저 식별합니다.

### 2단계: 영향 분석 (Impact Analysis - Chain of Thought)
- 각 사실이 대한민국의 {target_item} 수입에 어떤 연쇄적인 영향을 미치는지 단계별로 추론하세요.
- (예: "A사 파업" -> "생산 차질 발생" -> "시장 공급량 감소" -> "수입 가격 상승 압력")
- 이 추론 과정을 'reason'에 명확히 서술해야 합니다.

### 3단계: 초기 분류 (Initial Classification)
- '영향 분석' 결과를 바탕으로, 해당 사건이 수입자에게 [Positive], [Negative], [Neutral] 중 무엇인지, 그리고 그 영향의 강도를 [High, Medium, Low] 중에서 잠정적으로 판단합니다.

### 4단계: 자기 비판 및 최종 결정 (Self-Critique & Final Decision)
- 내린 잠정 판단이 [최종 판단 원칙]에 부합하는지 스스로 검토하세요.
- (예: "이 사건은 경기 회복 신호로 볼 수도 있지만, 명시적으로 '가격 상승'을 유발했으므로, 수입자 관점에서는 최종적으로 [Negative]로 분류하는 것이 원칙에 맞다.")
- 이 검토 과정을 거쳐 최종 분류(classification)와 강도(intensity)를 확정합니다.

---

[최종 판단 원칙]
- 🚨 **가격 변동 최우선 원칙**: '가격 상승' 또는 '상승 압력'이 언급되면, 다른 긍정적 맥락(예: 수요 증가)이 있더라도 무조건 [Negative]로 분류합니다. 반대로 '가격 하락'은 [Positive]입니다.
- **공급망 리스크 우선 원칙**: 공급 자체를 위협하는 '전쟁', '수출 전면 금지' 등은 가격과 무관하게 [Negative]이며, 강도는 [High]로 판단합니다.
- **과잉 추론 금지**: 문장에 명시된 사실에만 근거합니다.

---

[분석 대상]
- Target Item: {target_item}
- Context: {context}

---

# 출력 JSON 형식:

{{
    "analysis": [
        {{
            "sentence": "분석 대상 문장 원본",
            "classification": "Positive, Negative, Neutral 중 하나",
            "intensity": "High, Medium, Low 중 하나",
            "reason": "위 분석 절차 2단계와 4단계에 따른 상세한 판단 과정 서술"
        }}
    ],
    "overall_summary": "이번 달의 긍정/부정 요인들을 종합하여 한두 문장으로 요약합니다."
}}
"""
    )
    
    # API 키가 설정되어 있는지 확인
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("GOOGLE_API_KEY 환경변수를 설정해주세요.")
        return None
        
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite-preview-06-17", temperature=0)
    return prompt_template | llm | StrOutputParser()

def setup_rag_database(df):
    """RAG 데이터베이스 구축"""
    all_docs = []
    
    for _, row in df.iterrows():
        content = str(row.get("보고서 내용", ""))
        raw_date = str(row.get("날짜", "")).rstrip()
        
        try:
            if len(raw_date) == 6 and raw_date.isdigit():
                # YYYYMM vs YYMMDD 자동 감지
                potential_year = raw_date[:4]
                potential_month = raw_date[4:6]
                
                if (1900 <= int(potential_year) <= 2100 and 1 <= int(potential_month) <= 12):
                    # YYYYMM 형식 (예: 202301)
                    year_month = raw_date
                else:
                    # YYMMDD 형식 (예: 160111)
                    yy = raw_date[:2]
                    mm = raw_date[2:4]
                    yyyy = f"20{yy}" if int(yy) <= 30 else f"19{yy}"
                    year_month = f"{yyyy}{mm}"
            else:
                year_month = ""
        except:
            year_month = ""
            
        metadata = {
            "년월": year_month,
            "광물": str(row.get("광물이름", "")),
            "source": "uploaded_data"
        }
        
        doc = Document(page_content=content, metadata=metadata)
        all_docs.append(doc)
    
    if not all_docs:
        return None, None
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    splits = text_splitter.split_documents(all_docs)
    
    embedding_model = HuggingFaceEmbeddings(model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS")
    

    # persist_directory 제거하여 인메모리 모드 사용
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embedding_model,
    # persist_directory="/tmp/chroma_db" ← 이 부분 제거
        collection_name=f"rag_collection_{hash(str(splits[:10]))}"
    )
    
    return vectorstore, embedding_model

def get_sentences_from_text_with_llm(text_block: str, llm) -> list[str]:
    """LLM을 이용한 문장 분리"""
    segment_prompt = ChatPromptTemplate.from_template(
    """
# 역할:
당신은 긴 보고서에서 '단일 사건(a single, atomic event)'을 추출하여, 각각의 독립적인 의미를 가진 문장으로 분리하는 문맥 분석 전문가입니다.

# 핵심 지시사항:
1. 주어진 텍스트를 의미적으로 완전한 하나의 아이디어를 담고 있는 단위로 묶거나 나눠주세요.

2. **[분리 원칙]** 하나의 문장 안에 접속사(~했으며, ~하고 등)로 연결된 여러 사건이 있다면, 각각의 **독립적인 사건으로 반드시 분리**해야 합니다.
   - 예시: "중국은 A를 했고, 미국은 B를 했다." -> "중국은 A를 했다.", "미국은 B를 했다."

3. 🚨 **[매우 중요 - 병합 원칙]** 이와 반대로, **하나의 사건을 설명하기 위해 논리적으로 강하게 연결된 여러 문장은 반드시 하나의 완결된 문장으로 합쳐야 합니다.**
   - **원인과 결과, 주장과 근거, 현상과 부연 설명, 목적과 수단** 등은 분리해서는 안 되는 강력한 연결 관계입니다.

4. 각 단위는 그 자체로 독립적으로 이해될 수 있어야 합니다.

5. 답변은 다른 어떤 설명도 없이, 오직 JSON 형식의 문자열 리스트로만 출력하세요.

# 좋은 '병합' 예시 (여러 문장을 하나로 합쳐야 하는 경우):
- 원본: "Sherritt社는 비용절감 조치에 착수했습니다. 조직 간소화를 통한 인력 감축 및 비용 절감 프로그램을 추진하고 있습니다."
- 이상적인 출력: ["Sherritt사는 비용 절감 조치의 일환으로, 조직 간소화를 통한 인력 감축 프로그램을 추진하고 있습니다."]

# 🚨 좋은 '분리' 예시 (하나의 문장을 여러 개로 나눠야 하는 경우):
- 원본: "중국 상무부는 미국의 인플레이션 감축법(IRA) 보조금 지급 대상에서 중국산 전기차를 제외한 것에 대해 WTO에 제소했으며, 인도네시아 Nickel Industries사의 니켈 생산량은 전년 대비 크게 증가했습니다."
- 이상적인 출력: [
    "중국 상무부는 미국 IRA의 전기차 보조금 정책에 대해 WTO에 제소했습니다.",
    "인도네시아 Nickel Industries사의 니켈 생산량이 전년 대비 크게 증가했습니다."
  ]

# 이제 아래 텍스트를 분리해주세요:
{text}
    """
    )
    
    segment_chain = segment_prompt | llm | StrOutputParser()
    response_str = segment_chain.invoke({"text": text_block})
    sentences = safe_json_loads(response_str, default_value=[]) if isinstance(response_str, str) else []
    
    if sentences:
        return sentences
    else:
        return [text_block]

def analyze_monthly_data(vectorstore, analysis_chain, target_mineral, year_month_str):
    """특정 월 데이터 분석"""
    if not analysis_chain:
        return None, []
        
    llm_for_splitter = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite-preview-06-17", temperature=0)
    
    # 해당 월의 데이터 검색
    filter_dict = {
        "$and": [
            {"광물": {"$eq": target_mineral}},
            {"년월": {"$eq": year_month_str}}
        ]
    }
    
    retrieved_texts = vectorstore.get(where=filter_dict, include=["documents"]).get('documents', [])
    
    if not retrieved_texts:
        return None, []
    
    raw_context_str = "\n\n".join(retrieved_texts)
    all_sentences = get_sentences_from_text_with_llm(raw_context_str, llm_for_splitter)
    
    # 문장 분석
    all_analysis_results = []
    batch_size = 20
    
    for i in range(0, len(all_sentences), batch_size):
        batch_sentences = all_sentences[i:i + batch_size]
        final_context = "\n".join([f"- {s}" for s in batch_sentences])
        
        try:
            response_str = analysis_chain.invoke({
                "context": final_context,
                "target_item": target_mineral
            })
            response_json = safe_json_loads(response_str)
            
            for item in response_json.get("analysis", []):
                all_analysis_results.append({
                    "text": item.get("sentence"),
                    "classification": item.get("classification"),
                    "intensity": item.get("intensity", "Low"),
                    "reason": item.get("reason", "N/A")
                })
            time.sleep(1)
        except Exception as e:
            st.error(f"분석 중 오류 발생: {e}")
    
    # NSI 점수 계산
    intensity_map = {"High": 3, "Medium": 2, "Low": 1}
    p_weighted_sum = 0
    n_weighted_sum = 0
    
    for item in all_analysis_results:
        classification = item.get("classification")
        intensity_str = item.get("intensity", "Low")
        weight = intensity_map.get(intensity_str, 1)
        
        if classification == "Positive":
            p_weighted_sum += weight
        elif classification == "Negative":
            n_weighted_sum += weight
    
    total_analyzed_count = sum(1 for item in all_analysis_results 
                             if item.get("classification") in ["Positive", "Negative"])
    
    if total_analyzed_count > 0:
        nsi_score = (p_weighted_sum - n_weighted_sum) / (p_weighted_sum + n_weighted_sum)
    else:
        nsi_score = 0.0
    
    return nsi_score, all_analysis_results

def create_monthly_index_chart(df):
    """월별 AI 지수 차트 생성"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['final_supply_demand_index'],
        mode='lines+markers',
        name='AI 지정학적 리스크 지수',
        line=dict(color='blue', width=2),
        marker=dict(size=4)
    ))
    
    fig.add_hline(y=50, line_dash="dash", line_color="red", 
                  annotation_text="기준선(50)")
    
    fig.update_layout(
        title="월별 AI 지정학적 리스크 지수",
        xaxis_title="날짜",
        yaxis_title="지수",
        template="plotly_white",
        height=500
    )
    
    return fig

def create_demo_data():
    """실제 데이터 기반 데모 데이터 생성"""
    # 실제 데이터
    data = [
        ["2018-01-01", 48.30853833, 66.2572],
        ["2018-02-01", 35.70994376, 52.38063],
        ["2018-03-01", 40.99328251, 47.68762],
        ["2018-04-01", 32.15823809, 48.56763],
        ["2018-05-01", 33.00612464, 44.4211],
        ["2018-06-01", 35.97254191, 41.73109],
        ["2018-07-01", 22.87388158, 36.2654],
        ["2018-08-01", 22.70828271, 46.91906],
        ["2018-09-01", 21.75227603, 49.69394],
        ["2018-10-01", 33.72107281, 56.57],
        ["2018-11-01", 51.95780185, 59.52792],
        ["2018-12-01", 63.85931784, 69.12348],
        ["2019-01-01", 45.09565978, 72.45798],
        ["2019-02-01", 32.19951265, 65.48166],
        ["2019-03-01", 35.53526014, 55.47445],
        ["2019-04-01", 45.77399129, 52.35337],
        ["2019-05-01", 52.96013052, 54.2517],
        ["2019-06-01", 52.91600268, 60.77776],
        ["2019-07-01", 36.9126266, 60.04048],
        ["2019-08-01", 28.80366673, 48.03721],
        ["2019-09-01", 27.29606746, 34.50652],
        ["2019-10-01", 16.82657282, 25.11431],
        ["2019-11-01", 52.20227331, 26.65043],
        ["2019-12-01", 42.88224131, 37.25958],
        ["2020-01-01", 46.29867987, 47.78317],
        ["2020-02-01", 49.77218331, 60.25396],
        ["2020-03-01", 34.6890241, 60.75027],
        ["2020-04-01", 26.52550362, 67.99711],
        ["2020-05-01", 30.37710751, 68.4022],
        ["2020-06-01", 22.56849174, 62.48999],
        ["2020-07-01", 18.7330919, 56.98171],
        ["2020-08-01", 6.291914214, 49.70712],
        ["2020-09-01", 17.50595719, 39.97428],
        ["2020-10-01", 18.44355509, 37.74649],
        ["2020-11-01", 14.61262883, 35.00895],
        ["2020-12-01", 19.19167292, 29.32921],
        ["2021-01-01", 9.924368088, 25.78061],
        ["2021-02-01", 15.35632647, 17.26855],
        ["2021-03-01", 17.12032087, 13.15765],
        ["2021-04-01", 22.10754428, 21.45722],
        ["2021-05-01", 21.01290983, 18.40972],
        ["2021-06-01", 32.26048824, 12.08277],
        ["2021-07-01", 20.48203577, 11.29698],
        ["2021-08-01", 20.31609262, 9.066557],
        ["2021-09-01", 13.33105465, 9.075382],
        ["2021-10-01", 7.282059578, 8.793292],
        ["2021-11-01", 13.88703809, 7.901674],
        ["2021-12-01", 21.05767886, 10.1564],
        ["2022-01-01", 12.77927961, 8.811239],
        ["2022-02-01", 0, 7.40131],
        ["2022-03-01", 7.96516306, 6.242335],
        ["2022-04-01", 4.863766428, 6.338374],
        ["2022-05-01", 7.260939289, 8.119097],
        ["2022-06-01", 13.85991594, 7.586575],
        ["2022-07-01", 17.14285429, 9.819112],
        ["2022-08-01", 10.49676893, 16.41854],
        ["2022-09-01", 22.09083839, 16.36708],
        ["2022-10-01", 26.43560626, 16.6371],
        ["2022-11-01", 19.64676546, 17.81524],
        ["2022-12-01", 19.1562461, 11.87035],
        ["2023-01-01", 40.03198834, 9.528951],
        ["2023-02-01", 49.49331019, 9.590902],
        ["2023-03-01", 65.56501064, 13.99419],
        ["2023-04-01", 70.43662738, 19.87834],
        ["2023-05-01", 70.73481021, 18.53535],
        ["2023-06-01", 80.1414394, 25.89551],
        ["2023-07-01", 71.8909434, 29.55088],
        ["2023-08-01", 67.12988031, 27.99848],
        ["2023-09-01", 76.43800505, 30.96962],
        ["2023-10-01", 100, 31.97949],
        ["2023-11-01", 84.09436997, 41.74182],
        ["2023-12-01", 82.84669411, 47.39781],
        ["2024-01-01", 84.55067207, 48.13813],
        ["2024-02-01", 79.16141524, 52.14203],
        ["2024-03-01", 62.53566749, 51.21343],
        ["2024-04-01", 56.29632982, 43.99599],
        ["2024-05-01", 48.64784039, 36.39601]
    ]
    
    # DataFrame 생성
    demo_df = pd.DataFrame(data, columns=['date', 'ai_risk_index', 'actual_supply_index'])
    demo_df['date'] = pd.to_datetime(demo_df['date'])
    
    return demo_df

def create_interactive_dashboard(demo_df):
    """인터랙티브 대시보드 생성"""
    
    # 메인 비교 차트
    fig_main = go.Figure()
    
    # AI 리스크 지수 (초록색 점선)
    fig_main.add_trace(go.Scatter(
        x=demo_df['date'],
        y=demo_df['ai_risk_index'],
        mode='lines+markers',
        name='AI 지정학적 리스크 지수',
        line=dict(color='green', width=2, dash='dot'),
        marker=dict(size=4, color='green', symbol='circle'),
        hovertemplate='<b>AI 리스크 지수</b><br>날짜: %{x}<br>지수: %{y:.1f}<extra></extra>'
    ))
    
    # 실제 수급안정화 지수 (파란색)
    fig_main.add_trace(go.Scatter(
        x=demo_df['date'],
        y=demo_df['actual_supply_index'],
        mode='lines+markers',
        name='실제 수급안정화 지수',
        line=dict(color='#4169E1', width=3),
        marker=dict(size=5, color='#4169E1'),
        hovertemplate='<b>실제 수급안정화 지수</b><br>날짜: %{x}<br>지수: %{y:.1f}<extra></extra>'
    ))
    
    # 기준선 표시
    fig_main.add_hline(y=50, line_dash="dash", line_color="gray", 
                      annotation_text="기준선 (50)")
    
    fig_main.update_layout(
        title=" 니켈 AI 리스크 지수 vs 실제 수급안정화 지수 비교",
        xaxis_title="날짜",
        yaxis_title="지수",
        template="plotly_white",
        height=500,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig_main


# Streamlit 앱 메인
def main():
    st.title(" AI 리스크 지수 분석 시스템")
    st.markdown("Gemini 2.5 Flash + RAG 기술로 구현한 지정학적 리스크 분석 시스템 체험")
    st.markdown("---")
    
    # 데모 데이터 생성
    demo_df = create_demo_data()
    
    # 📈 인터랙티브 대시보드 섹션
    st.header(" ### 🔸 실시간 AI 리스크 분석 대시보드")
    
    # 메인 차트
    fig_main = create_interactive_dashboard(demo_df)
    st.plotly_chart(fig_main, use_container_width=True)
    st.title("AI 리스크 지수 분석 시스템")
    st.markdown("Gemini 2.5 Flash + RAG 기술로 구현한 지정학적 리스크 분석 시스템 체험")
    st.markdown("---")
    
    # 사이드바 설정
    st.sidebar.header("⚙️ 설정")
    
    # API 키는 이미 하드코딩되어 있음
    st.sidebar.info("Google API Key가 설정되어 있습니다.")
    
    # 파일 업로드
    uploaded_file = st.sidebar.file_uploader(
        "CSV 파일 업로드",
        type=['csv'],
        help="광물_주간동향_통합.csv 파일을 업로드하세요"
    )
    
    if uploaded_file is not None:
        # 데이터 로딩 (여러 인코딩 시도)
        try:
            df = pd.read_csv(uploaded_file, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(uploaded_file, encoding='cp949')
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(uploaded_file, encoding='euc-kr')
                except UnicodeDecodeError:
                    try:
                        df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
                    except UnicodeDecodeError:
                        st.error("파일 인코딩을 인식할 수 없습니다. UTF-8, CP949, EUC-KR, UTF-8-SIG 형식으로 저장된 파일을 업로드해주세요.")
                        return
        st.sidebar.success(f"데이터 로딩 완료: {len(df)}개 행")
        
        # 광물 선택
        minerals = df['광물이름'].unique()
        selected_mineral = st.sidebar.selectbox("분석할 광물 선택", minerals)
        
        # 메인 화면
        tab1, tab2, tab3 = st.tabs(["📈 월별 지수", "🔍 상세 분석", "📋 JSON 출력 형식"])
        
        with tab1:
            st.header(f"🔸{selected_mineral} 월별 AI 지정학적 리스크 지수")
            
            if st.button("지수 생성 시작"):
                if not os.getenv("GOOGLE_API_KEY"):
                    st.error("Google API Key가 설정되지 않았습니다.")
                    return
                
                with st.spinner("RAG 데이터베이스 구축 중..."):
                    vectorstore, _ = setup_rag_database(df)
                
                if vectorstore:
                    with st.spinner("AI 분석 체인 생성 중..."):
                        analysis_chain = create_improved_analysis_chain()
                    
                    if analysis_chain:
                        # 월별 데이터 분석 (YYMMDD -> YYYY-MM 형식으로 변환)
                        df['날짜'] = df['날짜'].astype(str)
                        # 160111 -> 201601 형식으로 변환
                        def convert_date_format(date_str):
                            date_str = str(date_str).strip()
                            if len(date_str) == 6 and date_str.isdigit():
                                potential_year = date_str[:4]
                                potential_month = date_str[4:6]
                                
                                if (1900 <= int(potential_year) <= 2100 and 1 <= int(potential_month) <= 12):
                                    return date_str  # YYYYMM 형식 (202301)
                                else:
                                    # YYMMDD 형식 (160111)
                                    yy = date_str[:2]
                                    mm = date_str[2:4]
                                    yyyy = f"20{yy}" if int(yy) <= 30 else f"19{yy}"
                                    return f"{yyyy}{mm}"
                            elif len(date_str) >= 4:
                                return f"20{date_str[:2]}{date_str[2:4]}"
                            else:
                                return date_str
                        
                        df['년월'] = df['날짜'].apply(convert_date_format)
                        
                        available_months = sorted(df[df['광물이름'] == selected_mineral]['년월'].unique())
                        
                        monthly_results = []
                        progress_bar = st.progress(0)
                        
                        for i, month in enumerate(available_months):
                            st.write(f"분석 중: {month[:4]}년 {month[4:6]}월")
                            
                            nsi_score, analysis_results = analyze_monthly_data(
                                vectorstore, analysis_chain, selected_mineral, month
                            )
                            
                            if nsi_score is not None:
                                monthly_results.append({
                                    'date': pd.to_datetime(f"{month}01", format='%Y%m%d'),
                                    'nsi_score': nsi_score,
                                    'month': month,
                                    'analysis_results': analysis_results
                                })
                            
                            progress_bar.progress((i + 1) / len(available_months))
                        
                        if monthly_results:
                            results_df = pd.DataFrame(monthly_results)
                            
                            # 평활화 및 지수 계산
                            results_df['nsi_score_smoothed'] = results_df['nsi_score'].ewm(span=6, adjust=False).mean()
                            
                            valid_scores = results_df['nsi_score_smoothed'].dropna()
                            if not valid_scores.empty:
                                min_val, max_val = valid_scores.min(), valid_scores.max()
                                range_val = max_val - min_val
                                if range_val == 0:
                                    results_df['final_supply_demand_index'] = 50.0
                                else:
                                    results_df['final_supply_demand_index'] = results_df.apply(
                                        lambda row: ((row['nsi_score_smoothed'] - min_val) / range_val) * 100 
                                        if pd.notna(row['nsi_score_smoothed']) else np.nan, axis=1
                                    )
                            
                            # 차트 표시
                            fig = create_monthly_index_chart(results_df)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # 데이터 테이블 표시
                            st.subheader("📊 월별 지수 데이터")
                            display_df = results_df[['date', 'final_supply_demand_index', 'nsi_score']].copy()
                            display_df['date'] = display_df['date'].dt.strftime('%Y-%m')
                            display_df.columns = ['날짜', 'AI 지수', 'NSI 점수']
                            st.dataframe(display_df)
                            
                            # 세션 상태에 결과 저장
                            st.session_state['monthly_results'] = monthly_results
                
        with tab2:
            st.header("🔸 특정 월 상세 분석")
            
            if 'monthly_results' in st.session_state:
                months = [result['month'] for result in st.session_state['monthly_results']]
                selected_month = st.selectbox(
                    "분석할 월 선택",
                    months,
                    format_func=lambda x: f"{x[:4]}년 {x[4:6]}월"
                )
                
                if selected_month:
                    month_data = next(
                        (result for result in st.session_state['monthly_results'] 
                         if result['month'] == selected_month), None
                    )
                    
                    if month_data:
                        st.subheader(f"{selected_month[:4]}년 {selected_month[4:6]}월 분석 결과")
                        
                        # 요약 정보
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("NSI 점수", f"{month_data['nsi_score']:.3f}")
                        with col2:
                            positive_count = sum(1 for item in month_data['analysis_results'] 
                                               if item.get('classification') == 'Positive')
                            st.metric("긍정 요인", positive_count)
                        with col3:
                            negative_count = sum(1 for item in month_data['analysis_results'] 
                                               if item.get('classification') == 'Negative')
                            st.metric("부정 요인", negative_count)
                        
                        # 상세 분석 결과
                        st.subheader("🔸 분석 상세 내용")
                        for i, item in enumerate(month_data['analysis_results']):
                            with st.expander(f"분석 {i+1}: {item.get('classification', 'N/A')} ({item.get('intensity', 'N/A')})"):
                                st.write("**원문:**", item.get('text', ''))
                                st.write("**분류:**", item.get('classification', ''))
                                st.write("**강도:**", item.get('intensity', ''))
                                st.write("**분석 이유:**", item.get('reason', ''))
            else:
                st.info("먼저 '월별 지수' 탭에서 분석을 실행해주세요.")
        
        with tab3:
            st.header("🔸create_improved_analysis_chain 출력 JSON 형식")
            
            st.code("""
{
    "analysis": [
        {
            "sentence": "분석 대상 문장 원본",
            "classification": "Positive, Negative, Neutral 중 하나",
            "intensity": "High, Medium, Low 중 하나",
            "reason": "상세한 판단 과정 서술"
        }
    ],
    "overall_summary": "이번 달의 긍정/부정 요인들을 종합하여 한두 문장으로 요약"
}
            """, language="json")
            
            st.markdown("""
            ### 🔸 JSON 형식 설명
            
            - **analysis**: 각 문장별 분석 결과 배열
              - **sentence**: 분석된 원본 문장
              - **classification**: 긍정(Positive), 부정(Negative), 중립(Neutral) 분류
              - **intensity**: 영향 강도 - 높음(High), 중간(Medium), 낮음(Low)
              - **reason**: AI의 판단 근거와 추론 과정
            
            - **overall_summary**: 해당 월의 전체적인 시장 상황 요약
            """)
    
    else:
        st.info("사이드바에서 CSV 파일을 업로드해주세요.")

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

if __name__ == "__main__":
    main()
