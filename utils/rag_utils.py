# rag_utils.py

import streamlit as st
from io import BytesIO
import tempfile
import os
import json
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# LangChain imports
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain.chains import RetrievalQA

# --- 1. RAG 데이터베이스 구축 (업로드된 파일들 처리) ---
def setup_rag_database_from_files(uploaded_files):
    """
    업로드된 CSV 파일들로부터 RAG 데이터베이스 구축
    """
    print(f"1단계: {len(uploaded_files)}개 업로드된 파일에서 데이터 로딩...")
    all_docs = []
    
    for uploaded_file in uploaded_files:
        filename = uploaded_file.name
        print(f"   -> {filename} 처리 중...")
        
        try:
            # 업로드된 파일을 DataFrame으로 읽기
            try:
                df = pd.read_csv(uploaded_file, encoding='utf-8', dtype=str)
            except UnicodeDecodeError:
                uploaded_file.seek(0)  # 파일 포인터 리셋
                df = pd.read_csv(uploaded_file, encoding='cp949', dtype=str)
                print(f"   -> '{filename}'은 cp949 인코딩으로 로드")

            for _, row in df.iterrows():
                content = str(row.get("보고서 내용", ""))
                raw_date = str(row.get("날짜", "")).rstrip()
                try:
                    dt = pd.to_datetime(raw_date, errors='coerce')
                    year_month = dt.strftime('%Y%m') if pd.notna(dt) else (raw_date[:6] if raw_date else "")
                except:
                    year_month = raw_date[:6] if raw_date else ""
                metadata = { "년월": year_month, "광물": str(row.get("광물", "")), "source": filename }
                doc = Document(page_content=content, metadata=metadata)
                all_docs.append(doc)
            print(f"   -> {filename} 로드 완료 ({len(df)}개 행)")
        except Exception as e:
            print(f"   - 오류: {filename} 파일 처리 중 문제 발생 - {e}")
    
    if not all_docs:
        print("경고: 로드된 문서가 없습니다.")
        return None, None
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    splits = text_splitter.split_documents(all_docs)
    print("텍스트 임베딩 및 벡터 DB 저장...")
    embedding_model = HuggingFaceEmbeddings(model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_model)
    print("✅ 성공: RAG 데이터베이스 구축 완료.")
    return vectorstore, embedding_model

# --- 1-1. 기존 디렉토리 방식 (호환성 유지) ---
def setup_rag_database(data_directory: str):
    """
    CSV 파일들로부터 RAG 데이터베이스 구축 (기존 방식)
    """
    print(f"1단계: '{data_directory}' 디렉토리에서 데이터 로딩...")
    all_docs = []
    if not os.path.exists(data_directory):
        print(f"경고: '{data_directory}' 디렉토리가 없습니다.")
        return None, None
    
    for filename in os.listdir(data_directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(data_directory, filename)
            try:
                try:
                    df = pd.read_csv(file_path, encoding='utf-8', dtype=str)
                except UnicodeDecodeError:
                    df = pd.read_csv(file_path, encoding='cp949', dtype=str)
                    print(f"   -> '{filename}'은 cp949 인코딩으로 로드")

                for _, row in df.iterrows():
                    content = str(row.get("보고서 내용", ""))
                    raw_date = str(row.get("날짜", "")).rstrip()
                    try:
                        dt = pd.to_datetime(raw_date, errors='coerce')
                        year_month = dt.strftime('%Y%m') if pd.notna(dt) else (raw_date[:6] if raw_date else "")
                    except:
                        year_month = raw_date[:6] if raw_date else ""
                    metadata = { "년월": year_month, "광물": str(row.get("광물", "")), "source": filename }
                    doc = Document(page_content=content, metadata=metadata)
                    all_docs.append(doc)
                print(f"   -> {filename} 로드 완료.")
            except Exception as e:
                print(f"   - 오류: {filename} 파일 처리 중 문제 발생 - {e}")
    
    if not all_docs:
        print("경고: 로드된 문서가 없습니다.")
        return None, None
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    splits = text_splitter.split_documents(all_docs)
    print("텍스트 임베딩 및 벡터 DB 저장...")
    embedding_model = HuggingFaceEmbeddings(model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_model)
    print("✅ 성공: RAG 데이터베이스 구축 완료.")
    return vectorstore, embedding_model

# 1) 기존 RAG 체인 초기화 함수 (Streamlit용 - 단일 CSV 파일 처리)
@st.cache_resource(show_spinner=False)
def init_rag_chain(csv_bytes: bytes):
    """
    CSV 파일(bytes) → Document 로딩 →
    HuggingFace 임베딩 → Chroma DB 생성 →
    RetrievalQA 체인 반환
    """
    # 1. CSV → Document
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.csv', delete=False) as tmp_file:
        tmp_file.write(csv_bytes)
        tmp_file_path = tmp_file.name
    
    try:
        loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8")
        docs = loader.load()
    except:
        # pandas로 직접 처리
        df = pd.read_csv(tmp_file_path, encoding='utf-8')
        docs = [Document(page_content=str(row.to_dict()), metadata={"source": f"row_{i}"}) 
                for i, row in df.iterrows()]
    finally:
        os.unlink(tmp_file_path)

    # 2. 텍스트 분할 및 임베딩
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    
    embeddings = HuggingFaceEmbeddings(model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS")
    vectordb = Chroma.from_documents(documents=splits, embedding=embeddings)

    # 3. RetrievalQA 체인 생성
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(),
        return_source_documents=False
    )

    return qa

# --- 2. 안전한 JSON 파싱을 위한 헬퍼 함수 ---
def safe_json_loads(response_str: str, default_value=None):
    if default_value is None:
        default_value = {}
    try:
        if '```json' in response_str:
            response_str = response_str.split('```json')[1].split('```')[0]
        return json.loads(response_str.strip())
    except (json.JSONDecodeError, IndexError):
        return default_value

# --- 3. 고도화된 분석 체인 (노트북에서 가져온 함수) ---
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
    # 고품질의 판단을 위해 더 성능이 좋은 모델 사용을 권장합니다.
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite-preview-06-17", temperature=0)
    return prompt_template | llm | StrOutputParser()

# --- 4. LLM 문장 분리기 (노트북에서 가져온 함수) ---
def get_sentences_from_text_with_llm(text_block: str, llm: ChatGoogleGenerativeAI) -> list[str]:
    print("   -> LLM을 이용해 문장 분리 작업 시작...")
    segment_prompt = ChatPromptTemplate.from_template(
    """
 # 역할:
        당신은 긴 보고서에서 '단일 사건(a single, atomic event)'을 추출하여, 각각의 독립적인 의미를 가진 문장으로 분리하는 문맥 분석 전문가입니다.

        # 핵심 지시사항:
        1.  주어진 텍스트를 의미적으로 완전한 하나의 아이디어를 담고 있는 단위로 묶거나 나눠주세요.

        2.  **[분리 원칙]** 하나의 문장 안에 접속사(~했으며, ~하고 등)로 연결된 여러 사건이 있다면, 각각의 **독립적인 사건으로 반드시 분리**해야 합니다.
            - 예시: "중국은 A를 했고, 미국은 B를 했다." -> "중국은 A를 했다.", "미국은 B를 했다."

        3.  🚨 **[매우 중요 - 병합 원칙]** 이와 반대로, **하나의 사건을 설명하기 위해 논리적으로 강하게 연결된 여러 문장은 반드시 하나의 완결된 문장으로 합쳐야 합니다.**
            - **원인과 결과, 주장과 근거, 현상과 부연 설명, 목적과 수단** 등은 분리해서는 안 되는 강력한 연결 관계입니다.

        4.  각 단위는 그 자체로 독립적으로 이해될 수 있어야 합니다.

        5.  답변은 다른 어떤 설명도 없이, 오직 JSON 형식의 문자열 리스트로만 출력하세요.

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
        print(f"   -> 성공: {len(sentences)}개의 문장으로 분리 완료.")
        return sentences
    else:
        print("   -> 문장 분리 중 오류 발생. 원본 텍스트를 통째로 사용합니다.")
        return [text_block]

# 2) 기존 리스크 점수 계산 함수 (간단한 키워드 기반)
def calculate_risk_score(answer: str) -> int:
    """
    AI 답변 텍스트에서 키워드 기반 간단 점수 산출 예시.
    실제 비즈니스 로직으로 교체하세요.
    """
    score = 50
    text = answer.lower()

    if "높음" in text or "위험" in text:
        score += 30
    if "중간" in text:
        score += 15
    if "낮음" in text or "안정" in text:
        score -= 10

    # 0~100 범위로 클램핑
    return max(0, min(100, score))

# --- 5. 월별 AI 지수 생성 (노트북에서 가져온 함수) ---
def create_monthly_ai_index_with_intensity(start_year: int, start_month: int, end_year: int, end_month: int, vectorstore, analysis_chain, target_mineral="니켈"):
    """
    월별 AI 지수 생성 - 강도 가중치 적용
    """
    print(f"\n최종 분석 단계: {start_year}년 {start_month}월부터 월별 수입 심리 지수(강도 가중) 생성 시작...")
    monthly_results = []
    llm_for_splitter = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite-preview-06-17", temperature=0)

    date_range = pd.date_range(start=f'{start_year}-{start_month}-01', end=f'{end_year}-{end_month}-01', freq='MS')

    for month_start_date in date_range:
        current_month_str_ym = month_start_date.strftime('%Y%m')
        current_month_str_kor = month_start_date.strftime('%Y년 %m월')
        print(f"--- {current_month_str_kor} 분석 중 ---")

        filter_dict = {"$and": [{"광물": {"$eq": target_mineral}}, {"년월": {"$eq": current_month_str_ym}}]}
        retrieved_texts = vectorstore.get(where=filter_dict, include=["documents"]).get('documents', [])

        if not retrieved_texts:
            print("   -> 해당 기간에 검색된 문서가 없습니다.")
            monthly_results.append({ "date": month_start_date, "nsi_score": np.nan, "summary": "데이터 없음", "analysis_results": [] })
            continue

        raw_context_str = "\n\n".join(retrieved_texts)
        all_sentences = get_sentences_from_text_with_llm(raw_context_str, llm_for_splitter)

        batch_size = 20
        all_analysis_results = []
        print(f"   -> 총 {len(all_sentences)}개 문장을 {batch_size}개씩 나누어 고도화된 분석 실행...")

        for i in range(0, len(all_sentences), batch_size):
            batch_sentences = all_sentences[i:i + batch_size]
            final_context = "\n".join([f"- {s}" for s in batch_sentences])
            print(f"     -> 배치 {i//batch_size + 1} 분석 중 ({len(batch_sentences)}개 문장)")

            try:
                response_str = analysis_chain.invoke({"context": final_context, "target_item": target_mineral})
                response_json = safe_json_loads(response_str)

                # 새로운 JSON 구조에 맞춰 결과 파싱
                for item in response_json.get("analysis", []):
                    all_analysis_results.append({
                        "text": item.get("sentence"),
                        "classification": item.get("classification"),
                        "intensity": item.get("intensity", "Low"), # intensity가 없으면 Low 기본값
                        "reason": item.get("reason", "N/A")
                    })
                time.sleep(1)
            except Exception as e:
                print(f"     -> 배치 분석 중 오류 발생: {e}")

        # --- 강도(Intensity)를 가중치로 사용한 NSI 점수 계산 ---
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

        total_analyzed_count = sum(1 for item in all_analysis_results if item.get("classification") in ["Positive", "Negative"])
        k = 0 # 분모가 0이 되는 것을 방지하고 점수 변동을 완만하게 하는 상수

        if total_analyzed_count > 0:
            nsi_score = (p_weighted_sum - n_weighted_sum) / (p_weighted_sum + n_weighted_sum + k)
        else:
            nsi_score = 0.0

        final_summary = f"총 {total_analyzed_count}개 요인 분석 (강도 가중치 적용): 긍정 가중합 {p_weighted_sum}, 부정 가중합 {n_weighted_sum}."

        print(f"   -> 최종 긍정/부정 가중합: {p_weighted_sum}/{n_weighted_sum}")
        print(f"   -> 계산된 NSI 점수 (Raw): {nsi_score:.3f}")

        monthly_results.append({
            "date": month_start_date, "nsi_score": nsi_score,
            "summary": final_summary, "analysis_results": all_analysis_results
        })

    # --- 데이터프레임 생성 및 후처리 ---
    df_monthly = pd.DataFrame(monthly_results)
    if df_monthly.empty: 
        return pd.DataFrame()
    
    df_monthly['nsi_score_smoothed'] = df_monthly['nsi_score'].ewm(span=6, adjust=False).mean()
    print("\n✅ 5개월 지수이동평균(EWM) 적용 완료. 평활화된 점수를 기준으로 최종 지수를 계산합니다.")

    valid_scores = df_monthly['nsi_score_smoothed'].dropna()
    if not valid_scores.empty:
        min_val, max_val = valid_scores.min(), valid_scores.max()
        range_val = max_val - min_val
        if range_val == 0:
            df_monthly['final_supply_demand_index'] = 50.0
        else:
            df_monthly['final_supply_demand_index'] = df_monthly.apply(
                lambda row: ((row['nsi_score_smoothed'] - min_val) / range_val) * 100 if pd.notna(row['nsi_score_smoothed']) else np.nan,
                axis=1
            )
    else:
        df_monthly['final_supply_demand_index'] = np.nan
    df_monthly['final_supply_demand_index'].fillna(50.0, inplace=True)
    
    return df_monthly
