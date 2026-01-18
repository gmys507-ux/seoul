import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# ------------------------------------------------------------------------------
# 1. Page Configuration
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="서울시 생활인구 심화 분석 대시보드",
    page_icon="🏙️",
    layout="wide"
)

# ------------------------------------------------------------------------------
# 2. Data Loading & Preprocessing
# ------------------------------------------------------------------------------
import os
from pathlib import Path

@st.cache_data
def load_data():
    """
    데이터 파일을 로드합니다. 로컬 및 Streamlit Cloud 환경 모두 지원.
    """
    # 현재 파일의 위치 확인
    current_file = Path(__file__).resolve()
    current_dir = current_file.parent
    
    # 디버깅: 현재 경로 정보 출력
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🔍 경로 디버깅 정보**")
    st.sidebar.caption(f"현재 파일: `{current_file}`")
    st.sidebar.caption(f"현재 디렉토리: `{current_dir}`")
    
    # 시도할 경로 목록 (우선순위 순)
    possible_paths = [
        # 1. 현재 디렉토리에 직접 있는 경우
        current_dir / "01_seoul_living_population_cleaned.parquet",
        
        # 2. 표준 구조: src/dashboard.py, data/파일
        current_dir.parent / "data" / "01_seoul_living_population_cleaned.parquet",
        
        # 3. data 폴더가 현재 디렉토리 안에 있는 경우
        current_dir / "data" / "01_seoul_living_population_cleaned.parquet",
        
        # 4. Streamlit Cloud 절대 경로 (이전 오류 메시지 기반)
        Path("/mount/src/seoul/260110_seoul_eda/data/01_seoul_living_population_cleaned.parquet"),
        
        # 5. 상위 폴더에 260110_seoul_eda가 있는 경우
        current_dir.parent.parent / "260110_seoul_eda" / "data" / "01_seoul_living_population_cleaned.parquet",
        
        # 6. 프로젝트 루트가 2단계 위인 경우
        current_dir.parent / ".." / "data" / "01_seoul_living_population_cleaned.parquet",
    ]
    
    # 각 경로를 순회하며 파일 찾기
    for file_path in possible_paths:
        try:
            file_path = file_path.resolve()  # 절대 경로로 변환
            if file_path.exists():
                st.sidebar.success(f"✅ 데이터 파일 발견: `{file_path.name}`")
                df = pd.read_parquet(file_path)
                return df
        except Exception as e:
            continue
    
    # 모든 경로에서 파일을 찾지 못한 경우
    st.error("❌ **데이터 파일을 찾을 수 없습니다.**")
    st.error("GitHub의 `data` 폴더에 `01_seoul_living_population_cleaned.parquet` 파일이 있는지 확인해주세요.")
    
    with st.expander("🔍 시도한 경로 목록 보기"):
        for i, path in enumerate(possible_paths, 1):
            try:
                resolved = path.resolve()
                exists = "✅ 존재" if path.exists() else "❌ 없음"
                st.code(f"{i}. {exists}: {resolved}")
            except:
                st.code(f"{i}. ⚠️ 오류: {path}")
    
    st.stop()

@st.cache_data
def preprocess_for_dong_time(df):
    """행정동별, 시간대별 총 생활인구 집계"""
    # 성별/연령대 구분을 합쳐서 '동-시간' 단위의 총 인구로 집계
    df_grouped = df.groupby(['행정동명', '시간대구분'])['생활인구수'].sum().reset_index()
    return df_grouped

@st.cache_data
def preprocess_for_age(df):
    """행정동별 연령대별 인구 집계"""
    df_age = df.groupby(['행정동명', '연령대'])['생활인구수'].sum().reset_index()
    return df_age

@st.cache_data
def preprocess_for_gender(df):
    """행정동별 성별 인구 집계"""
    df_gender = df.groupby(['행정동명', '성별'])['생활인구수'].sum().reset_index()
    return df_gender

def calculate_zscore(series):
    return (series - series.mean()) / series.std()

# Load Data
try:
    raw_df = load_data()
    df_dt = preprocess_for_dong_time(raw_df) # dong-time aggregated
except Exception as e:
    st.error(f"데이터 로드 중 오류 발생: {e}")
    st.stop()

# ------------------------------------------------------------------------------
# 3. Sidebar UI
# ------------------------------------------------------------------------------
st.sidebar.title("🏙️ 서울시 생활인구 심화 분석")

# Navigation
tabs = ["1. 이상치(Outlier) 분석", "2. 심화 주제 분석"]
selected_tab = st.sidebar.radio("분석 주제 선택", tabs)

# Common Filters if needed (e.g. select specific Dongs to highlight)
st.sidebar.markdown("---")
all_dongs = sorted(df_dt['행정동명'].unique())
default_dongs = ['서교동', '역삼1동', '신촌동']
selected_dongs = st.sidebar.multiselect("관심 행정동 선택 (Time Pattern)", all_dongs, default=default_dongs)

# ------------------------------------------------------------------------------
# 4. Main Content
# ------------------------------------------------------------------------------

if selected_tab == "1. 이상치(Outlier) 분석":
    st.title("1. 이상치(Outlier) 분석 시각화")
    st.markdown("데이터의 분포, 변동성, 그리고 예외적인 패턴을 식별합니다.")

    # --- 1-1. Boxplot ---
    st.subheader("1-1. 행정동별 인구 분포 이상치 (Boxplot)")
    
    # Top 30 변동성 큰 동만 먼저 보여주거나, 전체는 너무 많으므로 상위 인구수 기준 필터링 또는 사용자 선택
    # 여기서는 시각적 명확성을 위해 평균 인구 상위 30개 동만 기본 표시
    top_dongs = df_dt.groupby('행정동명')['생활인구수'].mean().nlargest(30).index
    df_boxplot = df_dt[df_dt['행정동명'].isin(top_dongs)]
    
    fig_box = px.box(
        df_boxplot, x='행정동명', y='생활인구수',
        title="상위 30개 행정동(인구수 기준) 인구 분포 및 이상치",
        points="outliers" # show only outliers
    )
    st.plotly_chart(fig_box, use_container_width=True)
    
    with st.expander("💡 해석 및 인사이트 (상세 보기)", expanded=True):
        st.markdown("""
        > **해석 및 인사이트**: 이 그래프(Boxplot)는 각 동네에 사람이 얼마나 모이는지, 그 분포가 얼마나 다양한지를 보여줍니다.
        > *   **네모 박스(Box)**: 평소 가장 흔하게 관찰되는 인구 수의 범위(상위 25%~75%)입니다. 박스가 길면 시간대별로 인구 변화가 크다는 뜻입니다.
        > *   **점들(Points)**: 박스 위아래로 찍힌 점들은 '이상치(Outlier)'입니다. 평소보다 유독 사람이 폭발적으로 몰리거나, 너무 없는 아주 특별한 시간대가 있었다는 뜻입니다. 서교동, 역삼1동 같은 곳에 점이 많다는 건 그만큼 예측하기 힘든 돌발적인 인구 급증 현상(출퇴근, 축제, 금요일 밤 등)이 자주 일어난다는 증거입니다.
        """)

    st.divider()

    # --- 1-1-1 ~ 1-1-3. Histograms (Drill-down) ---
    st.subheader("1-1-X. 주요 지역 시간대별 생활인구 (Drill-down)")
    
    col1, col2, col3 = st.columns(3)
    
    # Function helper for histogram
    def plot_time_hist(dong_name, color_seq):
        data = df_dt[df_dt['행정동명'] == dong_name]
        fig = px.bar(data, x='시간대구분', y='생활인구수', title=f"{dong_name} 시간대별 평균 인구")
        fig.update_layout(showlegend=False, xaxis_title="시간 (0-23시)", yaxis_title="인구수")
        return fig

    with col1:
        st.plotly_chart(plot_time_hist("서교동", 'red'), use_container_width=True)
        with st.expander("서교동 인사이트"):
            st.write("홍대 상권로, 늦은 오후(17시~)부터 밤(22시~)까지 인구가 높게 유지됩니다. 특정 시간 급증(이상치) 현상이 평균에 희석되었으나 야간 활성도가 뚜렷합니다.")

    with col2:
        st.plotly_chart(plot_time_hist("신촌동", 'blue'), use_container_width=True)
        with st.expander("신촌동 인사이트"):
            st.write("주요 대학가(연세대, 이화여대)가 위치하여 주간 대학생 활동과 저녁 상권 이용객이 결합된 패턴을 보입니다.")

    with col3:
        st.plotly_chart(plot_time_hist("역삼1동", 'green'), use_container_width=True)
        with st.expander("역삼1동 인사이트"):
            st.write("테헤란로 중심 업무지구로, 08-09시 급증 후 일과 시간 동안 높게 유지되다가 퇴근 후 감소하는 전형적인 오피스 패턴입니다.")

    st.divider()

    # --- 1-2. Total Histogram ---
    st.subheader("1-2. 서울시 전체 시간대별 인구 밀도")
    total_hourly = df_dt.groupby('시간대구분')['생활인구수'].sum().reset_index()
    
    fig_total = px.bar(total_hourly, x='시간대구분', y='생활인구수', title="서울시 시간대별 총 생활인구")
    # Y축 범위 조정 (미세 변동 강조를 위함 - 데이터 min/max 고려)
    min_val = total_hourly['생활인구수'].min() * 0.95
    max_val = total_hourly['생활인구수'].max() * 1.05
    fig_total.update_layout(yaxis_range=[min_val, max_val])
    
    st.plotly_chart(fig_total, use_container_width=True)
    with st.expander("💡 해석 및 인사이트", expanded=True):
        st.markdown("""
        > **해석**: 06시경 최저점을 찍고 출근 시간(07-09시)에 급증하여 주간에 높게 유지되다가, 22시 이후 서서히 감소하는 패턴입니다. 전체 인구 대비 주야간 변동폭은 약 25만 명 수준으로, 거주 인구 기반 도시의 특성상 일정한 베이스라인이 존재합니다.
        """)

    st.divider()

    # --- 1-3. Z-Score Heatmap ---
    st.subheader("1-3. 시공간 이상치 히트맵 (Z-Score)")
    # Pivot for heatmap: Rows=Dong, Cols=Time, Values=Pop
    # Filter only top 50 dongs by pop size for performance/readability
    top_50_dongs = df_dt.groupby('행정동명')['생활인구수'].mean().nlargest(50).index
    df_heatmap_src = df_dt[df_dt['행정동명'].isin(top_50_dongs)].copy()
    
    # Calculate Z-score per Dong
    df_heatmap_src['z_score'] = df_heatmap_src.groupby('행정동명')['생활인구수'].transform(calculate_zscore)
    
    pivot_z = df_heatmap_src.pivot(index='행정동명', columns='시간대구분', values='z_score')
    
    fig_heat = px.imshow(
        pivot_z, 
        color_continuous_scale='RdBu_r', 
        aspect='auto',
        title="행정동별 시간대 인구 Z-Score (Top 50 Dong)"
    )
    st.plotly_chart(fig_heat, use_container_width=True)
    
    with st.expander("💡 해석 및 인사이트", expanded=True):
        st.markdown("""
        > **대학가(이문1, 안암, 대학, 화양)**: 늦은 오후~밤 시간대 붉은색(High Z-Score). 대학생 하교 활동 및 야간 상주 특성.
        > **주거지(화곡1, 상도1)**: 출근 시간대 푸른색(감소), 퇴근 후 붉은색(복귀)의 베드타운 패턴.
        > **업무지(역삼1)**: 09~18시 매우 진한 붉은색. 직장인 유입으로 인한 폭발적 인구 증가.
        """)

    st.divider()

    # --- 1-5. Volatility & Extreme ---
    st.subheader("1-5. 변동성 상위 20개 지역 및 극단값")
    
    # Volatility (Std Dev)
    dong_std = df_dt.groupby('행정동명')['생활인구수'].std().reset_index()
    top20_std = dong_std.nlargest(20, '생활인구수')
    
    fig_vol = px.bar(top20_std, x='행정동명', y='생활인구수', title="인구 변동성(표준편차) 상위 20개 지역")
    st.plotly_chart(fig_vol, use_container_width=True)
    
    with st.expander("💡 해석 및 인사이트"):
        st.markdown("""
        > **변동성**: 상위에 랭크된 지역은 낮/밤 또는 평일/주말 인구 차이가 극심한 유동적 지역(업무지구, 번화가)입니다.
        """)


elif selected_tab == "2. 심화 주제 분석":
    st.title("2. 심화 주제 분석")
    st.markdown("군집화, 연령/성별 특성 등 다각도의 심화 인사이트를 제공합니다.")

    # --- 2-1. Clustering ---
    st.subheader("2-1. 주거지구 vs 업무지구 클러스터링")
    
    # Calculate Day (09-18) vs Night (19-06) mean
    day_hours = list(range(9, 19))
    night_hours = list(range(19, 24)) + list(range(0, 7))
    
    df_day = df_dt[df_dt['시간대구분'].isin(day_hours)].groupby('행정동명')['생활인구수'].mean().rename('주간인구')
    df_night = df_dt[df_dt['시간대구분'].isin(night_hours)].groupby('행정동명')['생활인구수'].mean().rename('야간인구')
    
    df_cluster = pd.concat([df_day, df_night], axis=1).reset_index()
    
    fig_cluster = px.scatter(
        df_cluster, x='주간인구', y='야간인구', hover_name='행정동명',
        title="주간 평균 인구 vs 야간 평균 인구"
    )
    # y=x line
    max_val = max(df_cluster['주간인구'].max(), df_cluster['야간인구'].max())
    fig_cluster.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val, line=dict(color="Red", dash="dash"))
    
    st.plotly_chart(fig_cluster, use_container_width=True)
    
    with st.expander("💡 클러스터링 기반 추천 지역 (Top 3)", expanded=True):
        st.markdown("""
        > **대각선 아래(X축 방향, 업무지구)**: 여의동, 서교동, 종로1.2.3.4가동 (주간 인구 월등히 높음)
        > **대각선 위(Y축 방향, 주거지구)**: 당산2동, 개포1동, 세곡동 (야간 인구 월등히 높음, 베드타운)
        """)

    st.divider()

    # --- 2-2. Age Heatmap ---
    st.subheader("2-2. 연령대별 핫플레이스 (Top 30 Dong)")
    df_age = preprocess_for_age(raw_df)
    
    # Top 30 dongs only
    top_30_dongs = df_dt.groupby('행정동명')['생활인구수'].sum().nlargest(30).index
    df_age_filtered = df_age[df_age['행정동명'].isin(top_30_dongs)]
    
    # Pivot
    pivot_age = df_age_filtered.pivot(index='행정동명', columns='연령대', values='생활인구수')
    # Row normalization to see proportion within dong? Or compare to average? 
    # Report says "Normalized Heatmap" - let's normalize by max of each column to highlight where each age group is concentrated
    pivot_age_norm = pivot_age / pivot_age.max(axis=0)
    
    fig_age = px.imshow(pivot_age_norm, aspect='auto', title="행정동별 연령대 집중도 (Column Normalized)")
    st.plotly_chart(fig_age, use_container_width=True)
    
    with st.expander("💡 해석 및 인사이트"):
        st.markdown("붉은색이 진할수록 해당 지역에 특정 연령대가 서울시 타 지역 대비 많이 모여있음을 의미합니다. 20대 비중이 높은 '대학가/핫플'과 고령층 비중이 높은 지역이 구분됩니다.")

    st.divider()

    # --- 2-3. Gender Diverging ---
    st.subheader("2-3. 성별 초과 밀집 지역")
    df_gender = preprocess_for_gender(raw_df)
    
    df_gender_pivot = df_gender.pivot(index='행정동명', columns='성별', values='생활인구수').fillna(0)
    df_gender_pivot['diff'] = df_gender_pivot['남'] - df_gender_pivot['여']
    
    # Sort by diff
    top_male = df_gender_pivot.nlargest(10, 'diff')
    top_female = df_gender_pivot.nsmallest(10, 'diff')
    df_diverging = pd.concat([top_male, top_female]).sort_values('diff')
    
    fig_div = px.bar(
        df_diverging, x='diff', y=df_diverging.index, orientation='h',
        color='diff', color_continuous_scale='RdBu',
        title="성별 인구 차이 (남성 - 여성) Top/Bottom 10"
    )
    st.plotly_chart(fig_div, use_container_width=True)
    
    with st.expander("💡 해석 및 인사이트", expanded=True):
        st.markdown("0을 기준으로 오른쪽(파란색)은 남성 초과 밀집, 왼쪽(빨간색)은 여성 초과 밀집 지역입니다. 산업군별 특성이나 지역 인구 구성이 반영되었습니다.")

    st.divider()

    # --- 2-4. Pattern Comparison ---
    st.subheader("2-4. 주요 행정동 패턴 비교")
    
    if not selected_dongs:
        selected_dongs = ['역삼1동', '상도1동'] # Default fallback
        
    df_pattern = df_dt[df_dt['행정동명'].isin(selected_dongs)]
    
    fig_line = px.line(
        df_pattern, x='시간대구분', y='생활인구수', color='행정동명',
        title="선택된 행정동의 시간대별 인구 패턴 비교", markers=True
    )
    st.plotly_chart(fig_line, use_container_width=True)
    
    with st.expander("💡 해석 및 인사이트", expanded=True):
        st.markdown("""
        > **M자형 패턴**: 전형적인 업무지구 (출근/퇴근 피크).
        > **완만한 상승/하강**: 주거지구 특성.
        > 지역별 라이프스타일 차이를 시계열 곡선으로 확인할 수 있습니다.
        """)

    st.divider()

    # --- 2-5. Stability Matrix ---
    st.subheader("2-5. 안정성(Stability) vs 유동성(Mobility)")
    
    dong_stats = df_dt.groupby('행정동명')['생활인구수'].agg(['mean', 'std'])
    dong_stats['cv'] = dong_stats['std'] / dong_stats['mean'] # 변동계수
    
    fig_stab = px.scatter(
        dong_stats, x='mean', y='cv', hover_name=dong_stats.index,
        title="인구 규모(Mean) vs 변동성(CV) 매트릭스",
        labels={'mean': '평균 인구 규모', 'cv': '변동계수 (CV)'}
    )
    # Add quadrants lines (median based)
    fig_stab.add_vline(x=dong_stats['mean'].median(), line_dash="dash", line_color="gray")
    fig_stab.add_hline(y=dong_stats['cv'].median(), line_dash="dash", line_color="gray")
    
    st.plotly_chart(fig_stab, use_container_width=True)
    
    with st.expander("💡 해석 및 인사이트 (매트릭스 4분면)", expanded=True):
        st.markdown("""
        > **회기동, 안암동 (대학가)**: 인구 규모 중상, 변동계수 높음 (계절적/시간적 유동성).
        > **화양동, 신촌동, 서교동 (복합 상권)**: 우상향(인구 많음 + 변동 큼). 서울에서 가장 핫하고 에너지가 넘치는 핫플레이스. 거주보다 방문객 위주.
        """)
