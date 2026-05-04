import os
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import pandas as pd
from fastapi import FastAPI
import src.mylib
from openai import OpenAI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

#프론트엔드 연동용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 모든 도메인에서의 접근을 허용합니다
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 데이터 수집 및 유틸리티 함수 ---
def buildUrl(key, type_, service, start, end, area_nm=''):
    _url = 'http://openAPI.seoul.go.kr:8088/'
    area_nm_encoded = urllib.parse.quote(area_nm)
    params = '/'.join([key, type_, service, str(start), str(end), '', '', area_nm_encoded])
    return urllib.parse.urljoin(_url, params)

def fetchAreaData(key, area_nm):
    """단일 지역 데이터 수집 (XML 파싱)"""
    url = buildUrl(key, 'xml', 'citydata', 1, 1, area_nm)
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            xml_data = response.read()
        root = ET.fromstring(xml_data)
        citydata = root.find('.//CITYDATA')
        ppltn = citydata.find('LIVE_PPLTN_STTS/LIVE_PPLTN_STTS') if citydata is not None else None
        
        if ppltn is None: return None

        def getText(node, tag):
            el = node.find(tag)
            return el.text if el is not None and el.text else '0'

        record = {
            'AREA_NM': getText(citydata, 'AREA_NM'),
            'AREA_CONGEST_LVL': getText(ppltn, 'AREA_CONGEST_LVL'),
            'AREA_PPLTN_MIN': float(getText(ppltn, 'AREA_PPLTN_MIN')),
            'AREA_PPLTN_MAX': float(getText(ppltn, 'AREA_PPLTN_MAX')),
            'MALE_PPLTN_RATE': float(getText(ppltn, 'MALE_PPLTN_RATE')),
            'FEMALE_PPLTN_RATE': float(getText(ppltn, 'FEMALE_PPLTN_RATE')),
            'NON_RESNT_PPLTN_RATE': float(getText(ppltn, 'NON_RESNT_PPLTN_RATE')),
            'PPLTN_RATE_0': float(getText(ppltn, 'PPLTN_RATE_0')),
            'PPLTN_RATE_10': float(getText(ppltn, 'PPLTN_RATE_10')),
            'PPLTN_RATE_20': float(getText(ppltn, 'PPLTN_RATE_20')),
            'PPLTN_RATE_30': float(getText(ppltn, 'PPLTN_RATE_30')),
            'PPLTN_RATE_40': float(getText(ppltn, 'PPLTN_RATE_40')),
            'PPLTN_RATE_50': float(getText(ppltn, 'PPLTN_RATE_50')),
            'PPLTN_RATE_60': float(getText(ppltn, 'PPLTN_RATE_60')),
            'PPLTN_RATE_70': float(getText(ppltn, 'PPLTN_RATE_70')),
        }
        
        # 하차 인원 합계 계산 (지하철 + 버스)
        sub = citydata.find('LIVE_SUB_PPLTN')
        bus = citydata.find('LIVE_BUS_PPLTN')
        sub_off = (float(getText(sub, 'SUB_30WTHN_GTOFF_PPLTN_MIN')) + float(getText(sub, 'SUB_30WTHN_GTOFF_PPLTN_MAX'))) / 2 if sub is not None else 0
        bus_off = (float(getText(bus, 'BUS_30WTHN_GTOFF_PPLTN_MIN')) + float(getText(bus, 'BUS_30WTHN_GTOFF_PPLTN_MAX'))) / 2 if bus is not None else 0
        record['GTOFF_AVG'] = sub_off + bus_off
        
        return record
    except Exception as e:
        return None

# --- 가중치 설정 ---
THEME_WEIGHTS = {
    "관광": {"non_res": 0.6, "ppltn": 0.3, "congest": -0.1, "gtoff": 0.0},
    "힐링": {"non_res": 0.1, "ppltn": 0.1, "congest": -0.7, "gtoff": 0.0},
    "맛집": {"non_res": 0.4, "ppltn": 0.5, "congest": -0.1, "gtoff": 0.0},
    "데이트": {"non_res": 0.4, "ppltn": 0.0, "congest": -0.2, "gtoff": 0.4}
}
AGE_WEIGHTS = {
    "10대": {"0" : 0.2, "10": 0.6, "20": 0.2},
    "20대": {"10": 0.2, "20": 0.6, "30": 0.2},
    "30대": {"20": 0.2, "30": 0.6, "40": 0.2},
    "40대": {"30": 0.2, "40": 0.6, "50": 0.2},
    "50대이상": {"40": 0.2, "50": 0.6, "60": 0.1, "70": 0.1}
}
GENDER_WEIGHTS = {
    "남성": {"MALE": 0.7, "FEMALE": 0.1, "PPLTN": 0.2},
    "여성": {"MALE": 0.1, "FEMALE": 0.7, "PPLTN": 0.2},
    "무관": {"MALE": 0.0, "FEMALE": 0.0, "PPLTN": 1.0}
}

# --- 핵심 추천 로직 ---
def get_final_recommendation(df, theme, age, gender):
    result_df = df.copy()
    
    # 테마 점수
    tw = THEME_WEIGHTS.get(theme, THEME_WEIGHTS["관광"])
    result_df['THEME_TOTAL'] = (
        (result_df['NON_RESNT_SCORE'] * tw.get('non_res', 0)) +
        (result_df['PPLTN_SCORE'] * tw.get('ppltn', 0)) +
        (result_df['CONGEST'] * tw.get('congest', 0)) +
        (result_df['GTOFF_SCORE'] * tw.get('gtoff', 0))
    )

    # 연령대 점수
    aw = AGE_WEIGHTS.get(age, {})
    result_df['AGE_TOTAL'] = 0
    for suffix, weight in aw.items():
        col = f"PPLTN_RATE_{suffix}"
        if col in result_df.columns:
            result_df['AGE_TOTAL'] += result_df[col] * weight

    # 성별 점수
    gw = GENDER_WEIGHTS.get(gender, GENDER_WEIGHTS["무관"])
    result_df['GENDER_TOTAL'] = (
        (result_df['FEMALE_PPLTN_RATE'] * gw.get('FEMALE', 0)) +
        (result_df['MALE_PPLTN_RATE'] * gw.get('MALE', 0))
    )

    result_df['FINAL_SCORE'] = result_df['THEME_TOTAL'] + result_df['AGE_TOTAL'] + result_df['GENDER_TOTAL']
    return result_df.sort_values('FINAL_SCORE', ascending=False).head(2)

#open ai연결
async def get_ai_explanation(client, area_name, theme, age):
    """추천된 장소에 대한 AI 요약 생성"""
    try:
        # AI에게 줄 미션(프롬프트)
        prompt = (
            f"당신은 서울 핫플레이스 가이드입니다. "
            f"'{age}' 연령층이 '{theme}'를 목적으로 '{area_name}'을(를) 방문하려고 합니다. "
            f"이 장소가 왜 이 사람에게 최고의 선택인지 딱 2줄로 친절하게 설명해주세요."
        )
        
        response = client.chat.completions.create(
            model="gpt-5.4-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"{area_name}은 현재 많은 분들이 찾고 있는 인기 지역입니다!"     #에러 발생했을 때


# --- API 엔드포인트 ---
@app.get("/recommend")
async def recommend(theme: str, age: str, gender: str):
    # 1. 키 로드 및 클라이언트 설정
    keyPath = os.path.join(os.getcwd(), 'src', 'key.properties')
    key_info = src.mylib.getKey(keyPath)
    
    # 서울시 키와 OpenAI 키를 각각 안전하게 가져옵니다
    SEOUL_KEY = key_info['dataseoul']
    client = OpenAI(api_key=key_info['openai_api_key'])
    
    AREA_LIST = [
        '가산디지털단지역', '강남역', '건대입구역', '고덕역', '고속터미널역',
        '교대역', '구로디지털단지역', '구로역', '군자역', '대림역',
        '동대문역', '뚝섬역', '미아사거리역', '발산역', '사당역',
        '삼각지역', '서울대입구역', '서울식물원·마곡나루역', '서울역', '선릉역',
        '성신여대입구역', '수유역', '신논현역·논현역', '신도림역', '신림역',
        '신촌·이대역', '양재역', '역삼역', '연신내역', '오목교역·목동운동장',
        '왕십리역', '용산역', '이태원역', '장지역', '장한평역',
        '천호역', '총신대입구(이수)역', '충정로역', '합정역', '혜화역',
        '홍대입구역(2호선)', '회기역', '쌍문역', '신정네거리역',
        '잠실새내역', '잠실역', '시의회 앞', '숭례문'
    ]
    
    # 2. 데이터 수집 (KEY 변수 오류 수정)
    rows = [fetchAreaData(SEOUL_KEY, area) for area in AREA_LIST]
    df = pd.DataFrame([r for r in rows if r])

    if df.empty: 
        return {"error": "데이터를 불러올 수 없습니다."}

    # 3. 실시간 정규화 및 점수 계산
    df['PPLTN_AVG'] = (df['AREA_PPLTN_MIN'] + df['AREA_PPLTN_MAX']) / 2
    
    def normalize(series):
        diff = series.max() - series.min()
        return (series - series.min()) / diff * 100 if diff != 0 else 0

    df['PPLTN_SCORE'] = normalize(df['PPLTN_AVG'])
    df['GTOFF_SCORE'] = normalize(df['GTOFF_AVG'])
    df['NON_RESNT_SCORE'] = normalize(df['NON_RESNT_PPLTN_RATE'])
    
    congest_map = {'여유': 0, '보통': 33, '약간 붐빔': 66, '붐빔': 100}
    df['CONGEST'] = df['AREA_CONGEST_LVL'].map(congest_map)

    # 4. 최종 추천 실행
    result = get_final_recommendation(df, theme, age, gender)
    
    # 5. AI 한줄평 생성 (1위 장소 대상)
    top_place_name = result.iloc[0]['AREA_NM']
    ai_summary = await get_ai_explanation(client, top_place_name, theme, age)
    
    # 6. 결과 반환
    return {
        "top_place_summary": ai_summary,
        "recommendations": result[['AREA_NM', 'FINAL_SCORE']].to_dict(orient="records")
    }
