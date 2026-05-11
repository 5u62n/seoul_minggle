import os
import requests
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import asyncio
import httpx

app = FastAPI()

# ✅ 1. CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 루트 경로 → index.html 서빙
@app.get("/")
async def root():
    return FileResponse("index.html")

# ✅ 2. 데이터 리스트
STATION_LIST = [
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

PLACE_LIST = ["강남 MICE 관광특구", "동대문 관광특구", "명동 관광특구", "이태원 관광특구", "잠실 관광특구", "종로·청계 관광특구", "홍대 관광특구",
             "경복궁", "광화문·덕수궁", "창덕궁·종묘", "가로수길", "노량진", "덕수궁길·정동길", "북촌한옥마을", "서촌", "성수카페거리",
             "압구정로데오거리", "여의도", "연남동", "영등포 타임스퀘어", "용리단길", "인사동", "해방촌·경리단길", "DDP(동대문디자인플라자)",
             "광화문광장", "국립중앙박물관·용산가족공원", "남산공원", "노들섬", "뚝섬한강공원", "망원한강공원", "반포한강공원",
             "서울숲공원", "어린이대공원", "여의도한강공원", "잠실종합운동장", "북창동 먹자골목", "남대문시장", "익선동", "송리단길·호수단길", "올림픽공원"]

# 친구 로직용 장소 분류
PLACE_TYPES = {
    "실외": ["경복궁", "광화문광장", "남산공원", "서울숲공원", "덕수궁길·정동길", "북촌한옥마을", "창덕궁·종묘"],
    "복합": ["강남 MICE 관광특구", "동대문 관광특구", "명동 관광특구", "이태원 관광특구", "잠실 관광특구", "홍대 관광특구", "가로수길", "성수카페거리", "압구정로데오거리", "용리단길", "인사동", "해방촌·경리단길", "익선동", "송리단길·호수단길", "서촌"],
    "실내": ["노량진", "영등포 타임스퀘어", "DDP(동대문디자인플라자)", "국립중앙박물관·용산가족공원", "북창동 먹자골목", "남대문시장", "광화문·덕수궁", "연남동", "잠실종합운동장", "여의도"]
}

TYPE_MAP = {name: t for t, names in PLACE_TYPES.items() for name in names}

# ✅ 3. 점수 계산 함수 (핵심! 이게 빠져있었어요)
def calculate_user_score(row, age_col):
    return round((row['NON_RESNT_PPLTN_RATE'] * 0.4) + (row[age_col] * 0.6), 2)

def calculate_friend_score(row):
    score = (row['NON_RESNT_PPLTN_RATE'] * 0.4 + row['PPLTN_RATE_30'] * 0.3 + row['MALE_PPLTN_RATE'] * 0.2)

    congest = row['AREA_CONGEST_LVL']
    if congest == '약간 붐빔': score -= 15
    elif congest == '붐빔': score -= 25
    elif congest == '매우 붐빔': score -= 50

    pm10 = row['PM10']
    weather_msg = row['AREA_CONGEST_MSG']
    is_raining = '비' in weather_msg or '눈' in weather_msg
    p_type = TYPE_MAP.get(row['AREA_NM'], "복합")

    if is_raining:
        if p_type == "실외": score = 0
        elif p_type == "복합": score *= 0.2
    elif pm10 >= 80:
        if p_type == "실외": score -= 20
        elif p_type == "복합": score -= 10

    return round(max(0, score), 2)

# ✅ 4. 비동기 데이터 수집 함수
async def fetch_seoul_data_async(client, api_key, area_nm):
    url = f'http://openapi.seoul.go.kr:8088/{api_key}/json/citydata/1/1/{area_nm}'
    try:
        res = await client.get(url, timeout=15)
        root = res.json().get('CITYDATA')
        if not root: return None
        ppl = root.get('LIVE_PPLTN_STTS', [{}])[0]
        weather = root.get('WEATHER_STTS', [{}])[0]
        return {
            'AREA_NM': area_nm,
            'AREA_CONGEST_LVL': ppl.get('AREA_CONGEST_LVL', ''),
            'AREA_CONGEST_MSG': ppl.get('AREA_CONGEST_MSG', ''),
            'NON_RESNT_PPLTN_RATE': float(ppl.get('NON_RESNT_PPLTN_RATE', 0)),
            'MALE_PPLTN_RATE': float(ppl.get('MALE_PPLTN_RATE', 0)),
            'PPLTN_RATE_10': float(ppl.get('PPLTN_RATE_10', 0)),
            'PPLTN_RATE_20': float(ppl.get('PPLTN_RATE_20', 0)),
            'PPLTN_RATE_30': float(ppl.get('PPLTN_RATE_30', 0)),
            'PPLTN_RATE_40': float(ppl.get('PPLTN_RATE_40', 0)),
            'PPLTN_RATE_50': float(ppl.get('PPLTN_RATE_50', 0)),
            'PM10': float(weather.get('PM10', 0))
        }
    except Exception as e:
        print(f"[ERROR] {area_nm}: {type(e).__name__}: {e}")
        return None

# ✅ 5. API 엔드포인트
@app.get("/recommend")
async def recommend(theme: str, age: str, gender: str):
    SEOUL_KEY = os.environ.get('dataseoul')
    openai_key = os.environ.get('openai_api_key')
    if not SEOUL_KEY or not openai_key:
        return {"error": "API 키 설정 누락"}

    client = OpenAI(api_key=openai_key)
    age_col = {"10대": "PPLTN_RATE_10", "20대": "PPLTN_RATE_20", "30대": "PPLTN_RATE_30", "40대": "PPLTN_RATE_40"}.get(age, "PPLTN_RATE_20")

    # 병렬 호출
    async with httpx.AsyncClient() as http:
        station_tasks = [fetch_seoul_data_async(http, SEOUL_KEY, s) for s in STATION_LIST]
        place_tasks   = [fetch_seoul_data_async(http, SEOUL_KEY, p) for p in PLACE_LIST]
        all_results   = await asyncio.gather(*station_tasks, *place_tasks)

    station_data = [r for r in all_results[:len(STATION_LIST)] if r]
    place_data   = [r for r in all_results[len(STATION_LIST):] if r]

    if not station_data:
        return {"error": "서울시 데이터 수집 실패"}

    df_stations = pd.DataFrame(station_data)
    df_stations['FINAL_SCORE'] = df_stations.apply(lambda r: calculate_user_score(r, age_col), axis=1)
    top_3_stations = df_stations.sort_values('FINAL_SCORE', ascending=False).head(3)

    df_places = pd.DataFrame(place_data)
    df_places['FINAL_SCORE'] = df_places.apply(calculate_friend_score, axis=1)
    top_3_places = df_places.sort_values('FINAL_SCORE', ascending=False).head(3)

    # AI 요약
    try:
        top_station = top_3_stations.iloc[0]
        prompt_s = f"{age} {gender}에게 '{theme}'를 위한 역 추천 1위 '{top_station['AREA_NM']}'의 추천 이유를 2줄로 써줘."
        res_s = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt_s}])
        station_summary = res_s.choices[0].message.content
    except Exception as e:
        traceback.print_exc()  # ← 이거 추가
        station_summary = "역 추천 요약을 가져오지 못했습니다."

    try:
        top_place = top_3_places.iloc[0]
        prompt_p = f"{age} {gender}에게 '{theme}'를 위한 장소 추천 1위 '{top_place['AREA_NM']}'의 추천 이유를 2줄로 써줘."
        res_p = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt_p}])
        place_summary = res_p.choices[0].message.content
    except Exception as e:
        traceback.print_exc()  # ← 이거 추가
        place_summary = "장소 추천 요약을 가져오지 못했습니다."

    final_result = pd.concat([top_3_stations, top_3_places])
    return {
        "top_station_summary": station_summary,
        "top_place_summary": place_summary,
        "recommendations": final_result[['AREA_NM', 'FINAL_SCORE', 'AREA_CONGEST_LVL']].to_dict(orient="records")
    }
