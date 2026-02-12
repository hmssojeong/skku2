import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from openai import OpenAI

# 1. 페이지 설정
st.set_page_config(page_title="RPG 성장 & 도파민 설계기 Pro", layout="wide")

# 2. API 설정 및 Session State 초기화
client = OpenAI(api_key=st.secrets["API_KEY"])

# [게임사별 전략 프리셋 정의]
PRESETS = {
    "Custom (직접 설정)": None,
    "🍁 MapleStory형 (강화/확률 중심)": {
        "base_atk": 10, "target_atk": 1000, "curve_type": "Exponential",
        "prob_legend": 0.015, "pity_count": 500,
        "enhance_prob": 30.0, "enhance_destroy": 5.0, "monster_hp": 800,
        "desc": "낮은 성공 확률과 파괴 리스크를 통한 하이리스크 하이리턴 강화 시스템입니다."
    },
    "😈 Diablo형 (파밍/드랍 중심)": {
        "base_atk": 25, "target_atk": 700, "curve_type": "Logarithmic",
        "prob_legend": 2.5, "pity_count": 1000,
        "enhance_prob": 85.0, "enhance_destroy": 0.0, "monster_hp": 400,
        "desc": "드랍 빈도는 높지만 유효 옵션 획득을 어렵게 설계한 파밍 최적화 시스템입니다."
    },
    "✨ 원신/가챠형 (천장/수집 중심)": {
        "base_atk": 40, "target_atk": 500, "curve_type": "S-Curve",
        "prob_legend": 0.6, "pity_count": 90,
        "enhance_prob": 100.0, "enhance_destroy": 0.0, "monster_hp": 1200,
        "desc": "기초 확률은 낮으나 확실한 천장 시스템을 통해 유저의 심리적 저항선을 관리합니다."
    },
    "💤 방치형 RPG (무한 성장 중심)": {
        "base_atk": 100, "target_atk": 9999, "curve_type": "Exponential",
        "prob_legend": 1.0, "pity_count": 100,
        "enhance_prob": 60.0, "enhance_destroy": 0.0, "monster_hp": 4500,
        "desc": "인플레이션 수치가 기하급수적으로 상승하며 특정 구간 '벽'을 돌파하는 재미를 줍니다."
    }
}

# [세션 상태 통합 초기화]
initial_states = {
    'base_atk': 10, 'atk_speed': 1.0, 'crit_rate': 10.0, 'crit_dmg': 150.0,
    'max_level': 50, 'target_atk': 500, 'curve_type': "Exponential",
    'monster_hp': 100, 'monster_def': 0, 'current_monster_hp': 100.0,
    'prob_legend': 0.1, 'pity_count': 100,
    'enhance_prob': 50.0, 'enhance_penalty': True, 'enhance_destroy': 1.0,
    'battle_log': [], 'current_preset': "Custom (직접 설정)"
}
for key, value in initial_states.items():
    if key not in st.session_state:
        st.session_state[key] = value


# [체력 동기화 콜백 함수]
def on_hp_change():
    # 사이드바에서 입력된 새로운 monster_hp 값을 즉시 현재 체력에 반영
    st.session_state.current_monster_hp = st.session_state.hp_input_key


# 3. 함수 정의: AI 통합 분석
def analyze_intent(user_query):
    prompt = f"""
    당신은 게임 밸런스 디자이너입니다. 사용자의 의도를 분석해 RPG 시스템 파라미터를 JSON으로 반환하세요.
    [사용자 요청]: "{user_query}"
    [반환 JSON 형식]:
    {{
        "base_atk": int, "max_level": int, "target_atk": int,
        "curve_type": "Exponential" | "Logarithmic" | "S-Curve",
        "prob_legend": float, "pity_count": int,
        "enhance_prob": float, "enhance_destroy": float,
        "monster_hp": int, "reason": "기획적 이유 요약"
    }}
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are a master game designer."},
                  {"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)


# 4. 수학적 시뮬레이션 함수들
def calculate_growth(base, max_lvl, target, curve):
    levels = np.arange(1, max_lvl + 1)
    if curve == "Exponential":
        r = (target / base) ** (1 / (max_lvl - 1)) if max_lvl > 1 else 1
        atk_values = base * (r ** (levels - 1))
    elif curve == "Logarithmic":
        atk_values = base + (target - base) * (np.log(levels) / np.log(max_lvl))
    else:  # S-Curve
        atk_values = base + (target - base) / (1 + np.exp(-0.2 * (levels - max_lvl / 2)))
    return levels, atk_values, 100 * (levels ** 1.5), atk_values * 4


def simulate_gacha(prob, pity):
    n = np.arange(1, pity + 1)
    p = prob / 100
    cum_prob = 1 - (1 - p) ** n
    if len(cum_prob) > 0:
        cum_prob[-1] = 1.0
    return n, cum_prob


def calculate_combat_metrics():
    crit_factor = 1 + (st.session_state.crit_rate / 100) * (st.session_state.crit_dmg / 100 - 1)
    avg_dmg = max(1, (st.session_state.base_atk - st.session_state.monster_def)) * crit_factor
    dps = avg_dmg * st.session_state.atk_speed
    hits_to_kill = np.ceil(st.session_state.monster_hp / avg_dmg)
    time_to_kill = hits_to_kill / st.session_state.atk_speed
    return avg_dmg, dps, hits_to_kill, time_to_kill


# 5. 사이드바 제어
st.sidebar.header("🕹️ 메이저 게임 프리셋")
selected_preset = st.sidebar.selectbox("밸런스 전략 선택", list(PRESETS.keys()))

# 프리셋 변경 감지 및 적용
if selected_preset != st.session_state.current_preset:
    data = PRESETS[selected_preset]
    if data:
        for k, v in data.items():
            if k in st.session_state: st.session_state[k] = v
        st.session_state.current_monster_hp = st.session_state.monster_hp
    st.session_state.current_preset = selected_preset
    st.rerun()

if PRESETS[selected_preset]:
    st.sidebar.info(f"💡 {PRESETS[selected_preset]['desc']}")

st.sidebar.divider()
st.sidebar.header("🛠️ 시스템 밸런스 제어")
mode = st.sidebar.radio("작업 모드", ["📈 성장 밸런스", "⚔️ 전투 시뮬레이터", "🎰 가챠 확률", "🔥 강화 리스크"])

# 6. 메인 로직
if mode == "📈 성장 밸런스":
    st.title("📊 캐릭터 성장 & 밸런스 곡선")
    st.session_state.base_atk = st.sidebar.number_input("레벨 1 공격력", value=st.session_state.base_atk)
    st.session_state.max_level = st.sidebar.slider("최대 레벨", 10, 100, value=st.session_state.max_level)
    st.session_state.curve_type = st.sidebar.selectbox("곡선 타입", ["Exponential", "Logarithmic", "S-Curve"],
                                                       index=["Exponential", "Logarithmic", "S-Curve"].index(
                                                           st.session_state.curve_type))
    st.session_state.target_atk = st.sidebar.number_input("만렙 공격력", value=st.session_state.target_atk)

    levels, atk_vals, exp_vals, mhp_vals = calculate_growth(st.session_state.base_atk, st.session_state.max_level,
                                                            st.session_state.target_atk, st.session_state.curve_type)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("성장 곡선 시각화")
        fig, ax1 = plt.subplots()
        ax1.plot(levels, atk_vals, label="Atk", color='#1f77b4', linewidth=2)
        ax1.plot(levels, mhp_vals, label="Monster HP", color='#ff7f0e', linestyle='--')
        ax2 = ax1.twinx()
        ax2.fill_between(levels, exp_vals, alpha=0.1, color='#2ca02c', label="Exp")
        st.pyplot(fig)
    with col2:
        st.subheader("🤖 AI 밸런스 진단")
        if st.button("성장 밸런스 분석"):
            with st.spinner("📊 성장 곡선을 정밀 분석 중입니다..."):
                res = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user",
                                                                                     "content": f"Atk {st.session_state.base_atk}~{st.session_state.target_atk} ({st.session_state.curve_type}) 분석해줘."}])
                st.write(res.choices[0].message.content)

elif mode == "⚔️ 전투 시뮬레이터":
    st.title("🎮 전투 시뮬레이터 & 도파민 체크")

    st.sidebar.subheader("전투 세부 설정")
    st.session_state.base_atk = st.sidebar.number_input("현재 공격력", value=st.session_state.base_atk)
    st.session_state.atk_speed = st.sidebar.slider("공격 속도 (회/초)", 0.1, 10.0, value=st.session_state.atk_speed)
    st.session_state.crit_rate = st.sidebar.slider("치명타 확률 (%)", 0.0, 100.0, value=st.session_state.crit_rate)

    # [중요 수정] number_input에 key와 on_change 콜백을 추가하여 수치 변경 즉시 체력 동기화
    st.session_state.monster_hp = st.sidebar.number_input(
        "몬스터 HP",
        value=st.session_state.monster_hp,
        key="hp_input_key",
        on_change=on_hp_change
    )

    avg_dmg, dps, hits, kill_time = calculate_combat_metrics()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("평균 데미지", f"{avg_dmg:.1f}")
    c2.metric("최종 DPS", f"{dps:.1f}")
    c3.metric("처치 타수", f"{int(hits)} 방")
    c4.metric("처치 시간", f"{kill_time:.2f} 초")

    st.divider()

    # --- 실시간 인터랙티브 시뮬레이션 구역 ---
    st.subheader("🕹️ 실시간 타격 시뮬레이션")
    sim_col1, sim_col2 = st.columns([1, 1])

    with sim_col1:
        st.write(f"캐릭터 (전략: {st.session_state.current_preset})")
        st.image("https://api.dicebear.com/7.x/adventurer/svg?seed=Hero", width=150)
        if st.button("⚔️ 공격 하기", use_container_width=True):
            is_crit = np.random.rand() < (st.session_state.crit_rate / 100)
            final_dmg = st.session_state.base_atk * (st.session_state.crit_dmg / 100 if is_crit else 1)
            st.session_state.current_monster_hp -= final_dmg
            crit_txt = "💥CRITICAL! " if is_crit else ""
            st.session_state.battle_log.insert(0, f"{crit_txt}플레이어가 {final_dmg:.1f}의 데미지를 주었습니다.")
            if st.session_state.current_monster_hp <= 0:
                st.session_state.current_monster_hp = st.session_state.monster_hp
                st.session_state.battle_log.insert(0, "🎊 몬스터 처치! 새로운 몬스터 등장.")
                st.balloons()

    with sim_col2:
        st.write("몬스터")
        st.image("https://api.dicebear.com/7.x/bottts/svg?seed=Monster", width=150)

        # 현재 체력이 최대 체력을 초과하지 않도록 보정하여 프로그레스 바 에러 방지
        hp_ratio = max(0.0, min(1.0, st.session_state.current_monster_hp / st.session_state.monster_hp))
        st.progress(hp_ratio)

        # 현재 체력 표시 (최대 체력을 넘으면 최대 체력으로 보이게 처리)
        display_hp = min(st.session_state.current_monster_hp, st.session_state.monster_hp)
        st.write(f"HP: {max(0.0, display_hp):.1f} / {st.session_state.monster_hp}")

        for log in st.session_state.battle_log[:3]:
            st.caption(log)

    st.divider()

    col_v1, col_v2 = st.columns([1, 1])
    with col_v1:
        st.subheader("⚔️ 전투 체감 가이드")
        if hits <= 2:
            st.success("🎯 [원샷원킬] 압도적인 도파민!")
        elif hits <= 6:
            st.info("⚡ [쾌속 사냥] 경쾌한 속도감.")
        elif hits <= 15:
            st.warning("🐢 [정체 구간] 지루함 유발.")
        else:
            st.error("🛑 [절망 구간] 유저 이탈 리스크 매우 높음.")

    with col_v2:
        st.subheader("🤖 AI 전투 체감 진단")
        if st.button("AI 심층 분석"):
            with st.spinner("⚔️ 전투 데이터를 기반으로 도파민 수치를 계산 중..."):
                prompt = f"DPS {dps}, 처치타수 {hits}, 처치시간 {kill_time}초 분석해줘."
                res = client.chat.completions.create(model="gpt-4o-mini",
                                                     messages=[{"role": "user", "content": prompt}])
                st.write(res.choices[0].message.content)

elif mode == "🎰 가챠 확률":
    st.title("🎰 가챠 드랍 & 도파민 설계기 Pro")

    # 1. 입력 섹션
    st.sidebar.subheader("가챠 확률 세부 설정")
    st.session_state.prob_legend = st.sidebar.number_input("레전드 확률 (%)", value=st.session_state.prob_legend,
                                                           format="%.4f")
    st.session_state.pity_count = st.sidebar.number_input("천장 횟수 (Pity)", value=st.session_state.pity_count)

    # 2. 통계 계산
    p = st.session_state.prob_legend / 100
    avg_tries = 1 / p if p > 0 else 0
    # 50% 확률 도달 시점: (1-p)^n = 0.5 -> n = log(0.5) / log(1-p)
    median_tries = np.log(0.5) / np.log(1 - p) if p > 0 else 0
    # 95% 신뢰구간 시점
    conf_95_tries = np.log(0.05) / np.log(1 - p) if p > 0 else 0

    # 3. 상단 핵심 지표
    m1, m2, m3 = st.columns(3)
    m1.metric("평균 획득 시도", f"{avg_tries:.1f}회")
    m2.metric("50% 유저가 얻는 시점", f"{int(median_tries)}회")
    m3.metric("95% 유저가 얻는 시점", f"{int(conf_95_tries)}회")

    st.divider()

    # 4. 그래프 섹션
    col1, col2 = st.columns(2)
    n, cum_p = simulate_gacha(st.session_state.prob_legend, int(conf_95_tries * 1.2) if p > 0 else 100)

    with col1:
        st.subheader("🎲 도파민 누적 확률 곡선")

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(n, cum_p * 100, color='#e74c3c', linewidth=2, label="누적 확률")
        ax.axhline(50, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(median_tries, color='blue', linestyle=':', label="50% 지점")
        ax.fill_between(n, 0, cum_p * 100, alpha=0.1, color='#e74c3c')
        ax.set_xlabel("시도 횟수")
        ax.set_ylabel("획득 성공 확률 (%)")
        ax.legend()
        st.pyplot(fig)

    with col2:
        st.subheader("🤖 AI 심리 리스크 분석")
        if st.button("도파민 커브 정밀 진단"):
            with st.spinner("🎰 확률 통계를 바탕으로 유저 이탈 리스크를 진단 중..."):
                risk_prompt = f"""
                확률 {st.session_state.prob_legend}%, 천장 {st.session_state.pity_count}회입니다.
                1. 유저가 느끼는 박탈감 구간
                2. 기대 좌절 지점
                3. 이탈 위험도
                를 전문적인 게임 밸런스 디자이너 관점에서 분석해줘.
                """
                res = client.chat.completions.create(model="gpt-4o-mini",
                                                     messages=[{"role": "user", "content": risk_prompt}])
                st.info(res.choices[0].message.content)

    # 5. 기대 좌절 구간 분석 시각화
    st.subheader("⚠️ 기대 좌절 & 폭발 구간 모니터링")
    frustration_idx = int(avg_tries * 0.7)
    st.warning(f"💡 현재 설계상 **{frustration_idx}회~{int(avg_tries)}회** 구간이 유저의 '기대 좌절'이 가장 큰 구간입니다. (평균에 근접함에도 못 얻는 유저 속출)")

elif mode == "🔥 강화 리스크":
    st.title("🔥 강화 시스템 리스크 설계")
    st.session_state.enhance_prob = st.sidebar.slider("성공 확률 (%)", 1.0, 100.0, value=st.session_state.enhance_prob)
    st.session_state.enhance_destroy = st.sidebar.slider("파괴 확률 (%)", 0.0, 10.0, value=st.session_state.enhance_destroy)
    col1, col2 = st.columns(2)
    with col1:
        exp_cost = 1 / (st.session_state.enhance_prob / 100)
        st.metric("평균 시도 횟수", f"{exp_cost:.2f} 회")
    with col2:
        st.subheader("🤖 AI 강화 경제 분석")
        if st.button("강화 시스템 진단"):
            with st.spinner("🔥 강화 리스크 및 경제 밸런스를 분석 중..."):
                res = client.chat.completions.create(model="gpt-4o-mini", messages=[
                    {"role": "user", "content": f"성공률 {st.session_state.enhance_prob}% 분석해줘."}])
                st.write(res.choices[0].message.content)

# 7. 하단 공통 구역
st.divider()
tab1, tab2, tab3 = st.tabs(["💡 AI 자연어 설정", "🎮 Unity C# 코드", "📄 데이터 확인"])

with tab1:
    user_input = st.text_input("의도 입력", placeholder="예: 초반엔 잘 나오다가 후반에 희귀템이 터지는 느낌")
    if st.button("AI 자동 설계 적용"):
        if user_input:
            with st.spinner("🤖 입력하신 의도를 시스템 파라미터로 변환 중입니다..."):
                result = analyze_intent(user_input)
                st.session_state.update(result)
                st.session_state.current_monster_hp = result.get('monster_hp', 100)
                st.success(f"✅ 반영 완료: {result['reason']}")
                st.rerun()

with tab2:
    st.subheader("Unity C# 통합 매니저 (시스템별 공식 적용)")

    # 성장 곡선 공식 문자열 선택
    if st.session_state.curve_type == "Exponential":
        r = (st.session_state.target_atk / st.session_state.base_atk) ** (
                1 / (st.session_state.max_level - 1)) if st.session_state.max_level > 1 else 1
        growth_formula = f"return baseAtk * Mathf.Pow({r:.4f}f, level - 1);"
    elif st.session_state.curve_type == "Logarithmic":
        growth_formula = f"return baseAtk + (targetAtk - baseAtk) * (Mathf.Log(level) / Mathf.Log(maxLevel));"
    else:  # S-Curve
        growth_formula = f"return baseAtk + (targetAtk - baseAtk) / (1.0f + Mathf.Exp(-0.2f * (level - maxLevel * 0.5f)));"

    st.code(f"""
using UnityEngine;

public class GameBalanceManager : MonoBehaviour 
{{
    [Header("Current Strategy: {selected_preset}")]

    // 1. 성장 시스템 (공격력 계산)
    public float GetAttackValue(int level, float baseAtk, float targetAtk, int maxLevel) 
    {{
        if (level <= 1) return baseAtk;
        {growth_formula}
    }}

    // 2. 가챠 시스템 (천장 포함)
    public bool TryGacha(int currentPityCount) 
    {{
        float successProb = {st.session_state.prob_legend / 100}f; // {st.session_state.prob_legend}%
        int pityThreshold = {st.session_state.pity_count};

        // 천장 체크
        if (currentPityCount >= pityThreshold) return true;

        // 난수 체크
        return Random.value <= successProb;
    }}

    // 3. 강화 시스템 (파괴 리스크 포함)
    public EnhanceResult UpgradeItem() 
    {{
        float successRate = {st.session_state.enhance_prob / 100}f; // {st.session_state.enhance_prob}%
        float destroyRate = {st.session_state.enhance_destroy / 100}f; // {st.session_state.enhance_destroy}%

        float roll = Random.value;

        if (roll <= successRate) return EnhanceResult.Success;
        if (roll >= (1.0f - destroyRate)) return EnhanceResult.Destroyed;

        return EnhanceResult.Fail;
    }}

    public enum EnhanceResult {{ Success, Fail, Destroyed }}
}}
""", language='csharp')

with tab3:
    if mode == "📈 성장 밸런스":
        st.dataframe(pd.DataFrame({"Level": np.arange(1, st.session_state.max_level + 1), "Atk":
            calculate_growth(st.session_state.base_atk, st.session_state.max_level, st.session_state.target_atk,
                             st.session_state.curve_type)[1]}))
    else:
        st.write(f"현재 선택된 전략: {st.session_state.current_preset}")