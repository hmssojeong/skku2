"""
RPG 성장 & 도파민 설계기 Pro (v3.0)
─────────────────────────────────────────────────────────────
v3.0 수정 사항:
  1. [버그수정] 강화: Python 시뮬 ↔ Unity 판정 로직 B안(꼬리형)으로 완전 통일
     - roll < sp → 성공 / roll >= 1-dp → 파괴 / else → 실패
     - sp+dp > 1 방지 안전 캡 적용
     - Tab3 단계 테이블도 동일 규칙 사용
  2. [버그수정] 가챠 상단 지표: 이론값(고정확률) + 실제값(시뮬 기반) 병행 표기
  3. [성능] 몬테카를로 n_sim 자동 축소 (pity 크면 표본 감소) + soft pity OFF시 벡터화
  4. [개선] TTK 역산: 연속/이산 모드 토글 + 검증 표시
  5. [안정성] OpenAI 클라이언트 레벨 timeout 설정
  6. [UX] battle_log 200개 상한 / 몬스터 리셋 버튼 / 시뮬 seed 토글
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import json
import platform
from openai import OpenAI

# ═══════════════════════════════════════════════════════════════
# 0. 초기 설정
# ═══════════════════════════════════════════════════════════════
def set_korean_font():
    sys = platform.system()
    if sys == "Windows":
        plt.rc('font', family='Malgun Gothic')
    elif sys == "Darwin":
        plt.rc('font', family='AppleGothic')
    else:
        plt.rc('font', family='NanumGothic')
    plt.rcParams['axes.unicode_minus'] = False

set_korean_font()
st.set_page_config(page_title="RPG 성장 & 도파민 설계기 Pro", layout="wide")

# ═══════════════════════════════════════════════════════════════
# 1. API – [수정5] 클라이언트 레벨 timeout
# ═══════════════════════════════════════════════════════════════
try:
    client = OpenAI(
        api_key=st.secrets["API_KEY"],
        timeout=25.0,          # 클라이언트 레벨 타임아웃 (SDK 지원 방식)
        max_retries=1,
    )
except Exception:
    client = None

def safe_ai_call(messages: list, model: str = "gpt-4o-mini") -> str | None:
    """AI 호출 래퍼 – 예외를 사용자 친화적 메시지로 변환"""
    if client is None:
        st.error("❌ OpenAI 클라이언트 초기화 실패. secrets.toml의 API_KEY를 확인해주세요.")
        return None
    try:
        res = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        return res.choices[0].message.content
    except Exception as e:
        msg = str(e)
        if "timeout" in msg.lower():
            st.error("⏱️ AI 응답 시간이 초과됐습니다. 잠시 후 다시 시도해주세요.")
        elif "api_key" in msg.lower() or "authentication" in msg.lower():
            st.error("🔑 API 키 오류입니다. Streamlit Secrets를 확인해주세요.")
        elif "rate_limit" in msg.lower():
            st.error("🚦 API 요청 한도 초과입니다. 잠시 후 다시 시도해주세요.")
        else:
            st.error(f"❌ AI 호출 실패: {msg[:150]}")
        return None

# ═══════════════════════════════════════════════════════════════
# 2. 프리셋 정의
# ═══════════════════════════════════════════════════════════════
PRESETS = {
    "Custom (직접 설정)": None,
    "🍁 MapleStory형 (강화/확률 중심)": {
        "base_atk": 10, "target_atk": 1000, "curve_type": "Exponential",
        "prob_legend": 0.015, "pity_count": 500,
        "soft_pity_enable": False, "soft_pity_start": 400,
        "enhance_prob": 30.0, "enhance_destroy": 5.0,
        "enhance_max_stage": 17, "safeguard_enable": True, "safeguard_stage": 12,
        "monster_hp": 800, "monster_def": 20,
        "atk_speed": 1.5, "crit_rate": 15.0, "crit_dmg": 200.0, "max_level": 250,
        "desc": "낮은 성공 확률과 파괴 리스크를 통한 하이리스크 하이리턴 강화 시스템입니다."
    },
    "😈 Diablo형 (파밍/드랍 중심)": {
        "base_atk": 25, "target_atk": 700, "curve_type": "Logarithmic",
        "prob_legend": 2.5, "pity_count": 1000,
        "soft_pity_enable": False, "soft_pity_start": 800,
        "enhance_prob": 85.0, "enhance_destroy": 0.0,
        "enhance_max_stage": 10, "safeguard_enable": False, "safeguard_stage": 8,
        "monster_hp": 400, "monster_def": 10,
        "atk_speed": 2.0, "crit_rate": 25.0, "crit_dmg": 175.0, "max_level": 100,
        "desc": "드랍 빈도는 높지만 유효 옵션 획득을 어렵게 설계한 파밍 최적화 시스템입니다."
    },
    "✨ 원신/가챠형 (천장/수집 중심)": {
        "base_atk": 40, "target_atk": 500, "curve_type": "S-Curve",
        "prob_legend": 0.6, "pity_count": 90,
        "soft_pity_enable": True, "soft_pity_start": 74,
        "enhance_prob": 100.0, "enhance_destroy": 0.0,
        "enhance_max_stage": 5, "safeguard_enable": False, "safeguard_stage": 4,
        "monster_hp": 1200, "monster_def": 30,
        "atk_speed": 1.0, "crit_rate": 5.0, "crit_dmg": 150.0, "max_level": 90,
        "desc": "기초 확률은 낮으나 확실한 천장 시스템을 통해 유저의 심리적 저항선을 관리합니다."
    },
    "💤 방치형 RPG (무한 성장 중심)": {
        "base_atk": 100, "target_atk": 9999, "curve_type": "Exponential",
        "prob_legend": 1.0, "pity_count": 100,
        "soft_pity_enable": False, "soft_pity_start": 80,
        "enhance_prob": 60.0, "enhance_destroy": 0.0,
        "enhance_max_stage": 20, "safeguard_enable": False, "safeguard_stage": 15,
        "monster_hp": 4500, "monster_def": 50,
        "atk_speed": 3.0, "crit_rate": 20.0, "crit_dmg": 160.0, "max_level": 999,
        "desc": "인플레이션 수치가 기하급수적으로 상승하며 특정 구간 '벽'을 돌파하는 재미를 줍니다."
    }
}

# ═══════════════════════════════════════════════════════════════
# 3. 세션 상태 초기화
# ═══════════════════════════════════════════════════════════════
DEFAULTS: dict = {
    'base_atk': 10, 'atk_speed': 1.0, 'crit_rate': 10.0, 'crit_dmg': 150.0,
    'monster_def': 0,
    'max_level': 50, 'target_atk': 500, 'curve_type': "Exponential",
    'ttk_mode': False, 'target_ttk': 5.0, 'ttk_discrete': False,
    'monster_hp': 100, 'current_monster_hp': 100.0, 'battle_log': [],
    'prob_legend': 0.1, 'pity_count': 100,
    'soft_pity_enable': False, 'soft_pity_start': 75,
    'enhance_prob': 50.0, 'enhance_destroy': 1.0,
    'enhance_max_stage': 15, 'safeguard_enable': False, 'safeguard_stage': 12,
    'sim_random_seed': True,   # [수정6] seed 토글
    'current_preset': "Custom (직접 설정)",
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ═══════════════════════════════════════════════════════════════
# 4. 핵심 수학·시뮬 함수
# ═══════════════════════════════════════════════════════════════

def calculate_growth(base: float, max_lvl: int, target: float, curve: str):
    """레벨별 (공격력, 경험치) 계산"""
    levels = np.arange(1, max_lvl + 1)
    if curve == "Exponential":
        r = (target / base) ** (1.0 / (max_lvl - 1)) if max_lvl > 1 else 1.0
        atk = base * (r ** (levels - 1))
    elif curve == "Logarithmic":
        atk = base + (target - base) * (np.log(levels) / np.log(max_lvl))
    else:  # S-Curve
        atk = base + (target - base) / (1.0 + np.exp(-0.2 * (levels - max_lvl / 2.0)))
    exp = 100.0 * (levels ** 1.5)
    return levels, atk, exp


# [수정4] TTK 역산: 연속/이산 모드 지원
def monster_hp_from_ttk(atk_vals: np.ndarray,
                         target_ttk: float,
                         atk_speed: float,
                         crit_rate: float,
                         crit_dmg: float,
                         monster_def: float,
                         discrete_mode: bool = False) -> np.ndarray:
    """
    목표 TTK(초) 기반 몬스터 HP 역산
    discrete_mode=True → 이산 타격(타수 ceil) 기반 계산
    """
    crit_factor = 1.0 + (crit_rate / 100.0) * (crit_dmg / 100.0 - 1.0)
    avg_dmg = np.maximum(1.0, atk_vals - monster_def) * crit_factor
    if discrete_mode:
        target_hits = np.ceil(target_ttk * max(0.001, atk_speed))
        return np.maximum(1.0, avg_dmg * target_hits)
    else:
        dps = np.maximum(0.001, avg_dmg * atk_speed)
        return np.maximum(1.0, dps * target_ttk)


def calculate_combat_metrics(base_atk: float, monster_def: float,
                              crit_rate: float, crit_dmg: float,
                              atk_speed: float, monster_hp: float):
    """전투 기댓값 (do_attack과 완전히 동일한 공식)"""
    base_dmg = max(1.0, float(base_atk) - float(monster_def))
    crit_factor = 1.0 + (crit_rate / 100.0) * (crit_dmg / 100.0 - 1.0)
    avg_dmg = max(1.0, base_dmg * crit_factor)
    dps = avg_dmg * atk_speed
    hits_to_kill = np.ceil(monster_hp / avg_dmg)
    time_to_kill = hits_to_kill / max(0.001, atk_speed)
    return avg_dmg, dps, hits_to_kill, time_to_kill


def do_attack(base_atk: float, monster_def: float,
              crit_rate: float, crit_dmg: float):
    """실제 공격 1회 – calculate_combat_metrics와 완전히 동일한 공식, 치명타만 랜덤"""
    base_dmg = max(1.0, float(base_atk) - float(monster_def))
    is_crit = np.random.rand() < (crit_rate / 100.0)
    dmg = base_dmg * (crit_dmg / 100.0 if is_crit else 1.0)
    return max(1.0, dmg), is_crit


# ─────────────────────────────────────────────────────────────
# [수정1] 강화 판정: B안(꼬리형) 통일
# Python 시뮬 / Unity 생성 코드 / Tab3 테이블 전부 동일 규칙 사용
# ─────────────────────────────────────────────────────────────

def _enhance_rates(p_success_base: float, p_destroy_base: float,
                   stage: int, safeguard_enable: bool, safeguard_stage: int):
    """
    단계 조정 후 (sp, dp) 반환 – 안전 캡 포함
    B안(꼬리형) 기준으로 범위 보장:  0 ≤ sp ≤ 1,  0 ≤ dp ≤ 1 - sp
    """
    penalty = max(0.1, 1.0 - stage * 0.03)
    sp = min(1.0, p_success_base * penalty)
    dp = min(1.0, p_destroy_base * (1.0 + stage * 0.05))

    if safeguard_enable and stage >= safeguard_stage:
        dp = 0.0
        sp = min(1.0, sp * 0.5)   # 안전 대신 성공률 반감

    # 안전 캡: dp 범위를 (1 - sp) 이하로 보장
    dp = min(dp, max(0.0, 1.0 - sp))
    return sp, dp


def _roll_enhance(sp: float, dp: float, rng) -> str:
    """
    [수정1] B안(꼬리형) 판정 (Python/Unity 동일)
    roll < sp         → "success"
    roll >= 1 - dp    → "destroy"
    else              → "fail"
    """
    roll = rng.random()
    if roll < sp:
        return "success"
    elif roll >= (1.0 - dp):
        return "destroy"
    return "fail"


@st.cache_data(show_spinner=False)
def simulate_enhancement(prob: float, destroy: float,
                          max_stage: int, safeguard_enable: bool,
                          safeguard_stage: int,
                          n_sim: int = 5000,
                          random_seed: bool = True):
    """
    [수정1] 강화 시뮬 – B안(꼬리형) 판정 통일
    random_seed=True → 랜덤 시드 / False → seed=42(재현 모드)
    """
    p_success = prob / 100.0
    p_destroy = destroy / 100.0
    seed_val = None if random_seed else 42
    rng = np.random.default_rng(seed_val)

    all_tries    = np.zeros(n_sim, dtype=np.int32)
    all_destroys = np.zeros(n_sim, dtype=np.int32)

    for i in range(n_sim):
        stage, total_tries, total_destroys = 0, 0, 0
        while stage < max_stage:
            sp, dp = _enhance_rates(p_success, p_destroy,
                                     stage, safeguard_enable, safeguard_stage)
            result = _roll_enhance(sp, dp, rng)
            total_tries += 1
            if result == "success":
                stage += 1
            elif result == "destroy":
                total_destroys += 1
                stage = 0
        all_tries[i]    = total_tries
        all_destroys[i] = total_destroys

    return all_tries, all_destroys


# ─────────────────────────────────────────────────────────────
# [수정3] 가챠 몬테카를로 – pity 크면 표본 자동 감소 + 벡터화
# ─────────────────────────────────────────────────────────────

def _auto_n_sim(pity: int) -> int:
    """pity 값에 따라 표본 수 자동 조정"""
    return max(5_000, min(100_000, int(5_000_000 // max(1, pity))))


@st.cache_data(show_spinner=False)
def monte_carlo_gacha(prob_pct: float, pity: int,
                      soft_pity_enable: bool, soft_pity_start: int,
                      random_seed: bool = True):
    """
    [수정3] pity 기반 자동 표본 축소
    soft pity OFF → 벡터화(numpy geometric 샘플링)로 고속 처리
    """
    p = prob_pct / 100.0
    n_sim = _auto_n_sim(pity)
    seed_val = None if random_seed else 42
    rng = np.random.default_rng(seed_val)

    # ── soft pity OFF → 벡터화 (매우 빠름) ──
    if not soft_pity_enable:
        if p <= 0:
            return np.full(n_sim, pity, dtype=np.int32)
        # 기하분포로 획득 시도 횟수 샘플링 후 천장 적용
        raw = rng.geometric(p, size=n_sim)
        return np.minimum(raw, pity).astype(np.int32)

    # ── soft pity ON → 반복 시뮬 (n_sim 자동 축소) ──
    safe_soft_start = max(1, min(soft_pity_start, pity - 1))
    results = []
    for _ in range(n_sim):
        for t in range(1, pity + 1):
            curr_p = p
            if t >= safe_soft_start:
                progress = (t - safe_soft_start) / max(1, pity - safe_soft_start)
                curr_p = min(1.0, p + (1.0 - p) * progress)
            if t == pity or rng.random() < curr_p:
                results.append(t)
                break
    return np.array(results, dtype=np.int32)


def cumulative_gacha_curve(prob_pct: float, pity: int,
                            soft_pity_enable: bool, soft_pity_start: int):
    """누적 확률 곡선 (soft pity 포함)"""
    p = prob_pct / 100.0
    safe_soft_start = max(1, min(soft_pity_start, pity - 1))
    n_arr = np.arange(1, pity + 1)
    cum_p = np.zeros(len(n_arr))
    survive = 1.0
    for i, t in enumerate(n_arr):
        curr_p = p
        if soft_pity_enable and t >= safe_soft_start:
            progress = (t - safe_soft_start) / max(1, pity - safe_soft_start)
            curr_p = min(1.0, p + (1.0 - p) * progress)
        cum_p[i] = 1.0 - survive * (1.0 - curr_p)
        survive *= (1.0 - curr_p)
    cum_p[-1] = 1.0
    return n_arr, cum_p


# ═══════════════════════════════════════════════════════════════
# 5. AI 분석 (스키마 검증 + 예외 처리)
# ═══════════════════════════════════════════════════════════════
_AI_SCHEMA = {
    "base_atk":        (1,     99999,   10),
    "max_level":       (2,     9999,    50),
    "target_atk":      (1,     9999999, 500),
    "curve_type":      (["Exponential", "Logarithmic", "S-Curve"], "Exponential"),
    "prob_legend":     (0.001, 100.0,   0.1),
    "pity_count":      (1,     9999,    100),
    "enhance_prob":    (1.0,   100.0,   50.0),
    "enhance_destroy": (0.0,   100.0,   1.0),
    "monster_hp":      (1,     9999999, 100),
}

def _validate_ai_result(raw: dict) -> dict:
    out = {}
    for key, spec in _AI_SCHEMA.items():
        val = raw.get(key)
        if isinstance(spec[0], list):
            out[key] = val if val in spec[0] else spec[1]
        else:
            lo, hi, default = spec
            try:
                val = type(default)(val)
                out[key] = max(lo, min(hi, val))
            except (TypeError, ValueError):
                out[key] = default
    reason = str(raw.get('reason', 'AI 설계 적용 완료'))
    out['reason'] = reason[:200] + ("..." if len(reason) > 200 else "")
    return out


def analyze_intent(user_query: str) -> dict:
    prompt = f"""
당신은 게임 밸런스 디자이너입니다. 사용자 의도를 RPG 시스템 파라미터 JSON으로 변환하세요.
[사용자 요청]: "{user_query}"
[반환 JSON]:
{{
  "base_atk":int(1~99999), "max_level":int(2~9999), "target_atk":int(1~9999999),
  "curve_type":"Exponential"|"Logarithmic"|"S-Curve",
  "prob_legend":float(0.001~100), "pity_count":int(1~9999),
  "enhance_prob":float(1~100), "enhance_destroy":float(0~100),
  "monster_hp":int(1~9999999), "reason":"기획 의도 요약(100자 이내)"
}}
"""
    if client is None:
        st.error("❌ API 클라이언트 없음.")
        return {}
    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a master game designer. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
        )
        raw = json.loads(res.choices[0].message.content)
        return _validate_ai_result(raw)
    except json.JSONDecodeError:
        st.error("❌ AI가 잘못된 JSON 형식을 반환했습니다. 다시 시도해주세요.")
    except Exception as e:
        msg = str(e)
        if "timeout" in msg.lower():
            st.error("⏱️ AI 응답이 시간 초과됐습니다.")
        else:
            st.error(f"❌ AI 호출 실패: {msg[:150]}")
    return {}


# ═══════════════════════════════════════════════════════════════
# 6. 사이드바 – 프리셋 & 전역 옵션
# ═══════════════════════════════════════════════════════════════
st.sidebar.header("🕹️ 메이저 게임 프리셋")
selected_preset = st.sidebar.selectbox("밸런스 전략 선택", list(PRESETS.keys()),
                                        key="preset_select")

if selected_preset != st.session_state["current_preset"]:
    data = PRESETS[selected_preset]
    if data:
        for k, v in data.items():
            if k in DEFAULTS:
                st.session_state[k] = v
        st.session_state["current_monster_hp"] = float(st.session_state["monster_hp"])
    st.session_state["current_preset"] = selected_preset
    st.rerun()

if PRESETS[selected_preset]:
    st.sidebar.info(f"💡 {PRESETS[selected_preset]['desc']}")

st.sidebar.divider()

# ── 툴팁 CSS (전역 1회 삽입) ─────────────────────────────────
st.markdown("""
<style>
/* 물음표 툴팁 공통 스타일 */
.tooltip-wrap {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 4px;
}
.tooltip-wrap .section-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: inherit;
}
.tooltip-icon {
    position: relative;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 18px;
    height: 18px;
    border-radius: 50%;
    background: #4A90D9;
    color: white;
    font-size: 11px;
    font-weight: bold;
    cursor: help;
    flex-shrink: 0;
    line-height: 1;
    user-select: none;
}
.tooltip-icon .tooltip-box {
    visibility: hidden;
    opacity: 0;
    width: 240px;
    background: #1e2533;
    color: #e8eaf0;
    font-size: 12px;
    line-height: 1.6;
    border-radius: 8px;
    padding: 10px 13px;
    position: absolute;
    left: 26px;
    top: 50%;
    transform: translateY(-50%);
    z-index: 9999;
    box-shadow: 0 4px 18px rgba(0,0,0,0.45);
    pointer-events: none;
    transition: opacity 0.18s ease;
    border: 1px solid #3a4460;
    white-space: normal;
    word-break: keep-all;
}
.tooltip-icon .tooltip-box::before {
    content: "";
    position: absolute;
    right: 100%;
    top: 50%;
    transform: translateY(-50%);
    border: 6px solid transparent;
    border-right-color: #3a4460;
}
.tooltip-icon:hover .tooltip-box {
    visibility: visible;
    opacity: 1;
}
</style>
""", unsafe_allow_html=True)

# ── 시뮬레이션 설정 헤더 + 물음표 툴팁 ──────────────────────
st.sidebar.markdown("""
<div class="tooltip-wrap">
  <span class="section-title">⚙️ 시뮬레이션 설정</span>
  <span class="tooltip-icon">?
    <span class="tooltip-box">
      <b>🔁 재현 모드란?</b><br>
      컴퓨터 난수는 <b>시작 숫자(시드)</b>에 따라 결과가 결정됩니다.<br><br>
      <b>ON (고정 seed=42)</b><br>
      → 몇 번 실행해도 가챠·강화 결과가 <b>항상 동일</b>.<br>
      파라미터 비교·밸런스 분석에 적합.<br><br>
      <b>OFF (랜덤)</b><br>
      → 실행마다 결과가 조금씩 달라져 <b>실제 게임 느낌</b>과 유사.
    </span>
  </span>
</div>
""", unsafe_allow_html=True)

# [수정6] 전역 시드 토글
st.session_state["sim_random_seed"] = not st.sidebar.checkbox(
    "🔁 재현 모드 (고정 시드 seed=42)",
    value=not st.session_state["sim_random_seed"]
)

st.sidebar.divider()
st.sidebar.header("🛠️ 작업 모드")
mode = st.sidebar.radio(
    "모드 선택",
    ["📈 성장 밸런스", "⚔️ 전투 시뮬레이터", "🎰 가챠 확률", "🔥 강화 리스크"]
)


# ═══════════════════════════════════════════════════════════════
# 7. 메인 – 성장 밸런스
# ═══════════════════════════════════════════════════════════════
if mode == "📈 성장 밸런스":
    st.title("📊 캐릭터 성장 & 밸런스 곡선")

    with st.sidebar.form("growth_form"):
        st.subheader("성장 설정")
        f_base       = st.number_input("레벨 1 공격력",  value=int(st.session_state["base_atk"]),   min_value=1)
        f_max_level  = st.slider("최대 레벨",           10, 9999, value=int(st.session_state["max_level"]))
        f_curve      = st.selectbox("곡선 타입",
                                    ["Exponential", "Logarithmic", "S-Curve"],
                                    index=["Exponential", "Logarithmic", "S-Curve"]
                                         .index(st.session_state["curve_type"]))
        f_target     = st.number_input("만렙 공격력",   value=int(st.session_state["target_atk"]),  min_value=1)
        f_ttk_mode   = st.checkbox("🎯 목표 TTK 기반 몬스터 HP 역산",
                                    value=st.session_state["ttk_mode"])
        # [수정4] 이산/연속 토글
        f_discrete   = st.checkbox("⚙️ 이산 TTK 모드 (타수 기반 ceil)",
                                    value=st.session_state["ttk_discrete"])
        f_target_ttk = st.slider("목표 처치 시간 (초)", 0.5, 60.0,
                                  value=float(st.session_state["target_ttk"]), step=0.5)
        if st.form_submit_button("✅ 적용"):
            st.session_state.update({
                "base_atk": int(f_base), "max_level": int(f_max_level),
                "curve_type": f_curve, "target_atk": int(f_target),
                "ttk_mode": f_ttk_mode, "ttk_discrete": f_discrete,
                "target_ttk": float(f_target_ttk),
            })

    levels, atk_vals, exp_vals = calculate_growth(
        st.session_state["base_atk"], st.session_state["max_level"],
        st.session_state["target_atk"], st.session_state["curve_type"]
    )

    if st.session_state["ttk_mode"]:
        mhp_vals = monster_hp_from_ttk(
            atk_vals, st.session_state["target_ttk"],
            st.session_state["atk_speed"], st.session_state["crit_rate"],
            st.session_state["crit_dmg"], st.session_state["monster_def"],
            discrete_mode=st.session_state["ttk_discrete"]
        )
        mode_label = "이산(타수 기반)" if st.session_state["ttk_discrete"] else "연속(DPS 기반)"
        st.info(f"🎯 목표 TTK **{st.session_state['target_ttk']}초** | 역산 모드: **{mode_label}**")

        # [수정4] 역산 검증 – 실제 지표로 확인
        _, _, hits_chk, ttk_chk = calculate_combat_metrics(
            float(atk_vals[len(atk_vals)//2]),  # 중간 레벨 샘플
            st.session_state["monster_def"],
            st.session_state["crit_rate"],
            st.session_state["crit_dmg"],
            st.session_state["atk_speed"],
            float(mhp_vals[len(mhp_vals)//2])
        )
        err_pct = abs(ttk_chk - st.session_state["target_ttk"]) / max(0.001, st.session_state["target_ttk"]) * 100
        if err_pct < 10:
            st.success(f"✅ 역산 검증 (중간 레벨 샘플): 실제 TTK ≈ {ttk_chk:.2f}초 (오차 {err_pct:.1f}%)")
        else:
            st.warning(f"⚠️ 역산 검증 (중간 레벨 샘플): 실제 TTK ≈ {ttk_chk:.2f}초 (오차 {err_pct:.1f}% – 이산 모드 고려)")
    else:
        mhp_vals = atk_vals * 4.0

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("성장 곡선 시각화")
        fig, ax1 = plt.subplots(figsize=(8, 4))
        ax1.plot(levels, atk_vals, color="#1f77b4", linewidth=2)
        ax1.plot(levels, mhp_vals, color="#ff7f0e", linestyle="--")
        ax1.set_xlabel("레벨"); ax1.set_ylabel("공격력 / 체력")
        ax2 = ax1.twinx()
        ax2.fill_between(levels, exp_vals, alpha=0.10, color="#2ca02c")
        ax2.set_ylabel("경험치 요구량")
        ax1.legend(handles=[
            Line2D([0], [0], color="#1f77b4", lw=2, label="캐릭터 공격력"),
            Line2D([0], [0], color="#ff7f0e", lw=2, linestyle="--", label="몬스터 체력"),
            Patch(facecolor="#2ca02c", alpha=0.3, label="필요 경험치"),
        ], loc="upper left")
        st.pyplot(fig); plt.close()

    with col2:
        st.subheader("🤖 AI 밸런스 진단")
        if st.button("성장 밸런스 분석"):
            with st.spinner("📊 분석 중..."):
                content = safe_ai_call([{"role": "user", "content":
                    f"RPG 캐릭터 공격력 {st.session_state['base_atk']}~{st.session_state['target_atk']}, "
                    f"레벨 1~{st.session_state['max_level']}, 곡선 {st.session_state['curve_type']}. "
                    f"밸런스 문제·개선점을 한국어로 분석해줘."}])
                if content: st.write(content)


# ═══════════════════════════════════════════════════════════════
# 8. 메인 – 전투 시뮬레이터
# ═══════════════════════════════════════════════════════════════
elif mode == "⚔️ 전투 시뮬레이터":
    st.title("🎮 전투 시뮬레이터 & 도파민 체크")

    with st.sidebar.form("combat_form"):
        st.subheader("전투 세부 설정")
        c_base   = st.number_input("현재 공격력",      value=int(st.session_state["base_atk"]),   min_value=1)
        c_def    = st.number_input("몬스터 방어력",    value=int(st.session_state["monster_def"]), min_value=0)
        c_speed  = st.slider("공격 속도 (회/초)", 0.1, 10.0, value=float(st.session_state["atk_speed"]))
        c_crit_r = st.slider("치명타 확률 (%)",  0.0, 100.0, value=float(st.session_state["crit_rate"]))
        c_crit_d = st.slider("치명타 피해 (%)", 100.0, 600.0, value=float(st.session_state["crit_dmg"]))
        c_hp     = st.number_input("몬스터 HP",        value=int(st.session_state["monster_hp"]),  min_value=1)
        combat_applied = st.form_submit_button("✅ 적용")

    if combat_applied:
        new_hp = max(1, int(c_hp))
        old_hp = max(1, st.session_state["monster_hp"])
        ratio  = st.session_state["current_monster_hp"] / old_hp
        st.session_state["current_monster_hp"] = min(float(new_hp), max(0.0, float(new_hp) * ratio))
        st.session_state.update({
            "base_atk": int(c_base), "monster_def": int(c_def),
            "atk_speed": float(c_speed), "crit_rate": float(c_crit_r),
            "crit_dmg": float(c_crit_d), "monster_hp": new_hp,
        })

    avg_dmg, dps, hits, kill_time = calculate_combat_metrics(
        st.session_state["base_atk"], st.session_state["monster_def"],
        st.session_state["crit_rate"], st.session_state["crit_dmg"],
        st.session_state["atk_speed"], st.session_state["monster_hp"]
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("평균 데미지 (기댓값)", f"{avg_dmg:.1f}")
    c2.metric("DPS (기댓값)",        f"{dps:.1f}")
    c3.metric("처치 타수 (기댓값)",  f"{int(hits)} 방")
    c4.metric("처치 시간 (기댓값)",  f"{kill_time:.2f} 초")

    st.divider()
    st.subheader("🕹️ 실시간 타격 시뮬레이션")
    sim_col1, sim_col2 = st.columns(2)

    with sim_col1:
        st.write(f"캐릭터 (전략: {st.session_state['current_preset']})")
        st.image("https://api.dicebear.com/7.x/adventurer/svg?seed=Hero", width=150)

        atk_col, reset_col = st.columns(2)
        with atk_col:
            if st.button("⚔️ 공격 하기", use_container_width=True):
                dmg, is_crit = do_attack(
                    st.session_state["base_atk"], st.session_state["monster_def"],
                    st.session_state["crit_rate"], st.session_state["crit_dmg"]
                )
                st.session_state["current_monster_hp"] -= dmg
                crit_txt = "💥CRITICAL! " if is_crit else ""
                log_msg = (f"{crit_txt}{dmg:.1f} 데미지 "
                           f"(Atk:{st.session_state['base_atk']}"
                           f" - Def:{st.session_state['monster_def']})")
                # [수정6] battle_log 상한 200개
                st.session_state["battle_log"].insert(0, log_msg)
                st.session_state["battle_log"] = st.session_state["battle_log"][:200]
                if st.session_state["current_monster_hp"] <= 0:
                    st.session_state["current_monster_hp"] = float(st.session_state["monster_hp"])
                    st.session_state["battle_log"].insert(0, "🎊 처치! 새 몬스터 등장.")
                    st.balloons()

        # [수정6] 몬스터 리셋 버튼
        with reset_col:
            if st.button("🔄 리셋", use_container_width=True):
                st.session_state["current_monster_hp"] = float(st.session_state["monster_hp"])
                st.session_state["battle_log"] = []
                st.rerun()

    with sim_col2:
        st.write("몬스터")
        st.image("https://api.dicebear.com/7.x/bottts/svg?seed=Monster", width=150)
        safe_hp   = max(1, st.session_state["monster_hp"])
        hp_ratio  = max(0.0, min(1.0, st.session_state["current_monster_hp"] / safe_hp))
        st.progress(hp_ratio)
        display_hp = max(0.0, min(st.session_state["current_monster_hp"], float(safe_hp)))
        st.write(f"HP: {display_hp:.1f} / {safe_hp}")
        for log in st.session_state["battle_log"][:5]:
            st.caption(log)

    st.divider()
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        st.subheader("⚔️ 전투 체감 가이드")
        if   hits <= 2:  st.success("🎯 [원샷원킬] 압도적인 도파민!")
        elif hits <= 6:  st.info("⚡ [쾌속 사냥] 경쾌한 속도감.")
        elif hits <= 15: st.warning("🐢 [정체 구간] 지루함 유발.")
        else:            st.error("🛑 [절망 구간] 유저 이탈 리스크 매우 높음.")

    with col_v2:
        st.subheader("🤖 AI 전투 체감 진단")
        if st.button("AI 심층 분석"):
            with st.spinner("⚔️ 분석 중..."):
                content = safe_ai_call([{"role": "user", "content":
                    f"RPG 전투: 공격력 {st.session_state['base_atk']}, 방어력 {st.session_state['monster_def']}, "
                    f"DPS {dps:.1f}, 처치타수 {int(hits)}, 처치시간 {kill_time:.2f}초, "
                    f"치명타 {st.session_state['crit_rate']}%/{st.session_state['crit_dmg']}%. "
                    f"도파민 설계 관점에서 한국어로 진단해줘."}])
                if content: st.write(content)


# ═══════════════════════════════════════════════════════════════
# 9. 메인 – 가챠 확률
# ═══════════════════════════════════════════════════════════════
elif mode == "🎰 가챠 확률":
    st.title("🎰 가챠 드랍 & 도파민 설계기 Pro")

    with st.sidebar.form("gacha_form"):
        st.subheader("가챠 설정")
        g_prob       = st.number_input("레전드 확률 (%)", value=float(st.session_state["prob_legend"]),
                                        format="%.4f", min_value=0.001, max_value=100.0)
        g_pity       = st.number_input("천장 횟수 (Pity)", value=int(st.session_state["pity_count"]),
                                        min_value=1, max_value=9999)
        g_soft_en    = st.checkbox("Soft Pity 활성화 (원신식 점진 상승)",
                                    value=st.session_state["soft_pity_enable"])
        g_soft_start = st.slider("Soft Pity 시작 지점",
                                  1, max(1, int(g_pity) - 1),
                                  value=min(int(st.session_state["soft_pity_start"]),
                                            max(1, int(g_pity) - 1)))
        if st.form_submit_button("✅ 적용"):
            st.session_state.update({
                "prob_legend": float(g_prob), "pity_count": int(g_pity),
                "soft_pity_enable": g_soft_en, "soft_pity_start": int(g_soft_start),
            })

    p    = st.session_state["prob_legend"] / 100.0
    pity = st.session_state["pity_count"]

    # [수정3] 자동 표본 수 안내
    n_sim_actual = _auto_n_sim(pity)
    if pity > 1000:
        st.info(f"ℹ️ pity={pity} (크기가 큼) → 표본 수 자동 감소: **{n_sim_actual:,}명** 시뮬레이션")

    with st.spinner(f"🎲 {n_sim_actual:,}명 시뮬레이션 중... (캐시 후 즉시)"):
        sim_data = monte_carlo_gacha(
            st.session_state["prob_legend"], pity,
            st.session_state["soft_pity_enable"],
            st.session_state["soft_pity_start"],
            st.session_state["sim_random_seed"]
        )

    # [수정2] 시뮬 기반 실제 지표
    sim_mean = float(np.mean(sim_data))
    p50      = int(np.percentile(sim_data, 50))
    p95      = int(np.percentile(sim_data, 95))
    p99      = int(np.percentile(sim_data, 99))

    # [수정2] 이론값 (고정확률 기준)
    theory_mean   = 1 / p if p > 0 else 0
    theory_median = int(np.log(0.5) / np.log(1 - p)) if p > 0 else 0
    theory_p95    = int(np.log(0.05) / np.log(1 - p)) if p > 0 else 0

    st.subheader("📊 핵심 지표 비교")
    theory_col, sim_col = st.columns(2)
    with theory_col:
        st.markdown("**📐 이론값 (고정확률 가챠 기준)**")
        tc1, tc2, tc3 = st.columns(3)
        tc1.metric("이론 평균",   f"{theory_mean:.1f}회")
        tc2.metric("이론 50%",   f"{theory_median}회")
        tc3.metric("이론 95%",   f"{theory_p95}회")
        if st.session_state["soft_pity_enable"] or True:
            st.caption("※ 이론값은 soft pity·천장 미반영 고정확률 기준입니다.")

    with sim_col:
        sp_label = f" + Soft Pity({st.session_state['soft_pity_start']}~)" if st.session_state["soft_pity_enable"] else ""
        st.markdown(f"**🎲 실제 시뮬값 (천장{sp_label} 반영)**")
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("실제 평균",   f"{sim_mean:.1f}회")
        sc2.metric("실제 50%",   f"{p50}회")
        sc3.metric("실제 95%",   f"{p95}회")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎲 몬테카를로 분포")
        fig, (ax_hist, ax_cdf) = plt.subplots(2, 1, figsize=(8, 6))

        ax_hist.hist(sim_data, bins=min(60, max(10, pity // 5)),
                     color="#e74c3c", alpha=0.7, density=True)
        for val, color, lbl in [(p50,"blue",f"50%({p50}회)"),
                                  (p95,"orange",f"95%({p95}회)"),
                                  (p99,"red",f"최악1%({p99}회)")]:
            ax_hist.axvline(val, color=color, linestyle="--", lw=1.5, label=lbl)
        ax_hist.axvline(pity, color="purple", linestyle="-", lw=2, label=f"천장({pity}회)")
        ax_hist.set_xlabel("획득까지 시도 횟수"); ax_hist.set_ylabel("밀도")
        ax_hist.set_title("획득 시도 횟수 분포"); ax_hist.legend(fontsize=8)

        n_arr, cum_p = cumulative_gacha_curve(
            st.session_state["prob_legend"], pity,
            st.session_state["soft_pity_enable"],
            st.session_state["soft_pity_start"]
        )
        ax_cdf.plot(n_arr, cum_p * 100, color="#e74c3c", lw=2)
        for pct, color, lbl in [(50,"blue","50%"),(95,"orange","95%"),(99,"red","최악 1%")]:
            ax_cdf.axhline(pct, color=color, linestyle="--", alpha=0.7, lw=1.2, label=lbl)
        ax_cdf.axvline(pity, color="purple", linestyle="-", lw=1.5, label=f"천장({pity})")
        if st.session_state["soft_pity_enable"]:
            ax_cdf.axvline(st.session_state["soft_pity_start"], color="green",
                           linestyle=":", lw=1.5,
                           label=f"Soft Pity({st.session_state['soft_pity_start']})")
        ax_cdf.set_xlabel("시도 횟수"); ax_cdf.set_ylabel("누적 확률 (%)")
        ax_cdf.set_title("누적 획득 확률 곡선"); ax_cdf.legend(fontsize=8)

        plt.tight_layout()
        st.pyplot(fig); plt.close()
        st.info(f"📊 시뮬({n_sim_actual:,}명) | 평균:{sim_mean:.1f} 50%:{p50} 95%:{p95} **최악1%:{p99}회** 천장:{pity}회")

    with col2:
        st.subheader("🤖 AI 심리 리스크 분석")
        if st.button("도파민 커브 정밀 진단"):
            with st.spinner("🎰 분석 중..."):
                sp_info = (f"Soft Pity ON({st.session_state['soft_pity_start']}회~)"
                           if st.session_state["soft_pity_enable"] else "Soft Pity OFF")
                content = safe_ai_call([{"role": "user", "content":
                    f"가챠: 확률 {st.session_state['prob_legend']}%, 천장 {pity}회, {sp_info}.\n"
                    f"시뮬({n_sim_actual:,}명): 평균={sim_mean:.1f}, 50%={p50}, 95%={p95}, 최악1%={p99}회.\n"
                    f"1.박탈감 구간 2.기대 좌절 지점 3.이탈 위험도를 한국어로 분석해줘."}])
                if content: st.info(content)

        st.subheader("⚠️ 기대 좌절 구간 모니터링")
        frustration_start = int(sim_mean * 0.7)
        st.warning(f"💡 **{frustration_start}회 ~ {int(sim_mean)}회** 구간: 평균에 근접하지만 미획득 유저 기대 좌절 극대화")
        st.error(f"😱 **최악 1% 구간: {p99}회 이상** → 천장까지 {'미도달' if p99 < pity else '도달(천장 보호 필수)'}")


# ═══════════════════════════════════════════════════════════════
# 10. 메인 – 강화 리스크
# ═══════════════════════════════════════════════════════════════
elif mode == "🔥 강화 리스크":
    st.title("🔥 강화 시스템 리스크 설계")

    with st.sidebar.form("enhance_form"):
        st.subheader("강화 설정")
        e_prob      = st.slider("기본 성공 확률 (%)",  1.0, 100.0, value=float(st.session_state["enhance_prob"]))
        e_destroy   = st.slider("기본 파괴 확률 (%)",  0.0,  50.0, value=float(st.session_state["enhance_destroy"]))
        e_max_stage = st.slider("목표 최대 강화 단계", 3, 30,      value=int(st.session_state["enhance_max_stage"]))
        e_safeguard = st.checkbox("세이프가드 (특정 단계 이상 파괴 방지)",
                                   value=st.session_state["safeguard_enable"])
        e_sg_stage  = st.slider("세이프가드 시작 단계", 1, max(1, int(e_max_stage) - 1),
                                 value=min(int(st.session_state["safeguard_stage"]),
                                           max(1, int(e_max_stage) - 1)))
        if st.form_submit_button("✅ 시뮬레이션 실행"):
            st.session_state.update({
                "enhance_prob": float(e_prob), "enhance_destroy": float(e_destroy),
                "enhance_max_stage": int(e_max_stage),
                "safeguard_enable": e_safeguard, "safeguard_stage": int(e_sg_stage),
            })

    c1, c2, c3 = st.columns(3)
    c1.metric("단순 1회 기댓값",  f"{1/(st.session_state['enhance_prob']/100):.2f}회")
    c2.metric("목표 강화 단계",   f"+{st.session_state['enhance_max_stage']}")
    c3.metric("세이프가드",        "✅ ON" if st.session_state["safeguard_enable"] else "❌ OFF")

    # [수정1] B안 판정 안내
    st.caption("🔧 판정 규칙: B안(꼬리형) — `roll < 성공률 → 성공` / `roll ≥ 1-파괴율 → 파괴` / `else → 실패` (Python 시뮬 ↔ Unity 코드 동일)")
    st.divider()

    with st.spinner(f"🔥 +{st.session_state['enhance_max_stage']} 달성 시뮬레이션 (5,000회)..."):
        tries_arr, destroy_arr = simulate_enhancement(
            st.session_state["enhance_prob"], st.session_state["enhance_destroy"],
            st.session_state["enhance_max_stage"],
            st.session_state["safeguard_enable"], st.session_state["safeguard_stage"],
            n_sim=5000,
            random_seed=st.session_state["sim_random_seed"]
        )

    t_mean = int(np.mean(tries_arr))
    t50    = int(np.percentile(tries_arr, 50))
    t90    = int(np.percentile(tries_arr, 90))
    t95    = int(np.percentile(tries_arr, 95))
    d_mean = float(np.mean(destroy_arr))
    d_pct  = float(np.mean(destroy_arr > 0) * 100)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("평균 시도",      f"{t_mean}회")
    m2.metric("최악 10% 시도",  f"{t90}회")
    m3.metric("평균 파괴 횟수", f"{d_mean:.1f}회")
    m4.metric("파괴 경험 비율", f"{d_pct:.1f}%")

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.subheader("📊 강화 시도 횟수 분포")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(tries_arr, bins=50, color="#e74c3c", alpha=0.7, density=True)
        for val, color, lbl in [(t50,"blue",f"중앙값({t50}회)"),
                                  (t90,"orange",f"최악10%({t90}회)"),
                                  (t95,"red",   f"최악5%({t95}회)")]:
            ax.axvline(val, color=color, linestyle="--", lw=1.5, label=lbl)
        ax.set_xlabel("총 시도 횟수"); ax.set_ylabel("밀도")
        ax.set_title(f"+{st.session_state['enhance_max_stage']} 달성 시도 횟수 분포")
        ax.legend(fontsize=8)
        st.pyplot(fig); plt.close()

    with col_g2:
        st.subheader("🤖 AI 강화 경제 분석")
        if st.button("강화 시스템 진단"):
            with st.spinner("🔥 분석 중..."):
                sg_info = (f"ON(+{st.session_state['safeguard_stage']}이상)"
                           if st.session_state["safeguard_enable"] else "OFF")
                content = safe_ai_call([{"role": "user", "content":
                    f"강화 시스템: 성공률 {st.session_state['enhance_prob']}%, "
                    f"파괴율 {st.session_state['enhance_destroy']}%, "
                    f"목표 +{st.session_state['enhance_max_stage']}, 세이프가드 {sg_info}.\n"
                    f"5,000회 시뮬: 평균 {t_mean}회, 최악10% {t90}회, "
                    f"파괴 경험 {d_pct:.0f}%, 평균 파괴 {d_mean:.1f}회.\n"
                    f"경제 밸런스·유저 경험을 한국어로 분석해줘."}])
                if content: st.write(content)


# ═══════════════════════════════════════════════════════════════
# 11. 하단 공통 탭
# ═══════════════════════════════════════════════════════════════
st.divider()
tab1, tab2, tab3 = st.tabs(["💡 AI 자연어 설정", "🎮 Unity C# 코드", "📄 데이터 확인"])

# ── Tab 1: AI 자연어 ─────────────────────────────────────────
with tab1:
    st.subheader("💡 AI 자동 밸런스 설계")
    user_input = st.text_input("의도 입력",
        placeholder="예: 초반엔 잘 나오다가 후반에 희귀템이 터지는 느낌")
    if st.button("AI 자동 설계 적용"):
        if user_input.strip():
            with st.spinner("🤖 의도를 파라미터로 변환 중..."):
                result = analyze_intent(user_input)
                if result:
                    st.session_state.update({k: v for k, v in result.items() if k in DEFAULTS})
                    if "monster_hp" in result:
                        st.session_state["current_monster_hp"] = float(result["monster_hp"])
                    st.success(f"✅ 반영 완료: {result.get('reason', '설계 적용')}")
                    st.rerun()
        else:
            st.warning("의도를 입력해주세요.")

# ── Tab 2: Unity C# ──────────────────────────────────────────
with tab2:
    st.subheader("🎮 Unity C# 통합 매니저")
    st.caption("✅ Python 시뮬 ↔ Unity 코드 동일 판정 규칙 (B안 꼬리형 강화 / Soft Pity 가챠 / 이산 TTK)")

    _b  = st.session_state["base_atk"]
    _t  = st.session_state["target_atk"]
    _ml = st.session_state["max_level"]
    _c  = st.session_state["curve_type"]

    if _c == "Exponential":
        r = (_t / _b) ** (1.0 / (_ml - 1)) if _ml > 1 else 1.0
        growth_cs = f"return baseAtk * Mathf.Pow({r:.6f}f, level - 1);"
    elif _c == "Logarithmic":
        growth_cs = "return baseAtk + (targetAtk - baseAtk) * (Mathf.Log(level) / Mathf.Log(maxLevel));"
    else:
        growth_cs = "return baseAtk + (targetAtk - baseAtk) / (1f + Mathf.Exp(-0.2f * (level - maxLevel * 0.5f)));"

    # [수정1] Unity 강화 판정 B안(꼬리형) 코드 생성
    e_prob_f    = st.session_state["enhance_prob"]
    e_dest_f    = st.session_state["enhance_destroy"]
    sg_enable   = "true" if st.session_state["safeguard_enable"] else "false"
    sg_stage_v  = st.session_state["safeguard_stage"]
    sp_enable   = "true" if st.session_state["soft_pity_enable"] else "false"
    sp_start_v  = st.session_state["soft_pity_start"]
    pity_v      = st.session_state["pity_count"]
    prob_f      = st.session_state["prob_legend"]

    st.code(f"""
using UnityEngine;

/// RPG Balance Manager — Auto-generated by 설계기 Pro v3
/// Strategy: {selected_preset}
public class GameBalanceManager : MonoBehaviour
{{
    // ── 1. 성장 시스템 ──────────────────────────────────────
    public float GetAttackValue(int level, float baseAtk, float targetAtk, int maxLevel)
    {{
        if (level <= 1) return baseAtk;
        {growth_cs}
    }}

    // ── 2. 전투 시스템 (기댓값 / 실제 1타) ─────────────────
    public float GetExpectedDPS(float baseAtk, float monsterDef,
                                 float critRate, float critDmgPct, float atkSpeed)
    {{
        float baseDmg    = Mathf.Max(1f, baseAtk - monsterDef);
        float critFactor = 1f + (critRate / 100f) * (critDmgPct / 100f - 1f);
        return Mathf.Max(1f, baseDmg * critFactor) * atkSpeed;
    }}

    public float CalculateDamage(float baseAtk, float monsterDef,
                                  float critRate, float critDmgPct, out bool isCrit)
    {{
        float baseDmg = Mathf.Max(1f, baseAtk - monsterDef);
        isCrit        = Random.value < (critRate / 100f);
        return Mathf.Max(1f, isCrit ? baseDmg * (critDmgPct / 100f) : baseDmg);
    }}

    // ── 3. 가챠 (Soft Pity + 천장) ──────────────────────────
    public bool TryGacha(int currentPity)
    {{
        float baseProb      = {prob_f / 100:.8f}f;   // {prob_f}%
        int   pityThreshold = {pity_v};
        bool  softPityOn    = {sp_enable};
        int   softPityStart = {sp_start_v};

        if (currentPity >= pityThreshold) return true;

        float prob = baseProb;
        if (softPityOn && currentPity >= softPityStart)
        {{
            float progress = (float)(currentPity - softPityStart)
                             / Mathf.Max(1, pityThreshold - softPityStart);
            prob = Mathf.Min(1f, baseProb + (1f - baseProb) * progress);
        }}
        return Random.value < prob;
    }}

    // ── 4. 강화 시스템 — B안(꼬리형) ★ Python 시뮬과 동일 ───
    //   roll < sp          → Success
    //   roll >= (1 - dp)   → Destroyed
    //   else               → Fail
    //   ※ sp + dp ≤ 1 안전 캡 적용됨
    public EnhanceResult UpgradeItem(int currentStage)
    {{
        float pSuccess       = {e_prob_f / 100:.6f}f;   // {e_prob_f}%
        float pDestroy       = {e_dest_f / 100:.6f}f;
        bool  safeguardOn    = {sg_enable};
        int   safeguardStage = {sg_stage_v};

        // 단계별 패널티 적용
        float penalty     = Mathf.Max(0.1f, 1f - currentStage * 0.03f);
        float sp          = Mathf.Min(1f, pSuccess * penalty);
        float dp          = Mathf.Min(1f, pDestroy * (1f + currentStage * 0.05f));

        // 세이프가드
        if (safeguardOn && currentStage >= safeguardStage)
        {{
            dp  = 0f;
            sp *= 0.5f;
        }}

        // ★ 안전 캡: dp ≤ (1 - sp)
        dp = Mathf.Min(dp, Mathf.Max(0f, 1f - sp));

        // ★ B안(꼬리형) 판정 — Python _roll_enhance()와 동일
        float roll = Random.value;
        if (roll < sp)            return EnhanceResult.Success;
        if (roll >= (1f - dp))    return EnhanceResult.Destroyed;
        return EnhanceResult.Fail;
    }}

    public enum EnhanceResult {{ Success, Fail, Destroyed }}
}}
""", language="csharp")

# ── Tab 3: 데이터 확인 ───────────────────────────────────────
with tab3:
    if mode == "📈 성장 밸런스":
        st.subheader("📈 레벨별 공격력 · 몬스터 HP")
        lvls, atk_v, _ = calculate_growth(
            st.session_state["base_atk"], st.session_state["max_level"],
            st.session_state["target_atk"], st.session_state["curve_type"]
        )
        mhp_v = (monster_hp_from_ttk(
            atk_v, st.session_state["target_ttk"], st.session_state["atk_speed"],
            st.session_state["crit_rate"], st.session_state["crit_dmg"],
            st.session_state["monster_def"],
            discrete_mode=st.session_state["ttk_discrete"]
        ) if st.session_state["ttk_mode"] else atk_v * 4)
        # [수정4] 역산 검증 열 추가
        _, _, hits_v, ttk_v = zip(*[
            calculate_combat_metrics(float(a), st.session_state["monster_def"],
                                     st.session_state["crit_rate"], st.session_state["crit_dmg"],
                                     st.session_state["atk_speed"], float(h))
            for a, h in zip(atk_v, mhp_v)
        ]) if st.session_state["ttk_mode"] else (None, None, None, None)

        df_data = {"Level": lvls.astype(int), "Atk": np.round(atk_v, 1), "Monster HP": np.round(mhp_v, 1)}
        if st.session_state["ttk_mode"] and ttk_v is not None:
            df_data["실제 TTK(초)"] = np.round(ttk_v, 2)
        st.dataframe(pd.DataFrame(df_data), use_container_width=True)

    elif mode == "🎰 가챠 확률":
        st.subheader("🎰 시도 횟수별 심리 상태 테이블")
        p_val = st.session_state["prob_legend"] / 100.0
        pity  = st.session_state["pity_count"]
        candidates = [1, 5, 10, 30, 50, 100, pity // 4, pity // 2,
                      int(1 / p_val) if p_val > 0 else pity, pity]
        tries_list = sorted({t for t in candidates if 0 < t <= pity})
        rows = []
        for t in tries_list:
            cp = 1 - (1 - p_val) ** t
            if   cp < 0.10: stage = "🧊 기대감 낮음"
            elif cp < 0.40: stage = "🌱 기대감 상승"
            elif cp < 0.60: stage = "🔥 도파민 피크"
            elif cp < 0.80: stage = "⚠️ 불안감 고조"
            else:           stage = "💀 천장 대기(해탈)"
            rows.append({"시도": f"{t}회", "누적 확률": f"{cp*100:.2f}%",
                         "미획득": f"{(1-cp)*100:.2f}%", "심리 상태": stage})
        st.table(pd.DataFrame(rows))
        st.info("💡 누적 확률 50% 전후에서 쾌감 최대. 80% 이상은 쾌감 → 안도감 전환.")

    elif mode == "🔥 강화 리스크":
        # [수정1] Tab3 단계 테이블 – B안(꼬리형) 동일 규칙
        st.subheader("🔥 단계별 성공·파괴율 테이블 (B안 꼬리형 규칙 적용)")
        rows = []
        p_s = st.session_state["enhance_prob"] / 100.0
        p_d = st.session_state["enhance_destroy"] / 100.0
        for s in range(st.session_state["enhance_max_stage"]):
            sp, dp = _enhance_rates(p_s, p_d, s,
                                     st.session_state["safeguard_enable"],
                                     st.session_state["safeguard_stage"])
            sg_active = st.session_state["safeguard_enable"] and s >= st.session_state["safeguard_stage"]
            rows.append({
                "단계":      f"+{s}→+{s+1}",
                "성공률":    f"{sp*100:.1f}%",
                "파괴율":    f"{dp*100:.2f}%",
                "실패율":    f"{(1-sp-dp)*100:.1f}%",
                "세이프가드": "✅" if sg_active else "-",
            })
        st.table(pd.DataFrame(rows))
        st.caption("※ 표의 확률은 Python 시뮬 / Unity 생성 코드와 완전히 동일한 B안(꼬리형) 규칙으로 계산됩니다.")

    else:
        st.write(f"현재 전략: **{st.session_state['current_preset']}**")
        st.json({k: v for k, v in st.session_state.items()
                 if k not in ("battle_log",) and isinstance(v, (int, float, str, bool))})