from __future__ import annotations

import math
import csv
from dataclasses import dataclass
from datetime import datetime, date, time, timedelta, timezone

# =========================
# 설정
# =========================
KST = timezone(timedelta(hours=9), name="KST")
UTC = timezone.utc

START_YEAR = 1920
END_YEAR = 2027

SCAN_STEP_MINUTES = 10
WINDOW_START_OFFSET_DAYS = -1   # 예상일 -1일 00:00 KST
WINDOW_END_OFFSET_DAYS = 2      # 예상일 +2일 23:50 KST (10분 스텝 맞춤)

OUT_CSV = "jeolgi_1920_2027_kst_10min_window_-1_+2.csv"

# =========================
# 24절기 정의: (이름, 목표 황경, 예상일(M,D))
# =========================
TERMS = [
    ("입춘", 315, (2, 4)),
    ("우수", 330, (2, 19)),
    ("경칩", 345, (3, 5)),
    ("춘분",   0, (3, 20)),
    ("청명",  15, (4, 5)),
    ("곡우",  30, (4, 20)),
    ("입하",  45, (5, 5)),
    ("소만",  60, (5, 21)),
    ("망종",  75, (6, 6)),
    ("하지",  90, (6, 21)),
    ("소서", 105, (7, 7)),
    ("대서", 120, (7, 22)),
    ("입추", 135, (8, 7)),
    ("처서", 150, (8, 23)),
    ("백로", 165, (9, 7)),
    ("추분", 180, (9, 23)),
    ("한로", 195, (10, 8)),
    ("상강", 210, (10, 23)),
    ("입동", 225, (11, 7)),
    ("소설", 240, (11, 22)),
    ("대설", 255, (12, 7)),
    ("동지", 270, (12, 22)),
    ("소한", 285, (1, 5)),
    ("대한", 300, (1, 20)),
]

# SessionCode: 소한=0 ... 동지=23
SESSION_ORDER = [
    "소한", "대한",
    "입춘", "우수", "경칩", "춘분", "청명", "곡우", "입하", "소만", "망종", "하지",
    "소서", "대서", "입추", "처서", "백로", "추분", "한로", "상강", "입동", "소설", "대설", "동지",
]
SESSION_CODE = {name: idx for idx, name in enumerate(SESSION_ORDER)}

# =========================
# 천문 계산 유틸
# =========================
def normalize_deg(x: float) -> float:
    x = x % 360.0
    return x + 360.0 if x < 0 else x

def deg_to_rad(d: float) -> float:
    return d * math.pi / 180.0

def julian_day_utc(dt_utc: datetime) -> float:
    """UTC datetime -> Julian Day (UT 근사). dt_utc는 timezone-aware여야 함."""
    if dt_utc.tzinfo is None:
        raise ValueError("dt_utc must be timezone-aware")
    dt_utc = dt_utc.astimezone(UTC)

    Y = dt_utc.year
    M = dt_utc.month
    D = (
        dt_utc.day
        + dt_utc.hour / 24.0
        + dt_utc.minute / 1440.0
        + (dt_utc.second + dt_utc.microsecond / 1_000_000.0) / 86400.0
    )

    if M <= 2:
        Y -= 1
        M += 12

    A = Y // 100
    B = 2 - A + (A // 4)

    JD = (
        math.floor(365.25 * (Y + 4716))
        + math.floor(30.6001 * (M + 1))
        + D + B - 1524.5
    )
    return JD

def solar_lambda_app_deg(dt_utc: datetime) -> float:
    """
    태양 겉보기 황경(λ_app, degrees).
    VSOP87 계열(Meeus/NOAA에서 널리 쓰는) 근사식.
    """
    jd = julian_day_utc(dt_utc)
    T = (jd - 2451545.0) / 36525.0

    # Mean longitude
    L0 = normalize_deg(280.46646 + 36000.76983 * T + 0.0003032 * T * T)

    # Mean anomaly
    M = normalize_deg(357.52911 + 35999.05029 * T - 0.0001537 * T * T)
    Mr = deg_to_rad(M)

    # Equation of center
    C = (
        (1.914602 - 0.004817 * T - 0.000014 * T * T) * math.sin(Mr)
        + (0.019993 - 0.000101 * T) * math.sin(2 * Mr)
        + 0.000289 * math.sin(3 * Mr)
    )

    true_long = normalize_deg(L0 + C)

    # Apparent correction (aberration + nutation simplified)
    omega = 125.04 - 1934.136 * T
    lam = true_long - 0.00569 - 0.00478 * math.sin(deg_to_rad(omega))
    return normalize_deg(lam)

# =========================
# 절기 탐색(스캔 + 선형보간)
# =========================
@dataclass(frozen=True)
class JeolgiResult:
    year: int
    session_code: int
    term: str
    target_deg: int
    scan_start_kst: datetime
    scan_end_kst: datetime
    scan_step_minutes: int
    dt_kst: datetime
    dt_utc: datetime
    lambda_at_result: float

def find_crossing_time_kst(
    year: int,
    term_name: str,
    target_deg: int,
    expected_month_day: tuple[int, int],
    step_minutes: int,
    start_offset_days: int,
    end_offset_days: int,
) -> JeolgiResult:
    exp_month, exp_day = expected_month_day
    expected = date(year, exp_month, exp_day)

    scan_start_kst = datetime.combine(expected + timedelta(days=start_offset_days), time(0, 0), tzinfo=KST)
    # 10분 스텝이므로 마지막은 23:50
    scan_end_kst = datetime.combine(expected + timedelta(days=end_offset_days), time(23, 50), tzinfo=KST)

    step = timedelta(minutes=step_minutes)

    # 스캔 배열
    times: list[datetime] = []
    lams: list[float] = []

    t = scan_start_kst
    while t <= scan_end_kst:
        times.append(t)
        lams.append(solar_lambda_app_deg(t.astimezone(UTC)))
        t += step

    # 황경 unwrap(360도 점프 제거) → 단조 증가에 가깝게 만들기
    unwrapped = [lams[0]]
    offset = 0.0
    for i in range(1, len(lams)):
        cur = lams[i]
        # 다음 값이 갑자기 크게 떨어지면(예: 359 -> 0), 360을 더해 연속화
        if cur + offset < unwrapped[-1] - 180.0:
            offset += 360.0
        unwrapped.append(cur + offset)

    # target도 같은 "분기"로 맞추기
    mid = unwrapped[len(unwrapped) // 2]
    k = round((mid - target_deg) / 360.0)
    target_u = target_deg + 360.0 * k

    # 교차 구간 찾기: unwrapped[i-1] < target <= unwrapped[i]
    idx = None
    for i in range(1, len(unwrapped)):
        if unwrapped[i - 1] < target_u <= unwrapped[i]:
            idx = i - 1
            break

    # 교차 못 찾으면(이상 케이스) 가장 가까운 격자점 선택
    if idx is None:
        j = min(range(len(unwrapped)), key=lambda i: abs(unwrapped[i] - target_u))
        dt_kst = times[j]
    else:
        t0, t1 = times[idx], times[idx + 1]
        u0, u1 = unwrapped[idx], unwrapped[idx + 1]
        frac = (target_u - u0) / (u1 - u0) if u1 != u0 else 0.0
        dt_kst = t0 + (t1 - t0) * frac

    dt_utc = dt_kst.astimezone(UTC)
    lam = solar_lambda_app_deg(dt_utc)

    return JeolgiResult(
        year=year,
        session_code=SESSION_CODE[term_name],
        term=term_name,
        target_deg=target_deg,
        scan_start_kst=scan_start_kst,
        scan_end_kst=scan_end_kst,
        scan_step_minutes=step_minutes,
        dt_kst=dt_kst,
        dt_utc=dt_utc,
        lambda_at_result=round(lam, 6),
    )

# =========================
# 메인: CSV 생성
# =========================
def main() -> None:
    rows = []
    for y in range(START_YEAR, END_YEAR + 1):
        for term_name, target, md in TERMS:
            r = find_crossing_time_kst(
                year=y,
                term_name=term_name,
                target_deg=target,
                expected_month_day=md,
                step_minutes=SCAN_STEP_MINUTES,
                start_offset_days=WINDOW_START_OFFSET_DAYS,
                end_offset_days=WINDOW_END_OFFSET_DAYS,
            )

            rows.append({
                "Year": r.year,
                "SessionCode": r.session_code,
                "Term": r.term,
                "TargetEclipticLongitudeDeg": r.target_deg,
                "ScanWindowStartKST": r.scan_start_kst.strftime("%Y-%m-%d %H:%M"),
                "ScanWindowEndKST": r.scan_end_kst.strftime("%Y-%m-%d %H:%M"),
                "ScanStepMinutes": r.scan_step_minutes,
                "DateKST": r.dt_kst.strftime("%Y-%m-%d"),
                "TimeKST": r.dt_kst.strftime("%H:%M"),
                "DateUTC": r.dt_utc.strftime("%Y-%m-%d"),
                "TimeUTC": r.dt_utc.strftime("%H:%M"),
                "LambdaAppDeg_at_result": r.lambda_at_result,
            })

    # 결과 정렬(원하시면 Year, SessionCode 기준으로 정렬)
    rows.sort(key=lambda x: (x["Year"], x["SessionCode"]))

    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"OK: wrote {OUT_CSV}  ({START_YEAR}~{END_YEAR}, step={SCAN_STEP_MINUTES}min, window={WINDOW_START_OFFSET_DAYS}..+{WINDOW_END_OFFSET_DAYS} days)")

if __name__ == "__main__":
    main()
