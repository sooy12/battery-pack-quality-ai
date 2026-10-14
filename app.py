
# ============================================================
# 배터리팩 방전 AI 품질검사 시스템
# PCA 1+8 + Direct Row
# ============================================================

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# 1. 화면 기본 설정
# ============================================================

st.set_page_config(
    page_title="배터리팩 AI 품질검사",
    page_icon="🔋",
    layout="wide"
)

st.title("🔋 배터리팩 방전 AI 품질검사")
st.caption("PCA 1+8 기반 Direct Row 이상 탐지 시스템")


# ============================================================
# 2. 저장된 AI 모델 불러오기
# ============================================================

MODEL_PATH = Path(
    r"C:\batt_ai\models\discharge_pca_direct_v1.joblib"
)


@st.cache_resource
def load_pca_model():
    return joblib.load(MODEL_PATH)


if not MODEL_PATH.exists():
    st.error(f"모델 파일이 없습니다: {MODEL_PATH}")
    st.stop()

try:
    model = load_pca_model()
except Exception as e:
    st.error(f"모델 불러오기 실패: {e}")
    st.stop()


feature_cols = list(model["feature_cols"])
voltage_cols = list(model["voltage_cols"])
temp_cols = list(model["temp_cols"])

voltage_idx = np.asarray(model["voltage_idx"], dtype=int)
temp_idx = np.asarray(model["temp_idx"], dtype=int)

threshold = float(model["threshold_95"])

# 입력 센서 순서와 PCA 구조 확인
if (
    len(feature_cols) != 208
    or len(voltage_cols) != 176
    or len(temp_cols) != 32
    or len(voltage_idx) != 176
    or len(temp_idx) != 32
):
    st.error("저장된 모델의 센서 구성이 예상과 다릅니다.")
    st.stop()


with st.sidebar:
    st.header("AI 모델 정보")
    st.write("모델: PCA 1+8")
    st.write("예측 방식: Direct Row")
    st.write("분석 데이터: 방전")
    st.write("입력 센서: 208개")
    st.write("전압 센서: 176개")
    st.write("온도 센서: 32개")

    st.metric(
        "이상 판정 임계값",
        f"{threshold:.6f}"
    )

    st.divider()
    st.caption(
        "기존에 학습된 PCA v1 모델을 사용합니다. "
        "새로운 CSV 업로드 시 재학습하지 않습니다."
    )

st.success("PCA 1+8 모델 연결 완료")


# ============================================================
# 3. CSV 업로드
# ============================================================

st.header("1. 배터리팩 방전 데이터 업로드")

uploaded_file = st.file_uploader(
    "검사할 방전 CSV 파일을 선택하세요.",
    type=["csv"]
)

if uploaded_file is None:
    st.info("CSV 파일을 업로드하면 AI 품질 검사를 시작합니다.")
    st.stop()


try:
    df = pd.read_csv(
        uploaded_file,
        encoding="utf-8-sig"
    )

except Exception as e:
    st.error(f"CSV 읽기 오류: {e}")
    st.stop()

if df.empty:
    st.error("CSV에 데이터가 없습니다.")
    st.stop()

st.success(f"업로드 완료: {uploaded_file.name}")

c1, c2 = st.columns(2)

c1.metric("전체 측정 행", f"{len(df):,}")
c2.metric("전체 컬럼", f"{len(df.columns):,}")

with st.expander("원본 데이터 미리보기"):
    st.dataframe(
        df.head(10),
        width="stretch"
    )


# ============================================================
# 4. 입력 데이터 검증
# ============================================================

st.header("2. 입력 데이터 검증")

missing_cols = [
    col for col in feature_cols
    if col not in df.columns
]

if missing_cols:
    st.error(
        f"필수 센서 컬럼 {len(missing_cols)}개가 누락됐습니다."
    )
    st.write(missing_cols)
    st.stop()

X_df = df[feature_cols].apply(
    pd.to_numeric,
    errors="coerce"
)

X = X_df.to_numpy(dtype=np.float64)

if not np.isfinite(X).all():
    invalid_cols = [
        feature_cols[i]
        for i in np.where(
            ~np.isfinite(X).all(axis=0)
        )[0]
    ]

    st.error("결측값 또는 숫자가 아닌 센서값이 있습니다.")
    st.write("문제 센서:", invalid_cols)
    st.stop()

st.success("208개 센서 및 입력 데이터 검증 완료")


# ============================================================
# 5. PCA 1+8 Direct Row 이상 예측
# ============================================================

try:
    # 기존 학습 Scaler 적용
    X_scaled = model["scaler"].transform(X)

    # 전압 PCA + 온도 PCA 재구성
    X_reconstructed = np.empty_like(X_scaled)

    X_reconstructed[:, voltage_idx] = (
        model["voltage_pca"].inverse_transform(
            model["voltage_pca"].transform(
                X_scaled[:, voltage_idx]
            )
        )
    )

    X_reconstructed[:, temp_idx] = (
        model["temperature_pca"].inverse_transform(
            model["temperature_pca"].transform(
                X_scaled[:, temp_idx]
            )
        )
    )

    # 센서별 제곱 재구성 오차
    squared_errors = (
        X_scaled - X_reconstructed
    ) ** 2

    # Direct Row: 행별 208개 센서 MSE
    row_mse = squared_errors.mean(axis=1)

    # Calibration P95 임계값
    predictions = (
        row_mse > threshold
    ).astype(int)

except Exception as e:
    st.error(f"AI 예측 오류: {e}")
    st.stop()


# ============================================================
# 6. AI 품질 검사 결과
# ============================================================

st.header("3. AI 품질 검사 결과")

total_count = len(predictions)
abnormal_count = int(predictions.sum())
normal_count = total_count - abnormal_count

abnormal_ratio = (
    abnormal_count / total_count * 100
)

if abnormal_count == 0:
    st.success("이상 신호 미감지")
else:
    st.warning("이상 신호 감지 — 추가 검사 필요")

c1, c2, c3, c4 = st.columns(4)

c1.metric("전체 행", f"{total_count:,}")
c2.metric("정상 판정 행", f"{normal_count:,}")
c3.metric("이상 의심 행", f"{abnormal_count:,}")
c4.metric("이상 의심 비율", f"{abnormal_ratio:.2f}%")

st.caption(
    "현재 판정은 측정 행 단위의 이상 탐지 결과입니다. "
    "이상 행이 발견되더라도 배터리팩 불량 확정은 아닙니다."
)


# ============================================================
# 7. 행별 이상 점수 그래프
# ============================================================

st.header("4. PCA 이상 점수 분석")

result_df = pd.DataFrame({
    "Row": np.arange(total_count),
    "PCA_MSE": row_mse,
    "Threshold": threshold,
    "Prediction": predictions
})

st.line_chart(
    result_df.set_index("Row")[
        ["PCA_MSE", "Threshold"]
    ],
    width="stretch"
)

st.caption(
    "PCA 재구성 오차가 임계값보다 큰 행을 "
    "이상 의심으로 분류합니다."
)

if abnormal_count > 0:
    first_abnormal = int(
        np.flatnonzero(predictions == 1)[0]
    )

    st.write(
        f"최초 이상 의심 행: {first_abnormal} "
        "(0부터 시작)"
    )


# ============================================================
# 8. 연속 이상 구간 분석
# ============================================================

st.header("5. 연속 이상 구간")


def find_abnormal_runs(pred):
    runs = []
    start = None

    for i, value in enumerate(pred):

        if value == 1 and start is None:
            start = i

        elif value == 0 and start is not None:
            runs.append((
                start,
                i - 1,
                i - start
            ))
            start = None

    if start is not None:
        runs.append((
            start,
            len(pred) - 1,
            len(pred) - start
        ))

    return runs


abnormal_runs = find_abnormal_runs(predictions)

if abnormal_runs:
    run_df = pd.DataFrame(
        abnormal_runs,
        columns=[
            "시작 행",
            "종료 행",
            "연속 이상 행 수"
        ]
    )

    longest_run = int(
        run_df["연속 이상 행 수"].max()
    )

    st.metric(
        "최장 연속 이상 구간",
        f"{longest_run:,}행"
    )

    st.dataframe(
        run_df.sort_values(
            "연속 이상 행 수",
            ascending=False
        ),
        width="stretch",
        hide_index=True
    )

else:
    st.info("연속 이상 구간이 없습니다.")


# ============================================================
# 9. 모듈별 및 셀별 이상 의심 분석
# ============================================================

st.header("6. 모듈별 및 셀별 이상 의심 분석")

if abnormal_count == 0:
    st.info(
        "이상으로 판정된 행이 없어 "
        "의심 센서 순위를 표시하지 않습니다."
    )

else:
    # 이상 판정된 행의 전압 센서 오차
    voltage_errors = squared_errors[
        predictions == 1
    ][:, voltage_idx]

    sensor_df = pd.DataFrame({
        "Sensor": voltage_cols,
        "Mean_PCA_Error": voltage_errors.mean(axis=0)
    })

    # 모듈 번호와 전압 채널 번호
    sensor_df["Module"] = (
        sensor_df["Sensor"].str.extract(
            r"^(M\d{2})CV\d{2}$"
        )[0]
    )

    sensor_df["Cell_Channel"] = (
        sensor_df["Sensor"].str.extract(
            r"^M\d{2}CV(\d{2})$"
        )[0]
    )

    sensor_df = sensor_df.sort_values(
        "Mean_PCA_Error",
        ascending=False
    )

    # -----------------------------------------
    # 9-1. 이상 의심 전압 채널 TOP 10
    # -----------------------------------------

    st.subheader("이상 의심 전압 채널 TOP 10")

    st.dataframe(
        sensor_df.head(10).reset_index(drop=True),
        width="stretch",
        hide_index=True
    )

    st.caption(
        "전압 채널별 PCA 재구성 오차 순위입니다. "
        "실제 불량 셀 위치를 확정한 결과는 아닙니다."
    )

    # -----------------------------------------
    # 9-2. 모듈별 이상 의심 순위
    # -----------------------------------------

    module_df = (
        sensor_df
        .groupby(
            "Module",
            as_index=False
        )["Mean_PCA_Error"]
        .mean()
        .sort_values(
            "Mean_PCA_Error",
            ascending=False
        )
    )

    st.subheader("16개 모듈별 이상 의심 순위")

    st.dataframe(
        module_df.reset_index(drop=True),
        width="stretch",
        hide_index=True
    )

    st.bar_chart(
        module_df.set_index("Module")[
            "Mean_PCA_Error"
        ],
        width="stretch"
    )

    # -----------------------------------------
    # 9-3. 온도 센서 이상 오차 분석
    # -----------------------------------------

    temp_errors = squared_errors[
        predictions == 1
    ][:, temp_idx]

    temp_df = pd.DataFrame({
        "Sensor": temp_cols,
        "Mean_PCA_Error": temp_errors.mean(axis=0)
    }).sort_values(
        "Mean_PCA_Error",
        ascending=False
    )

    st.subheader("온도 센서 오차 TOP 10")

    st.dataframe(
        temp_df.head(10).reset_index(drop=True),
        width="stretch",
        hide_index=True
    )

    st.caption(
        "모듈 및 센서 순위는 이상 구간의 PCA 오차를 "
        "기준으로 한 점검 우선순위입니다. "
        "센서별 정상 오차 수준이 다를 수 있습니다."
    )


# ============================================================
# 10. 행별 예측 결과 다운로드
# ============================================================

st.header("7. AI 검사 결과 다운로드")

output_df = result_df.copy()

output_df["Result"] = np.where(
    output_df["Prediction"] == 1,
    "Abnormal_Suspected",
    "Normal_Pattern"
)

st.dataframe(
    output_df.head(30),
    width="stretch",
    hide_index=True
)

st.download_button(
    label="AI 예측 결과 CSV 다운로드",
    data=output_df.to_csv(
        index=False
    ).encode("utf-8-sig"),
    file_name="discharge_prediction.csv",
    mime="text/csv"
)


# ============================================================
# 11. 모델 사용 안내
# ============================================================

st.divider()

st.caption(
    "PCA 1+8 Direct Row v1 | "
    "Calibration P95 임계값 | "
    "학습된 모델 그대로 예측"
)

st.caption(
    "이상 신호 미감지는 양품 보증이 아니며, "
    "이상 신호 감지는 불량품 확정이 아닙니다."
)
