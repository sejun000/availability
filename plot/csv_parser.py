#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

class Parser:
    """
    두 가지 입력 포맷 자동 인식
    1) CSV : 헤더 + ',' 다수
    2) KV  : "key | value | ..." 형태
    숫자·불리언은 모두 자동 캐스팅한다.
    """
    def __init__(self, kv_delimiter: str = "|"):
        self.kv_delim = kv_delimiter

    # ─────────────────────── KV 전용 내부 함수 ───────────────────────
    def _parse_kv_line(self, line: str) -> dict:
        tokens = [t.strip() for t in line.split(self.kv_delim) if t.strip()]
        if len(tokens) % 2:
            raise ValueError("KV 라인의 토큰 수가 홀수입니다.")
        return {tokens[i]: self._auto_cast(tokens[i + 1])
                for i in range(0, len(tokens), 2)}

    @staticmethod
    def _auto_cast(val: str):
        """True/False → bool, 숫자 → int/float, 그 외 → str"""
        val = val.strip().replace(",", "")            # 공백·쉼표 제거
        if val.lower() in {"true", "false"}:
            return val.lower() == "true"
        try:
            return int(val) if "." not in val else float(val)
        except ValueError:
            return val

    # ───────────────────────── CSV 후처리 헬퍼 ───────────────────────
    def _post_cast_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        모든 object 열을 대상:
          - True/False → boolean
          - strip/replace(",") 후 숫자로 완전히 변환되면 Int64/float64 로 승격
        """
        for col in df.select_dtypes(include="object").columns:
            series = df[col].astype(str).str.strip()            # 공백 제거
            uniq = set(series.dropna().unique())

            # ① bool 열
            if uniq <= {"True", "False", "true", "false"}:
                df[col] = series.map({"True": True, "true": True,
                                      "False": False, "false": False})\
                                 .astype("boolean")
                continue

            # ② 숫자 열
            series_num = pd.to_numeric(series.str.replace(",", ""),
                                        errors="coerce")
            if series_num.notna().all():                        # 전부 숫자
                # 모두 정수값인가?
                if (series_num.dropna() % 1 == 0).all():
                    df[col] = series_num.astype("Int64")
                else:
                    df[col] = series_num.astype("float64")

        return df

    # ────────────────────────── public API ──────────────────────────
    def parse_file_to_dataframe(self, input_file: str) -> pd.DataFrame:
        path = Path(input_file)
        with path.open("r") as f:
            first_line = f.readline()

        is_csv_like = (first_line.count(",") > first_line.count(self.kv_delim)
                       and not first_line.strip().isdigit())

        if is_csv_like:
            # ── CSV ───────────────────────────────────────────────
            df = pd.read_csv(path, dtype=str)   # 우선 모두 문자열로 읽기
            df = self._post_cast_dataframe(df)
            
            for col in df.columns:
                if df[col].dtype.name in ("Int64", "UInt64"):
                    df[col] = df[col].astype(int)
            return df

        # ── KV ───────────────────────────────────────────────────
        records = []
        with path.open("r") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    records.append(self._parse_kv_line(line))
                except Exception as e:
                    print(f"[WARN] 라인 파싱 실패: {line.strip()}\n  ↳ {e}")
        return self._post_cast_dataframe(pd.DataFrame(records))
