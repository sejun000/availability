#!/usr/bin/env python3
import pandas as pd

class Parser:
    """
    Parser 클래스는 입력 파일을 읽어 각 라인이 "key | value | key | value | ..." 형식으로 되어 있다고 가정하고,
    이를 딕셔너리로 파싱한 후 pandas DataFrame으로 반환합니다.
    """
    def __init__(self, delimiter="|"):
        self.delimiter = delimiter

    def parse_line(self, line: str) -> dict:
        # 구분자로 split 후 공백 제거
        tokens = [token.strip() for token in line.split(self.delimiter) if token.strip() != ""]
        result = {}
        # key, value 쌍이어야 함.
        if len(tokens) % 2 != 0:
            raise ValueError("토큰의 개수가 짝수가 아닙니다. 입력 형식을 확인하세요.")
        for i in range(0, len(tokens), 2):
            key = tokens[i]
            value = tokens[i+1]
            result[key] = self._convert_value(value)
        return result

    def _convert_value(self, value: str):
        # 값이 숫자이면 float 또는 int로 변환, 그렇지 않으면 그대로 문자열 반환
        try:
            if '.' in value:
                num = float(value)
            else:
                num = int(value)
            return num
        except ValueError:
            return value

    def parse_file_to_dataframe(self, input_file: str) -> pd.DataFrame:
        with open(input_file, 'r') as f:
            lines = f.readlines()
        data = []
        for line in lines:
            try:
                parsed = self.parse_line(line)
                data.append(parsed)
            except Exception as e:
                print(f"Error parsing line: {line}\n{e}")
        return pd.DataFrame(data)