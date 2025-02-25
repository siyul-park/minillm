# GPT-1 Style Language Model in PyTorch

이 프로젝트는 GPT-1 스타일의 언어 모델을 PyTorch로 구현하고 WikiText-2 데이터셋을 이용해 학습하는 예제입니다.

## 구성

- **data/**
  - `raw/`: 다운로드한 원본 WikiText-2 데이터 (zip 및 추출 결과)
  - `processed/`: 전처리된 텍스트 데이터
- **models/**
  - `gpt1_model.py`: GPT-1 스타일 모델 정의
- **scripts/**
  - `download_data.py`: WikiText-2 데이터 다운로드 및 압축 해제 스크립트
  - `train.py`: 모델 학습 스크립트
- **tests/**: (선택사항) 모델 테스트 코드

## 설치

1. Python 가상환경(예: Miniforge/Miniconda)을 만듭니다.
   ```bash
   conda create -n gpt1-env python=3.9 -y
   conda activate gpt1-env
