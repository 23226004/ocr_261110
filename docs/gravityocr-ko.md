# GravityOCR-KO 모델 조사 및 검증

- 작성일: 2026-11-09
- 모델: [trillionlabs/GravityOCR-KO](https://huggingface.co/trillionlabs/GravityOCR-KO)
- 목적: 데스크톱 OCR 프로그램의 기본 OCR 모델로 쓸 수 있는지 검증

## 1. 모델 개요

Trillion Labs의 **약 1B 파라미터 한국어 특화 문서 OCR 모델**입니다. 이미지를 넣으면 텍스트를 출력하는 비전-언어 모델(VLM)입니다.

### 계보

```
zai-org/GLM-OCR (원본)
  └─ trillionlabs/GravityOCR (파인튜닝, GRPO 이후 버전)
       └─ trillionlabs/GravityOCR-KO (한국어 추가 학습)
```

- 구조: `GlmOcrForConditionalGeneration` (표준 GLM-OCR 체크포인트 + `block_diffusion.json`)
- 한국어 데이터로 추가 학습하면서 기존 영어·중일어 데이터를 섞어(replay) 학습했습니다. 그래서 표·수식 인식 능력이 유지된다고 적혀 있습니다.

### 사양

| 항목 | 내용 |
|---|---|
| 크기 | 약 1B 파라미터, BF16, `model.safetensors` 2.21GB |
| 라이선스 | MIT |
| 필요 라이브러리 | `transformers>=5.8`, `torch` |
| 학습 한도 | max_length 16,384 / 출력 최대 6,144 토큰 / 이미지 토큰 최대 8,192 |
| 프롬프트 형식 | GLM 채팅 템플릿 (`[gMASK]<sop>`, `<\|user\|>`, `<\|assistant\|>`, 이미지는 `<\|begin_of_image\|><\|image\|><\|end_of_image\|>`) |

### 특징: block diffusion 함께 학습

- 일반적인 한 토큰씩 생성하는 방식(AR)과 block diffusion(블록 크기 32)을 함께 학습했습니다. (`ar_loss_weight=1.0`, `bd_size=32`)
- 그래서 diffusion으로 초안을 여러 토큰 미리 쓰고 AR로 검증하는 **자기 추측 디코딩(self-speculative decoding)**이 가능합니다.
- 다만 이 기능은 nanoVLM-DiffusionOCR 저장소나 SGLang 패치가 있어야 씁니다. 일반 `transformers`로는 **AR 디코딩만** 돌아갑니다.

### 모델 카드에 없는 것

- 작업별 프롬프트. 현재는 원본 GLM-OCR의 `Text Recognition:` / `Table Recognition:` / `Formula Recognition:`을 씁니다.
- 출력 형식(마크다운/HTML/JSON)
- 벤치마크 수치
- 영수증·손글씨·PDF 지원 여부
- Mac(MPS)·MLX·GGUF 지원. 예제는 `device_map="cuda"`만 있습니다.

> ⚠️ 모델 카드 제목이 **"GravityOCR-KO (internal, private)"**로 되어 있습니다. 정식으로 쓰기 전에 공개 범위와 이용 조건을 확인해야 합니다.

## 2. 실행 환경 검증 (Apple M5, 24GB)

- `torch 2.15.0`, `transformers 5.22.2`, MPS 사용
- **Mac에서 정상 동작합니다.** 모델 로드는 1초 안쪽, 메모리는 여유가 있습니다.
- bf16 / fp16 / fp32의 출력이 똑같습니다. MPS 정밀도 문제는 없습니다.

## 3. 실제 영수증 벤치마크

### 방법

- 데이터: 운영 서비스(laonOCR)의 정답 포함 샘플 20장 (`samples/groundtruth.yaml`)
- 측정: 정답 값(금액·날짜·시각·상호명)이 **OCR 원문 텍스트 안에 들어 있는지** 확인 (`bench.py`)
- 주의: 운영 서비스 수치는 필드를 직접 뽑아낸 결과이고, GravityOCR 수치는 텍스트 포함 여부만 본 것입니다. GravityOCR 쪽은 **필드 추출 정확도의 상한선**으로 봐야 합니다.

### 결과

| 항목 | 운영 서비스 v1~v4 (원본 DPI) | GravityOCR-KO |
|---|---|---|
| 금액 | 18~19 / 20 | **20 / 20** |
| 날짜 | 19~20 / 20 | **20 / 20** |
| 시각(분) | 18~20 / 20 | **20 / 20** |
| 상호명 정확 일치 | 6~9 / 20 | 8 / 20 |
| 상호명 부분 일치 | 7~10 / 20 | 13 / 20 |
| 처리 속도 | 2.9~4.2초 (GPU 서버) | **평균 20초** (M5 Mac) / **평균 3.3초** (RTX 4090) |

#### 운영 서버 GPU 실행 (2026-11-10)

- 서버: RTX 4090 24GB. 운영 vLLM이 13.6GB를 쓰는 상태에서 별도 폴더 `~/ocr_261110_test`의 독립 환경으로 실행했습니다. 운영 폴더는 건드리지 않았습니다.
- `torch 2.15.0+cu132`, `transformers 5.22.2`, bf16
- 금액·날짜·시각 20/20, 상호명 정확 일치 8/20, 부분 일치 12/20 (Mac과 거의 같고, 부분 일치만 1장 차이)
- 평균 3.3초. 첫 장은 GPU 준비 시간 때문에 9.3초이고, 나머지는 1.5~4.2초입니다. **운영 서비스와 비슷한 속도**입니다.
- 최대 GPU 메모리 3.35GB. 테스트가 끝난 뒤 GPU에는 운영 vLLM만 남은 것을 확인했습니다.

### 발견한 문제

1. **같은 줄 반복**
   - 쿠팡이츠 영수증 3장에서 `27,000` 같은 줄을 출력 한도까지 반복했습니다. 장당 약 126초가 걸렸습니다.
   - `repetition_penalty=1.1`로 해결했습니다(→ 약 21초). 전체를 다시 돌렸을 때 다른 영수증의 정확도는 떨어지지 않았습니다. `ocr.py`에 기본값으로 반영했습니다.
2. **비슷한 한글 혼동**
   - 더스푼→더스폰, 옹심이→용심이, 그리너→그리니, 에이글→에이클, 잔→전처럼 모양이 비슷한 글자를 틀립니다.
   - 상호명 점수가 낮은 주된 원인입니다. dtype을 바꿔도 같아서 모델 자체의 한계로 봅니다.
   - 작은 글씨는 이미지를 2~3배 확대하면 일부 나아집니다(합성 이미지 기준).
3. **속도**
   - 일반 `transformers`의 AR 디코딩만 써서 장당 약 20초가 걸립니다.
   - block diffusion 추측 디코딩이나 GPU 서버를 쓰면 줄어들 여지가 있습니다.

## 4. 결론 및 다음 단계

- 숫자·날짜 인식은 운영 서비스와 같거나 더 낫고, 상호명은 비슷한 수준입니다. **기본 OCR 모델로 쓸 만합니다.**
- 남은 과제:
  1. OCR 텍스트에서 금액·일시·상호명을 뽑는 **필드 추출** 단계. 운영 서비스의 `receipt_rules.py` 재사용을 검토합니다.
  2. **데스크톱 앱 UI**: 파일을 끌어다 놓으면 결과가 나오는 화면, PDF는 페이지별 이미지로 변환
  3. 속도 개선: 추측 디코딩 적용 검토
  4. 모델 이용 조건 확인 (internal 표기)

## 관련 파일

- `ocr.py`: 추론 함수 `ocr(image, task)`
- `bench.py`: 정답 비교 벤치마크 (`python bench.py samples/real`)
- `samples/real/`: 테스트 영수증 (실제 데이터라 git 제외)
