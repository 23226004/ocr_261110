# GravityOCR-KO 라이선스 분석

- 작성일: 2026-11-10
- 대상: `trillionlabs/GravityOCR-KO`와 그 계보, 그리고 이 프로젝트가 쓰는 라이브러리
- 주의: 공개 페이지를 근거로 한 기술적 검토이며 법률 자문이 아닙니다. 상용 배포 전에는 법무 검토를 권장합니다.

## 1. 결론 요약

| 판단 | 내용 |
|---|---|
| 표기상 라이선스 | **MIT**: 계보 전체(GLM-OCR → GravityOCR → GravityOCR-KO)가 MIT로 표기되어 있습니다. 상업적 사용·수정·재배포가 가능하고 고지 의무만 있습니다. |
| 핵심 리스크 | GravityOCR-KO에는 **LICENSE 파일이 없고**, 모델 카드 제목이 **"internal, private"**입니다. MIT 태그만 있고 의도가 불분명해서, 공개 의도가 확인되지 않은 상태입니다. |
| 권장 | ① 개발·내부 테스트는 진행해도 됩니다. ② **외부 배포나 상용 서비스 전에 Trillion Labs에 라이선스를 서면으로 확인**해야 합니다. ③ 확인이 안 되면 정식 MIT 모델인 `trillionlabs/GravityOCR`를 대안으로 씁니다. |

## 2. 계보별 라이선스

| 모델 | HF 라이선스 태그 | LICENSE 파일 | 비고 |
|---|---|---|---|
| `zai-org/GLM-OCR` (원본) | `mit` | HF 저장소에는 없음 | 모델 카드: "The GLM-OCR model is released under the MIT License." |
| GitHub `zai-org/GLM-OCR` (코드) | Apache-2.0 | 있음 (Apache-2.0, Copyright 2026 Zhipu AI) | **코드는 Apache-2.0, 가중치는 MIT**로 서로 다릅니다. |
| `trillionlabs/GravityOCR` | `mit` | **있음** (MIT, Copyright (c) 2026 Trillion Labs) | "fine-tune of GLM-OCR and are distributed under the same MIT License" |
| `trillionlabs/GravityOCR-KO` (사용 모델) | `mit` | **없음** | 카드 제목 "(internal, private)", 라이선스 섹션 없음 |

### 구성 요소

- 비전 인코더 CogViT와 0.5B 텍스트 디코더는 GLM-OCR에서 그대로 가져왔습니다. 각 구성 요소의 별도 라이선스는 어디에도 적혀 있지 않고, GLM-OCR의 MIT를 따르는 것으로 봅니다.
- GLM-OCR **전체 파이프라인**에 들어 있는 레이아웃 분석 모델 `PP-DocLayoutV3`는 Apache-2.0입니다. **이 프로젝트는 쓰지 않습니다.** 모델만 직접 호출합니다.
- GravityOCR의 SGLang 패치(`patches/sglang/`)는 Apache-2.0입니다. 추측 디코딩용 서빙 패치이며 현재 쓰지 않습니다.
- `nanoVLM-DiffusionOCR` 저장소는 카드에 이름만 있고 링크가 없습니다("in process"). **라이선스를 알 수 없습니다.** 쓰게 되면 다시 확인해야 합니다.

### 학습 데이터

- 세 모델 모두 학습 데이터 출처와 라이선스를 밝히지 않았습니다.
- GravityOCR는 학습 섹션을 모델 카드에서 지웠습니다(커밋 bdbe9ab).
- MIT는 가중치에 대한 라이선스입니다. 학습 데이터의 저작권 문제까지 보증하지는 않습니다. 이 부분은 업계 공통의 잔여 리스크입니다.

## 3. GravityOCR-KO의 불확실성

1. **"internal, private" 표기**: 저장소는 공개되어 있고 접근 제한(gated)도 없습니다. 하지만 제목은 내부용임을 뜻합니다. 실수로 공개되었을 가능성이 있습니다.
2. **LICENSE 파일 없음**: MIT는 저작권 고지와 허가 문구를 함께 배포해야 하는데, 이 저장소에는 그 원문이 없습니다. 형제 모델 GravityOCR의 LICENSE(Copyright Trillion Labs)를 준용하는 것으로 보는 게 합리적이지만, 명시적이지는 않습니다.
3. **초기 상태**: 커밋은 "Initial commit" 하나뿐이고, 의존 저장소도 아직 준비 중입니다.
4. **조직 차원의 약관 없음**: Trillion Labs의 HF 조직 페이지에 별도 이용 약관이나 사용 정책이 없습니다.

→ HF 태그 `mit`만으로도 일반적으로 MIT 허가로 해석됩니다. 다만 권리자가 나중에 저장소를 비공개로 돌리거나 라이선스를 바꿀 가능성이 다른 모델보다 높습니다. **이미 받은 가중치에 대한 MIT 허가는 소급해서 철회되지 않는다고 보는 것이 일반적**이지만, 분쟁 소지를 없애려면 확인을 받아두는 게 좋습니다.

## 4. 사용 시나리오별 판단

| 시나리오 | 판단 | 조건 |
|---|---|---|
| 개발·벤치마크·내부 테스트 (현재) | ✅ 문제 없음 | 없음 |
| 사내 서버에서 OCR API 운영 (가중치 비배포) | ✅ 가능 | MIT는 서비스 제공 시 고지 의무가 사실상 없습니다. 출처 기록은 권장합니다. |
| 데스크톱 앱에 **가중치를 포함해 배포** | ⚠️ 확인 후 | 앱 안에 MIT 고지(저작권 + 허가 문구)를 넣어야 합니다. "internal" 표기 확인이 필요합니다. |
| 외부 고객 대상 상용 제품 | ⚠️ 확인 후 | Trillion Labs의 서면 확인을 권장합니다. 안 되면 GravityOCR로 교체합니다. |
| 파인튜닝한 파생 모델 배포 | ⚠️ 확인 후 | MIT 고지를 유지해야 합니다. 데이터 라이선스도 별도로 검토해야 합니다. |

**추천 구성**은 OCR을 서버에서 돌리고 데스크톱 앱이 API를 호출하는 방식입니다. 이렇게 하면 가중치를 배포하지 않으므로 라이선스 리스크가 가장 작습니다. 속도 면에서도 유리합니다(서버 3.3초 vs Mac 20초).

## 5. 배포 시 넣을 고지문 (예시)

```
This product uses GravityOCR-KO by Trillion Labs (https://huggingface.co/trillionlabs/GravityOCR-KO),
a fine-tune of GLM-OCR by Zhipu AI / Z.ai, licensed under the MIT License.

MIT License
Copyright (c) 2026 Trillion Labs
Permission is hereby granted, free of charge, to any person obtaining a copy ... (MIT 전문)
```

## 6. 이 프로젝트의 라이브러리

| 라이브러리 | 라이선스 | 비고 |
|---|---|---|
| PyTorch | BSD-3-Clause | 고지 필요 |
| transformers / accelerate | Apache-2.0 | 고지와 NOTICE 유지 |
| Pillow | MIT-CMU (HPND) | 고지 필요 |
| PyYAML | MIT | 벤치마크 전용 |

모두 상업적으로 쓸 수 있는 허용형 라이선스이고, GPL 계열은 없습니다.

## 7. 해야 할 일

- [ ] Trillion Labs에 GravityOCR-KO의 공개 의도와 라이선스(MIT 적용 여부)를 문의
- [ ] 답을 받기 전까지 배포 계획은 "서버 API" 방식으로 진행
- [x] 대안 `trillionlabs/GravityOCR`(정식 MIT)로 같은 벤치마크를 돌려 성능 차이 확인. 금액·날짜는 같고 상호명 정확 일치는 6/20 vs 8/20입니다. 자세한 내용은 `gravityocr-ko.md`에 있습니다.
- [ ] 앱 배포 시 오픈소스 고지 화면(About / Licenses)에 위 고지문 포함

## 출처

- https://huggingface.co/trillionlabs/GravityOCR-KO (raw README, tree)
- https://huggingface.co/trillionlabs/GravityOCR (LICENSE, README)
- https://github.com/trillion-labs/GravityOCR
- https://huggingface.co/zai-org/GLM-OCR (README "License")
- https://github.com/zai-org/GLM-OCR (LICENSE: Apache-2.0)
- https://huggingface.co/trillionlabs (조직 페이지)
