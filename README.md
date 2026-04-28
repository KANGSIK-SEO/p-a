---
title: P-A Paul Guillaume × Ambroise Vollard
emoji: 🖼️
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: apache-2.0
---

# 🖼️ P-A : 폴 기욤 × 앙브루아즈 볼라르

> 두 명의 가상 미술 감정가가 동시에 판정하는 미술품 진위 판별 챗봇

## 개요

폴 기욤(Paul Guillaume)과 앙브루아즈 볼라르(Ambroise Vollard)는 20세기 초 파리에서 활동한
전설적인 미술상이자 감정가입니다. 이 서비스는 그들의 이름을 빌려, 두 개의 독립적인
이미지 분류 모델을 **앙상블**하여 미술품의 진위를 사전 판정합니다.

- ✅ **두 감정가 모두 진품 판정** → "와우!"
- ⚠️ **한 명이라도 가품 의심** → "가품 의심이니 고려해보세요"

## 사용 모델

| 감정가 | 모델 | 라이선스 |
|---|---|---|
| 🎩 폴 기욤 | [`prithivMLmods/Deep-Fake-Detector-v2-Model`](https://huggingface.co/prithivMLmods/Deep-Fake-Detector-v2-Model) | Apache 2.0 |
| 🎨 앙브루아즈 볼라르 | [`umm-maybe/AI-image-detector`](https://huggingface.co/umm-maybe/AI-image-detector) | 공개 |

## 로컬 실행

```bash
pip install -r requirements.txt
python app.py
```

브라우저에서 http://127.0.0.1:7860 접속.

## Hugging Face Spaces 배포

이 저장소를 그대로 [HF Spaces](https://huggingface.co/spaces)에 연결하면 자동 배포됩니다.

1. https://huggingface.co/new-space 에서 새 Space 생성 (SDK: Gradio)
2. `Repository` → 이 GitHub 레포 연결 또는 파일 직접 업로드
3. 자동 빌드 후 사용 가능

## 한계

본 서비스는 일반적인 AI 생성 이미지 / 딥페이크 탐지 모델을 활용한
**참고용 사전 판정 도구**입니다. 정밀한 미술품 감정은 반드시
전문 감정 기관에 의뢰하세요.

## License

Apache License 2.0
