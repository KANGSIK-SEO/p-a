"""
P-A: Paul Guillaume × Ambroise Vollard
미술품 진위 판정 챗봇 — 두 명의 가상 감정가가 동시에 판정합니다.
"""

import gradio as gr
from transformers import pipeline
from PIL import Image

# ─────────────────────────────────────────────────────────────
# 두 명의 "감정가" 모델 (각자 다른 백본을 써서 의견이 갈리도록 구성)
# ─────────────────────────────────────────────────────────────
GUILLAUME_MODEL = "Ateeqq/ai-vs-human-image-detector"            # 12만 장 학습
VOLLARD_MODEL = "umm-maybe/AI-image-detector"                    # ViT 기반

print("Loading Paul Guillaume judge...")
guillaume = pipeline("image-classification", model=GUILLAUME_MODEL)

print("Loading Ambroise Vollard judge...")
vollard = pipeline("image-classification", model=VOLLARD_MODEL)


# ─────────────────────────────────────────────────────────────
# 결과 파싱: 각 모델마다 "진품"으로 간주할 라벨이 다름
# ─────────────────────────────────────────────────────────────
REAL_LABELS = {
    "realism", "real", "human", "hum", "human-made", "natural",
    "authentic", "real_image", "not_ai", "not-ai",
}
AI_LABELS = {"ai", "ai-generated", "artificial", "fake", "deepfake", "deep-fake", "ai_image"}


def real_score(predictions):
    """모델 출력에서 '진품(human/real)'에 해당하는 확률을 추출"""
    # 1순위: 진품 라벨 직접 매칭
    for p in predictions:
        label = p["label"].lower().strip()
        if label in REAL_LABELS or any(k in label for k in ("human", "real", "natural", "authentic")):
            return float(p["score"])
    # 2순위: AI/가품 라벨의 반대값 사용
    for p in predictions:
        label = p["label"].lower().strip()
        if label in AI_LABELS or any(k in label for k in ("ai", "fake", "deep")):
            return 1.0 - float(p["score"])
    # 마지막 fallback
    return 0.5


# ─────────────────────────────────────────────────────────────
# 메인 판정 함수
# ─────────────────────────────────────────────────────────────
THRESHOLD = 0.5


def appraise(image):
    if image is None:
        return "이미지를 업로드해주세요.", "", ""

    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    else:
        image = image.convert("RGB")

    g_pred = guillaume(image)
    v_pred = vollard(image)

    g_real = real_score(g_pred)
    v_real = real_score(v_pred)

    g_pass = g_real >= THRESHOLD
    v_pass = v_real >= THRESHOLD

    if g_pass and v_pass:
        verdict = "🎉 **와우!** 폴 기욤과 앙브루아즈 볼라르 모두 진품 판정을 내렸습니다."
    else:
        verdict = "⚠️ **가품 의심이니 고려해보세요.**"

    g_detail = (
        f"**🎩 폴 기욤 판정**: {'✅ 진품' if g_pass else '❌ 가품 의심'}  \n"
        f"진품 확률 `{g_real:.1%}`"
    )
    v_detail = (
        f"**🎨 앙브루아즈 볼라르 판정**: {'✅ 진품' if v_pass else '❌ 가품 의심'}  \n"
        f"진품 확률 `{v_real:.1%}`"
    )

    return verdict, g_detail, v_detail


# ─────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────
DESCRIPTION = """
# 🖼️ P-A : 폴 기욤 × 앙브루아즈 볼라르

> 두 명의 가상 미술 감정가가 사진을 보고 진위를 판정합니다.

**사용법** : 의심 가는 미술품 사진을 업로드하세요. 두 감정가가 동시에 판정합니다.

- ✅ 둘 다 진품 → 와우!
- ⚠️ 한 명이라도 가품 의심 → 고려해보세요

> ⚠️ *본 서비스는 참고용 사전 판정 도구이며, 실제 감정은 전문 기관에 의뢰하세요.*
"""

with gr.Blocks(title="P-A : 미술품 진위 판정", theme=gr.themes.Soft()) as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="🖼️ 미술품 이미지 업로드")
            submit = gr.Button("🔎 감정 시작", variant="primary")

        with gr.Column(scale=1):
            verdict_out = gr.Markdown(label="최종 판정")
            guillaume_out = gr.Markdown(label="폴 기욤")
            vollard_out = gr.Markdown(label="앙브루아즈 볼라르")

    submit.click(
        fn=appraise,
        inputs=image_input,
        outputs=[verdict_out, guillaume_out, vollard_out],
    )

    gr.Markdown(
        "---\n"
        "Built with 🤗 Transformers · "
        f"Models: `{GUILLAUME_MODEL}` × `{VOLLARD_MODEL}`"
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_api=False)
