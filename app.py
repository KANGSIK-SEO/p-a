"""
P-A: Paul Guillaume × Ambroise Vollard
미술품 진위 판정 챗봇 — 두 명의 가상 감정가가 동시에 판정합니다.
"""

import gradio as gr
from transformers import pipeline
from PIL import Image

# ─────────────────────────────────────────────────────────────
# 두 명의 "감정가" 모델
# ─────────────────────────────────────────────────────────────
GUILLAUME_MODEL = "Ateeqq/ai-vs-human-image-detector"
VOLLARD_MODEL = "umm-maybe/AI-image-detector"

print("Loading Paul Guillaume judge...")
guillaume = pipeline("image-classification", model=GUILLAUME_MODEL)

print("Loading Ambroise Vollard judge...")
vollard = pipeline("image-classification", model=VOLLARD_MODEL)


# ─────────────────────────────────────────────────────────────
# 결과 파싱
# ─────────────────────────────────────────────────────────────
REAL_KEYS = ("human", "real", "natural", "authentic", "hum")
AI_KEYS = ("ai", "fake", "deep", "artificial", "gen")


def ai_score(predictions):
    """AI/가품 확률 추출 (0~1)"""
    for p in predictions:
        label = p["label"].lower().strip()
        if any(k in label for k in AI_KEYS):
            return float(p["score"])
    for p in predictions:
        label = p["label"].lower().strip()
        if any(k in label for k in REAL_KEYS):
            return 1.0 - float(p["score"])
    return 0.5


# ─────────────────────────────────────────────────────────────
# 판정 로직: "강한 AI 확신"이 양쪽에 있을 때만 가품 판정
# (그림은 일반적으로 'AI같다'로 살짝 기우는 경향이 있어 관대하게 처리)
# ─────────────────────────────────────────────────────────────
AI_THRESHOLD = 0.85  # 이 값 이상이어야 "확실한 AI"로 봄 (그림에 관대하게)


def appraise(image):
    if image is None:
        return "이미지를 업로드해주세요.", "", ""

    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    else:
        image = image.convert("RGB")

    g_pred = guillaume(image)
    v_pred = vollard(image)

    g_ai = ai_score(g_pred)
    v_ai = ai_score(v_pred)
    g_real = 1.0 - g_ai
    v_real = 1.0 - v_ai

    # 두 감정가 모두 "확실한 AI"로 판정해야 가품
    g_says_fake = g_ai >= AI_THRESHOLD
    v_says_fake = v_ai >= AI_THRESHOLD

    if not g_says_fake and not v_says_fake:
        verdict = "🎉 **와우!** 폴 기욤과 앙브루아즈 볼라르 모두 진품 판정을 내렸습니다."
    elif g_says_fake and v_says_fake:
        verdict = "⚠️ **가품 의심이니 고려해보세요.** (두 감정가 모두 AI 생성 의심)"
    else:
        verdict = "🤔 **의견이 갈립니다.** 한 명만 가품을 의심하므로 추가 검토를 권합니다."

    g_status = "✅ 진품 판정" if not g_says_fake else "❌ 가품 의심"
    v_status = "✅ 진품 판정" if not v_says_fake else "❌ 가품 의심"

    g_detail = (
        f"### 🎩 폴 기욤\n"
        f"- 판정: **{g_status}**\n"
        f"- 진품 확률: `{g_real:.1%}`\n"
        f"- AI 의심도: `{g_ai:.1%}`\n"
        f"- 원본 출력: `{[(p['label'], round(p['score'], 3)) for p in g_pred]}`"
    )
    v_detail = (
        f"### 🎨 앙브루아즈 볼라르\n"
        f"- 판정: **{v_status}**\n"
        f"- 진품 확률: `{v_real:.1%}`\n"
        f"- AI 의심도: `{v_ai:.1%}`\n"
        f"- 원본 출력: `{[(p['label'], round(p['score'], 3)) for p in v_pred]}`"
    )

    return verdict, g_detail, v_detail


# ─────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────
DESCRIPTION = """
# 🖼️ P-A : 폴 기욤 × 앙브루아즈 볼라르

> 두 명의 가상 미술 감정가가 사진을 보고 진위를 판정합니다.

**사용법** : 의심 가는 미술품 사진을 업로드하세요.

- ✅ 둘 다 진품 → 와우!
- ⚠️ 둘 다 가품 의심 (AI 의심도 85% 이상) → 가품 의심
- 🤔 한 명만 의심 → 의견 분분 (추가 검토 권장)

> ⚠️ *본 서비스는 AI 생성 위작 1차 필터링용 참고 도구입니다.
> 정식 감정은 전문 기관에 의뢰하세요.*
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
