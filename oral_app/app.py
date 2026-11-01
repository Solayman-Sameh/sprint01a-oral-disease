"""Bilingual Gradio interface. Run with: python -m oral_app.app."""

from html import escape
import base64
import json
import logging
import os
from pathlib import Path
import time

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr
from PIL import Image

from .inference import ImageInputError, MAX_IMAGE_PIXELS, load_model, predict
from .translations import friendly_label, text

ROOT = Path(__file__).resolve().parent.parent
ASSETS = Path(__file__).resolve().parent / "assets"
MODEL_INFO = json.loads((ASSETS / "model_info.json").read_text(encoding="utf-8"))
EXAMPLES = [item for item in json.loads((ASSETS / "examples/manifest.json").read_text(encoding="utf-8"))["examples"]
            if (ASSETS / "examples" / item["file"]).is_file()]
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
LOGGER = logging.getLogger(__name__)


def bundled_font_css():
    """Embed licensed local fonts; no Google Fonts connection is required."""
    rules = []
    for family, filename, weight in [
        ("Manrope", "Manrope-Variable.woff2", "200 800"),
        ("IBM Plex Sans Arabic", "IBMPlexSansArabic-Regular.woff2", "400"),
        ("IBM Plex Sans Arabic", "IBMPlexSansArabic-SemiBold.woff2", "600"),
    ]:
        data = base64.b64encode((ASSETS / "fonts" / filename).read_bytes()).decode("ascii")
        rules.append(f'@font-face {{font-family:"{family}";src:url(data:font/woff2;base64,{data}) format("woff2");'
                     f'font-weight:{weight};font-style:normal;font-display:swap;}}')
    return "\n".join(rules)


def heading_html(lang):
    # :has(.rtl-mode) switches the entire app's direction without custom JS.
    mode = "rtl-mode" if lang == "ar" else "ltr-mode"
    return f'''<div class="app-heading {mode}" lang="{lang}">
        <div class="eyebrow">{text("eyebrow", lang)}</div>
        <h1><span class="tooth-icon" aria-hidden="true">🦷</span> {text("title", lang)}</h1>
        <p>{text("subtitle", lang)}</p></div>'''


def section_heading(key, lang):
    return f'<h2 class="section-title">{escape(text(key, lang))}</h2>'


def empty_results(lang="en", message=None):
    status = f'<p class="input-message" role="alert">{escape(text(message, lang))}</p>' if message else ""
    return f'''<div class="empty-results" lang="{lang}" dir="{"rtl" if lang == "ar" else "ltr"}">
        <span class="empty-icon" aria-hidden="true">🦷</span>
        <h3>{text("empty_title", lang)}</h3><p>{text("empty_hint", lang)}</p>{status}
        <div class="empty-lines" aria-hidden="true"><i></i><i></i><i></i></div></div>'''


def build_results_html(predictions, lang="en"):
    if not predictions:
        return empty_results(lang)
    top = predictions[0]
    top_name = escape(text(top["class"], lang))
    html = f'''<div class="prediction-results" lang="{lang}" dir="{"rtl" if lang == "ar" else "ltr"}">
        <div class="top-result-card"><div class="eyebrow">{text("top_prediction", lang)}</div>
        <h3 class="disease-name">{top_name}</h3>
        <div class="top-score"><span class="friendly-label">{friendly_label(top["probability"], lang)}</span>
        <b dir="ltr">{top["probability"]:.1%}</b></div>
        <p class="technical-detail">{text("softmax", lang)}</p></div>
        <h3 class="scores-heading">{text("all_scores", lang)}</h3>'''
    for index, prediction in enumerate(predictions):
        name = escape(text(prediction["class"], lang))
        label = friendly_label(prediction["probability"], lang)
        percentage = prediction["probability"] * 100
        html += f'''<div class="score-row">
            <div class="score-label"><span>{name}</span><span class="score-value" dir="ltr">{percentage:.1f}%</span></div>
            <div class="prob-bar-track" role="meter" aria-label="{name}" aria-valuemin="0"
                aria-valuemax="100" aria-valuenow="{percentage:.3f}" aria-valuetext="{percentage:.1f}% — {label}">
                <div class="prob-bar-fill {"top" if index == 0 else ""}" style="width:{percentage:.5f}%"></div></div>
            <div class="bar-friendly-label">{label}</div></div>'''
    return html + "</div>"


def about_markdown(lang="en"):
    info = MODEL_INFO
    lines = [
        f'**{text("architecture", lang)}:** ResNet50 · 23.5M',
        f'**{text("accuracy", lang)}:** {info["test_accuracy"]:.2%} · **{text("macro_f1", lang)}:** {info["test_macro_f1"]:.4f}',
        f'### {text("data_title", lang)}', text("data_details", lang),
        f'**{text("dataset", lang)}:** [Kaggle · salmansajid05/oral-diseases](https://www.kaggle.com/datasets/salmansajid05/oral-diseases)',
        f'| {text("class", lang)} | {text("precision", lang)} | {text("recall", lang)} | F1 | {text("support", lang)} |',
        '|---|---:|---:|---:|---:|',
    ]
    # Keep table rows together; blank lines separate paragraphs above and below.
    table = lines[-2:]
    for row in info["per_class"]:
        table.append(f'| {text(row["class"], lang)} | {row["precision"]:.3f} | {row["recall"]:.3f} | {row["f1"]:.3f} | {row["support"]} |')
    return "\n\n".join(lines[:-2]) + "\n\n" + "\n".join(table) + "\n\n" + text("evaluation_note", lang)


def disclaimer_html():
    return f'''<div class="disclaimer">
        <p lang="en">{text("disclaimer", "en")}<br>{text("limitations", "en")}</p>
        <p lang="ar" dir="rtl">{text("disclaimer", "ar")}<br>{text("limitations", "ar")}</p></div>'''


def gallery_items(lang):
    return [(str(ASSETS / "examples" / item["file"]), text(item["class"], lang)) for item in EXAMPLES]


def classify_image(image, lang):
    if image is None:
        return None, empty_results(lang, "no_image")
    try:
        model, idx_to_class = load_model()
        predictions = predict(model, image, idx_to_class)
        return predictions, build_results_html(predictions, lang)
    except ImageInputError as error:
        return None, empty_results(lang, str(error))
    except (OSError, ValueError):
        return None, empty_results(lang, "invalid_image")
    except Exception:
        LOGGER.exception("Image classification failed")
        return None, empty_results(lang, "prediction_error")


def clear_image(lang):
    return None, None, empty_results(lang), gr.update(selected_index=None)


def select_example(lang, event: gr.SelectData):
    index = event.index
    if not isinstance(index, int) or not 0 <= index < len(EXAMPLES):
        raise gr.Error(text("invalid_image", lang))
    with Image.open(ASSETS / "examples" / EXAMPLES[index]["file"]) as image:
        return image.copy(), None, empty_results(lang)


def switch_language(choice, predictions):
    lang = "ar" if choice == "ar" else "en"
    return (
        lang, heading_html(lang), section_heading("upload_title", lang),
        gr.update(label=text("upload_label", lang), placeholder=text("upload_placeholder", lang)),
        text("upload_hint", lang), gr.update(value=text("classify_btn", lang)),
        gr.update(value=text("clear_btn", lang)), section_heading("examples_label", lang),
        text("examples_hint", lang), gr.update(value=gallery_items(lang)),
        section_heading("results_title", lang), text("results_hint", lang),
        build_results_html(predictions, lang), gr.update(label=text("about_title", lang)),
        about_markdown(lang),
    )


def build_app():
    started = time.perf_counter()
    load_model()
    LOGGER.info("Selected ResNet50 loaded once on CPU in %.2fs", time.perf_counter() - started)
    with gr.Blocks(title="Oral Disease Classifier", analytics_enabled=False,
                   delete_cache=(1800, 1800)) as demo:
        lang_state = gr.State("en")
        predictions = gr.State(None)
        with gr.Column(elem_id="app-shell", min_width=0):
            with gr.Row(elem_classes="app-header", equal_height=False):
                heading = gr.HTML(heading_html("en"), elem_id="app-heading", scale=5)
                language = gr.Radio(choices=[("EN", "en"), ("العربية", "ar")], value="en",
                                    show_label=False, container=False, min_width=160, scale=0,
                                    elem_id="language-toggle", elem_classes="lang-toggle")
            with gr.Row(elem_id="main-content", equal_height=True):
                with gr.Column(scale=1, min_width=300, elem_classes="upload-column"):
                    upload_title = gr.HTML(section_heading("upload_title", "en"))
                    image = gr.Image(type="pil", image_mode=None, sources=["upload"], format="png",
                                     label=text("upload_label"), placeholder=text("upload_placeholder"),
                                     height=290, buttons=[], elem_id="image-upload", elem_classes="upload-area")
                    upload_hint = gr.Markdown(text("upload_hint"), elem_classes="upload-hint")
                    with gr.Row(elem_classes="action-buttons"):
                        classify = gr.Button(text("classify_btn"), variant="primary", elem_id="classify-button",
                                             elem_classes="classify-btn", min_width=100, scale=2)
                        clear = gr.Button(text("clear_btn"), elem_id="clear-button", elem_classes="clear-btn",
                                          min_width=80, scale=1)
                    example_title = gr.HTML(section_heading("examples_label", "en"), elem_classes="example-heading", visible=bool(EXAMPLES))
                    example_hint = gr.Markdown(text("examples_hint"), elem_classes="example-hint", visible=bool(EXAMPLES))
                    gallery = gr.Gallery(gallery_items("en"), show_label=False, container=False,
                                         columns=3, rows=2, height=225, object_fit="cover", allow_preview=False,
                                         interactive=False, buttons=[], elem_id="example-gallery", visible=bool(EXAMPLES))
                with gr.Column(scale=1, min_width=300, elem_classes="results-column"):
                    results_title = gr.HTML(section_heading("results_title", "en"))
                    results_hint = gr.Markdown(text("results_hint"), elem_classes="results-hint")
                    results = gr.HTML(empty_results(), elem_id="prediction-output")
            with gr.Accordion(text("about_title"), open=False, elem_id="about-model", elem_classes="about-accordion") as about:
                about_body = gr.Markdown(about_markdown(), elem_id="model-details")
            gr.HTML(disclaimer_html(), elem_id="disclaimer")

        # Serialize image/language changes with classification so a result cannot
        # overwrite a newer upload or leave scores in the previous language.
        event_options = dict(concurrency_id="oral_ui", concurrency_limit=1, show_progress="hidden")
        classify.click(classify_image, [image, lang_state], [predictions, results],
                       api_name="classify", **event_options)
        clear.click(clear_image, [lang_state], [image, predictions, results, gallery],
                    api_name="clear", **event_options)
        image.input(lambda lang: (None, empty_results(lang)), [lang_state], [predictions, results],
                    api_name="image_changed", api_visibility="private", **event_options)
        gallery.select(select_example, [lang_state], [image, predictions, results],
                       api_name="select_example", api_visibility="private", **event_options)
        language.change(switch_language, [language, predictions], [
            lang_state, heading, upload_title, image, upload_hint, classify, clear,
            example_title, example_hint, gallery, results_title, results_hint, results, about, about_body,
        ], api_name="language", **event_options)
    return demo.queue(max_size=16, default_concurrency_limit=1)


def launch_app():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    demo = build_app()
    demo.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0" if os.environ.get("SPACE_ID") else "127.0.0.1"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")), share=False,
        css_paths=[ASSETS / "style.css"],
        theme=gr.themes.Base(font=["Manrope", "IBM Plex Sans Arabic", "Arial", "sans-serif"],
                             font_mono=["Consolas", "monospace"]),
        allowed_paths=[str(ASSETS / "examples")],
        blocked_paths=[str(ROOT / p) for p in ["models", "data", "research", "notebooks"]],
        max_file_size="10mb", footer_links=[], ssr_mode=False,
        css=bundled_font_css(),
    )


if __name__ == "__main__":
    launch_app()
