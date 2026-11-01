"""All app text lives here; predictions retain their original class identifiers."""

TEXTS = {
    "title": {"en": "Oral Disease Classifier", "ar": "مُصنِّف أمراض الفم"},
    "subtitle": {"en": "Upload an oral image to explore potential conditions.", "ar": "ارفع صورة فموية لاستكشاف الحالات المحتملة."},
    "eyebrow": {"en": "A CLOSER LOOK AT ORAL HEALTH", "ar": "نظرة أقرب إلى صحة الفم"},
    "upload_title": {"en": "Start with a photo", "ar": "ابدأ بصورة"},
    "upload_label": {"en": "Upload an oral image", "ar": "ارفع صورة فموية"},
    "upload_placeholder": {"en": "Drop an image here or click to browse", "ar": "اسحب صورة هنا أو انقر لاختيارها"},
    "upload_hint": {"en": "JPG, PNG or WebP · up to 10 MB · longest side up to 4,472 px", "ar": "JPG أو PNG أو WebP · حتى 10 ميجابايت · أطول ضلع حتى 4,472 بكسل"},
    "classify_btn": {"en": "Classify", "ar": "صنِّف"},
    "clear_btn": {"en": "Clear", "ar": "مسح"},
    "results_title": {"en": "Your results", "ar": "نتائجك"},
    "results_hint": {"en": "Six conditions, ranked by the model’s confidence.", "ar": "ست حالات مرتّبة حسب درجة ثقة النموذج."},
    "top_prediction": {"en": "TOP PREDICTION", "ar": "التنبؤ الأعلى"},
    "all_scores": {"en": "All scores", "ar": "جميع الدرجات"},
    "softmax": {"en": "Softmax confidence · not a medical probability", "ar": "ثقة النموذج (Softmax) · ليست احتمالًا طبيًا"},
    "empty_title": {"en": "A little clarity starts here", "ar": "هنا تبدأ الرؤية الأوضح"},
    "empty_hint": {"en": "Upload a photo or try an example, then select Classify.", "ar": "ارفع صورة أو جرّب مثالًا، ثم اضغط «صنِّف»."},
    "about_title": {"en": "About this model", "ar": "حول هذا النموذج"},
    "disclaimer": {
        "en": "For educational purposes only. This tool does not replace professional dental diagnosis.",
        "ar": "هذه الأداة لأغراض تعليمية فقط ولا تغني عن التشخيص السني المتخصص.",
    },
    "limitations": {
        "en": "Scores are uncalibrated model confidence. The model has no healthy class and cannot reliably reject non-oral images.",
        "ar": "الدرجات تعكس ثقة النموذج دون معايرة طبية. لا يتضمن النموذج فئة «سليم»، ولا يستبعد الصور غير الفموية بشكل موثوق.",
    },
    "no_image": {"en": "Please upload an image first.", "ar": "يرجى رفع صورة أولًا."},
    "invalid_image": {"en": "That image could not be read. Please try a JPG, PNG or WebP photo.", "ar": "تعذّرت قراءة الصورة. جرّب صورة بصيغة JPG أو PNG أو WebP."},
    "image_too_large": {"en": "Please resize the photo so its longest side is at most 4,472 pixels.", "ar": "يرجى تصغير الصورة بحيث لا يتجاوز أطول ضلع 4,472 بكسل."},
    "prediction_error": {"en": "The image could not be classified. Please try again.", "ar": "تعذّر تصنيف الصورة. يرجى المحاولة مجددًا."},
    "examples_label": {"en": "Try an example", "ar": "جرّب مثالًا"},
    "examples_hint": {"en": "Choose a photo to load it. Examples come from the training set.", "ar": "اختر صورة لتحميلها. هذه الأمثلة من مجموعة التدريب."},
    "Calculus": {"en": "Calculus (Tartar)", "ar": "تراكم الجير"},
    "Caries": {"en": "Dental Caries (Cavities)", "ar": "تسوّس الأسنان"},
    "Gingivitis": {"en": "Gingivitis", "ar": "التهاب اللثة"},
    "Hypodontia": {"en": "Hypodontia", "ar": "نقص الأسنان الخلقي"},
    "Mouth Ulcer": {"en": "Mouth Ulcer", "ar": "تقرّح الفم"},
    "Tooth Discoloration": {"en": "Tooth Discoloration", "ar": "تغيّر لون الأسنان"},
    "very_likely": {"en": "Very Likely", "ar": "مرجّح جدًا"},
    "likely": {"en": "Likely", "ar": "مرجّح"},
    "possible": {"en": "Possible", "ar": "محتمل"},
    "unlikely": {"en": "Unlikely", "ar": "غير مرجّح"},
    "very_unlikely": {"en": "Very Unlikely", "ar": "مستبعد"},
    "architecture": {"en": "Architecture", "ar": "البنية"},
    "accuracy": {"en": "Test accuracy", "ar": "دقة الاختبار"},
    "macro_f1": {"en": "Test macro-F1", "ar": "متوسط F1 للاختبار"},
    "data_title": {"en": "Data & evaluation", "ar": "البيانات والتقييم"},
    "data_details": {
        "en": "3,448 curated images across 6 classes: **2,415 training**, **518 validation**, **515 test**. ResNet50 was fine-tuned from ImageNet; trial 1, epoch 9 was selected by validation macro-F1 before test evaluation. Input: 224 × 224 RGB, black square padding, bilinear resize and ImageNet normalization.",
        "ar": "3,448 صورة منقّحة عبر 6 فئات: **2,415 للتدريب** و**518 للتحقق** و**515 للاختبار**. ضُبط ResNet50 انطلاقًا من ImageNet. اختيرت التجربة الأولى، الحقبة التاسعة، وفق متوسط F1 للتحقق قبل تقييم الاختبار. المدخل: RGB بحجم 224 × 224، مع حشوة مربعة سوداء وتغيير حجم ثنائي الخطية وتطبيع ImageNet.",
    },
    "dataset": {"en": "Source dataset", "ar": "مصدر البيانات"},
    "class": {"en": "Class", "ar": "الفئة"},
    "precision": {"en": "Precision", "ar": "الإحكام"},
    "recall": {"en": "Recall", "ar": "الاستدعاء"},
    "support": {"en": "Test images", "ar": "صور الاختبار"},
    "evaluation_note": {
        "en": "Caries is the weakest class (F1 ≈ 0.70; 28 test images). Related images were grouped to reduce leakage, but patient independence and clinical labels remain unverified. These results describe this dataset; they do not establish clinical performance.",
        "ar": "تسوّس الأسنان هو أضعف الفئات (F1 ≈ 0.70؛ 28 صورة اختبار). جُمّعت الصور المرتبطة لتقليل تسرب البيانات، لكن استقلال المرضى وصحة التسميات السريرية لم يُتحقق منهما. تصف هذه النتائج مجموعة البيانات، ولا تثبت أداءً سريريًا.",
    },
}


def text(key, lang="en"):
    return TEXTS[key]["ar" if lang == "ar" else "en"]


def friendly_label(probability, lang="en"):
    for threshold, key in [(0.70, "very_likely"), (0.40, "likely"), (0.15, "possible"),
                           (0.05, "unlikely"), (0.0, "very_unlikely")]:
        if probability >= threshold:
            return text(key, lang)
    return text("very_unlikely", lang)
