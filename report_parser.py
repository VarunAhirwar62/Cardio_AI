"""Read a medical report (digital PDF, scanned PDF or image) and pull out the values the models need."""
import io
import re

RANGES = {"age": (1, 120), "bp": (70, 250), "glucose": (40, 500), "chol": (100, 600), "bmi": (10, 70),
          "thalach": (60, 220), "oldpeak": (0, 7)}


def extract_text(data: bytes, filename: str):
    """Returns (text, notes). Digital PDFs are read directly; scans/images need Tesseract OCR."""
    notes, name = [], filename.lower()
    if name.endswith(".pdf"):
        try:
            import pymupdf
        except ImportError:
            import fitz as pymupdf
        doc = pymupdf.open(stream=data, filetype="pdf")
        text = "\n".join(p.get_text() for p in doc)
        if len(text.strip()) >= 30:
            return text, notes
        notes.append("No selectable text found, so OCR was used (accuracy is lower; verify every value).")
        imgs = [p.get_pixmap(dpi=200).tobytes("png") for p in doc]
    else:
        imgs = [data]
        notes.append("Image report read with OCR (accuracy is lower; verify every value).")
    try:
        import pytesseract
        from PIL import Image
        return "\n".join(pytesseract.image_to_string(Image.open(io.BytesIO(b))) for b in imgs), notes
    except Exception as e:
        raise RuntimeError("OCR is not available on this machine (install Tesseract). "
                           "Digital PDFs still work. Details: " + str(e)[:80])


def _num_after(pattern, text, gap=70):
    """First number after a label, ignoring (reference ranges) and '<200' style limits."""
    for m in re.finditer(pattern, text, re.I):
        seg = re.sub(r"\([^)]*\)", " ", text[m.end(): m.end() + gap])
        seg = re.sub(r"[<>≤≥]\s*=?\s*\d+(?:\.\d+)?", " ", seg)
        n = re.search(r"\d+(?:\.\d+)?", seg)
        if n:
            snippet = " ".join(text[m.start(): m.end() + n.end()].split())
            return float(n.group()), snippet
    return None


def parse_report(text):
    """Returns (found, warnings). found: key -> (value, matched text)."""
    found, warns = {}, []
    t = text.replace("\u00a0", " ")

    m = re.search(r"age\s*/\s*(?:sex|gender)\s*[:\-]?\s*(\d{1,3})\s*(?:y|yrs?|years?)?\s*/\s*(male|female|m|f)\b", t, re.I)
    if m:
        found["age"] = (float(m.group(1)), m.group(0))
        found["sex"] = ("Male" if m.group(2).lower().startswith("m") else "Female", m.group(0))
    else:
        r = _num_after(r"\bage\b\s*[:\-]?", t, 12)
        if r:
            found["age"] = r
        m = re.search(r"(?:sex|gender)\s*[:\-]?\s*(male|female|m|f)\b", t, re.I)
        if m:
            found["sex"] = ("Male" if m.group(1).lower().startswith("m") else "Female", m.group(0))

    m = re.search(r"(?:blood\s*pressure|\bbp\b)[^0-9]{0,25}(\d{2,3})\s*/\s*(\d{2,3})", t, re.I)
    if m:
        found["bp"] = (float(m.group(1)), m.group(0))

    r = _num_after(r"total\s+cholesterol", t)
    if not r:  # plain 'cholesterol' that is not HDL/LDL/VLDL
        for m in re.finditer(r"cholesterol", t, re.I):
            if not re.search(r"(hdl|ldl|vldl|non)[\s\-]*$", t[max(0, m.start() - 8): m.start()], re.I):
                r = _num_after(re.escape(m.group()) + r"(?!\s*(?:/|ratio))", t[m.start():])
                break
    if r:
        v, s = r
        if v < 20:
            v, s = v * 38.67, s + "  (converted from mmol/L)"
        found["chol"] = (round(v, 1), s)

    for pat in (r"fasting\s+(?:blood\s+)?(?:glucose|sugar|plasma\s+glucose)", r"\bfbs\b|\bfbg\b",
                r"(?:random|average|mean)\s+(?:blood\s+)?(?:glucose|sugar)", r"blood\s+(?:glucose|sugar)|\bglucose\b"):
        r = _num_after(pat, t)
        if r:
            v, s = r
            if v < 30:
                v, s = v * 18.016, s + "  (converted from mmol/L)"
            found["glucose"] = (round(v, 1), s)
            break

    r = _num_after(r"\bbmi\b|body\s+mass\s+index", t)
    if r:
        found["bmi"] = r
    else:
        h, w = _num_after(r"\bheight\b[^0-9]{0,15}", t, 20), _num_after(r"\bweight\b[^0-9]{0,15}", t, 20)
        if h and w and 100 < h[0] < 230:
            found["bmi"] = (round(w[0] / (h[0] / 100) ** 2, 1), f"{h[1]}; {w[1]} (BMI calculated)")

    r = _num_after(r"max(?:imum)?\.?\s*heart\s*rate(?:\s*achieved)?|peak\s*heart\s*rate|\bmax\s*hr\b", t)
    if r:
        found["thalach"] = r
    r = _num_after(r"st[\s\-]*(?:segment\s*)?depression|oldpeak", t, 30)
    if r:
        found["oldpeak"] = r

    for key, pat in (("exang", r"exercise[\s\-]*induced\s+angina"), ("hypertension", r"hypertension"),
                     ("heart_disease", r"heart\s+disease")):
        m = re.search(pat + r"\s*[:\-]?\s*(yes|no)\b", t, re.I)
        if m:
            found[key] = (m.group(1).capitalize(), m.group(0))
    m = re.search(r"smok\w*\s*[:\-]?\s*(never|non|no|former|ex|current|yes)\w*", t, re.I)
    if m:
        w = m.group(1).lower()
        found["smoking"] = ("formerly smoked" if w in ("former", "ex") else "smokes" if w in ("current", "yes")
                            else "never smoked", m.group(0))
    if re.search(r"non[\s\-]*smoker", t, re.I) and "smoking" not in found:
        found["smoking"] = ("never smoked", "non-smoker")

    for k, (lo, hi) in RANGES.items():
        if k in found and not (lo <= found[k][0] <= hi):
            warns.append(f"Ignored {k}: value {found[k][0]} is outside the plausible range {lo}-{hi}.")
            del found[k]
    return found, warns
