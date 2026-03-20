def postprocess_bengali_caption(raw_text: str) -> str:
    cleaned = raw_text.strip()
    if not cleaned.endswith(('।', '.', '?')):
        cleaned += '।'
    return cleaned
