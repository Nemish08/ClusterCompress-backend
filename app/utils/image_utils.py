import base64

def image_bytes_to_base64_string(image_bytes: bytes) -> str:
    """
    Encodes image bytes into a data URL (base64 string) for easy embedding in
    JSON and HTML/CSS.
    """
    encoded_bytes = base64.b64encode(image_bytes)
    encoded_string = encoded_bytes.decode('utf-8')
    # The data URL format is required for the frontend to render the image directly
    return f"data:image/png;base64,{encoded_string}"