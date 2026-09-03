import base64
import os
import sys
import numpy as np
import cv2
from openai import OpenAI



UNSLOTH_BASE_URL = os.getenv("BASE_URL", "http://localhost:8888/v1")
UNSLOTH_API_KEY = os.getenv("API_KEY", "api-key")
MODEL_NAME = os.getenv("MODEL", "default")

client = OpenAI(
    base_url=UNSLOTH_BASE_URL,
    api_key=UNSLOTH_API_KEY,
)



def call_llm(user_input: str, system_prompt: str = "You are a helpful assistant.") -> str | None:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]

    try:
        response = client.chat.completions.create(
            model="unsloth-local",
            messages=messages,
            temperature=0.7
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"An error occurred while communicating with API: {e}"


def encode_image_to_base64(frame: np.ndarray) -> str:
    """Encodes an OpenCV image frame to a Base64 string."""
    success, buffer = cv2.imencode(".jpg", frame)
    if not success:
        raise ValueError("Failed to encode image to JPG format.")
    return base64.b64encode(buffer).decode("utf-8")


def describe_image(base64_image: str) -> str:
    print("Processing image...")
    prompt_text = ("""You are a assistive tool for people with vision imparements you need to analyse the given image 
    and generate a brief description of what is in the image and transcribe any text that is visible in the image you 
    need to brief and to the point the final output must contain only the relevant information and must be very short, 
    you must be very short and to the point and have at max 20 words and describe only the relevant details for the 
    person to navigate, give only short useful information, MAX 10 WORDS"""
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=500,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"An error occurred while communicating with API: {e}"

