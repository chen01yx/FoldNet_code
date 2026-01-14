import os

import base64
import json
import numpy as np

import requests
from openai import OpenAI

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

class JsonEncoder(json.JSONEncoder):
    """Convert numpy classes to JSON serializable objects."""

    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(JsonEncoder, self).default(obj)

class GPT_Client:
    def __init__(self, timeout=10):
        self.timeout = timeout

    def get_response(self, messages, model="gpt-4o", max_tokens=2048, timeout=10):
        pass

    def infer_score(self, prompt, images):
        # Getting the base64 string
        base64_images = [encode_image(image) for image in images]

        content = []
        content.append({"type": "text", "text": prompt})
        for img in base64_images:
            content.append({
	    	    "type": "image_url",
                "image_url": {
                	"url": f"data:image/jpeg;base64,{img}"
                }})

        messages=[
            {
                "role": "system",
                "content": """
                    You are a helpful AI assistant.
                """
            },
            {
                "role": "user",
                "content": content,
            },
        ]

        scores = self.get_response(model="gpt-4o", messages=messages, max_tokens=2048, timeout=self.timeout)
        if scores is None:
            return None
        return np.array(scores.strip().split())
    
class OpenAI_Client(GPT_Client):
    def __init__(self, timeout=10):
        super().__init__(timeout=timeout)
        self.client = OpenAI()

    def get_response(self, messages, model="gpt-4o", max_tokens=2048, timeout=10):
        try:
            response = self.client.chat.completions.create(
    	    	model=model,
    	    	messages=messages,
    	    	max_tokens=max_tokens,
                timeout=timeout
    	    )
            return response.choices[0].text
        except:
            print("[ ERROR ] OpenAI API Timeout!")
            return None

class DMIAPI_Client(GPT_Client):
    def __init__(self, timeout=10):
        super().__init__(timeout=timeout)
        self.url = "https://vip.DMXapi.com/v1/chat/completions"
        self.headers = {
            'Accept': 'application/json',
            'Authorization': os.environ.get("DMI_API_KEY"), # PUT YOUR KEY HERE
            'User-Agent': 'mtuopenai/1.0.0 (https://vip.dmxapi.com)',
            'Content-Type': 'application/json'
        }

    def get_response(self, messages, model="gpt-4o", max_tokens=2048, timeout=10):
        payload = json.dumps({
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
        })

        try:
            response = requests.request("POST", self.url, headers=self.headers, 
                                        data=payload, timeout=timeout).json()
            return response['choices'][0]['message']['content']
        except:
            print("[ ERROR ] DMI API Timeout!")
            return None