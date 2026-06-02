import base64
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def encode_image(image_path: str) -> str:
    """
    Convert image to base64 string.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def explain_figure(image_path: str, question: str = None) -> str:
    """
    Send an image to Groq Vision and get an explanation.
    """
    base64_image = encode_image(image_path)
    
    if question:
        prompt = f"You are an expert in robotics and computer vision research. Answer this question about the image: {question}"
    else:
        prompt = "You are an expert in robotics and computer vision research. Describe this figure in detail. If it is a chart, explain the results. If it is an architecture diagram, explain how it works."

    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
    )
    
    return response.choices[0].message.content


if __name__ == "__main__":
    # Test with a sample image
    print("Testing Groq Vision...")
    print("Please add a test image as 'test_image.jpg' in the project root")
    
    if os.path.exists("test_image.jpg"):
        result = explain_figure("test_image.jpg")
        print(f"\nFigure Explanation:\n{result}")
    else:
        print("No test_image.jpg found — add one to test!")