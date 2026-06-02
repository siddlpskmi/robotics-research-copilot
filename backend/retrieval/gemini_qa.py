from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def answer_question(question: str, context_chunks: list[str]) -> str:
    context = "\n\n".join(context_chunks)
    
    prompt = f"""You are an expert research assistant specializing in robotics and computer vision.

Use ONLY the context below to answer the question.
If the answer is not in the context, say "I couldn't find that in the provided paper."

Context:
{context}

Question: {question}

Answer:"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    sample_chunks = [
        "The RPN shares convolutional features with the detection network.",
        "Faster R-CNN achieves real-time object detection using region proposals.",
        "The system processes images at 5 fps on a GPU.",
        "The model uses anchor boxes of different scales and aspect ratios."
    ]
    
    question = "How fast does the system process images?"
    print(f"Question: {question}")
    print("\nAsking Groq...")
    answer = answer_question(question, sample_chunks)
    print(f"\nAnswer: {answer}")
