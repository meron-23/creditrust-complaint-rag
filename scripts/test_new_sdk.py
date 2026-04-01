from google import genai
import os
from dotenv import load_dotenv

def test_gen():
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key)
    
    try:
        response = client.models.generate_content(
            model="models/gemini-3-flash-preview",
            contents="Say hello"
        )
        print(f"Response: {response.text}")
    except Exception as e:
        print(f"Gen Error: {e}")

if __name__ == "__main__":
    test_gen()
