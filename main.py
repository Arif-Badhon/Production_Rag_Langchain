# pyrefly: ignore [missing-import]

from dotenv import load_dotenv
load_dotenv()
from langchain_core import __version__ as langchain_core_version
#from langgraph import __version__ as langgraph_version
from langchain_google_genai import GoogleGenerativeAI


print(f"Langchain core version: {langchain_core_version}")
#print(f"Langgraph version: {langgraph_version}")
#print(f"GoogleGenerativeAI: {GoogleGenerativeAI}")


def main():
    llm = GoogleGenerativeAI(
        model="gemini-3.8-flash",
        temperature=0.7,
        max_output_tokens=1024,
        top_p=0.9,
        top_k=40,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    
    
    response = llm.invoke("Tell me a joke")
    print(f"Response: {response}")


if __name__ == "__main__":
    main()