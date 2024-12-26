from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from database import load_vectorstore

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")
vectorstore = load_vectorstore()


# Function to manually retrieve documents
def retrieve_documents(query):
    return vectorstore.similarity_search(query, 1)


# Function to format documents into a single context string
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Custom function to create a prompt
def create_prompt(context, question):
    return f"""
Bạn là một trợ lý AI chuyên nghiệp về lĩnh vực luật đất đai tại Việt Nam. Hãy sử dụng thông tin sau để trả lời câu hỏi của người dùng một cách ngắn gọn, chính xác và kèm theo các căn cứ pháp lý cụ thể (bao gồm điều, khoản, chương của luật, nếu có).

Thông tin:
{context}

Câu hỏi:
{question}

Trả lời (kèm căn cứ pháp lý):
"""



# Query for task decomposition
query = "nội dung của Điều 4. Người sử dụng đất?"

# Retrieve and format documents
retrieved_docs = retrieve_documents(query)
context = format_docs(retrieved_docs)
print(context)

# Create a manual prompt
custom_prompt = create_prompt(context, query)

# Invoke the LLM directly
response = llm.invoke(custom_prompt)

# Print the answer
print(response.content)
