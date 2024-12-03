from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
import os
import json
import PyPDF2
from fixthaipdf import clean
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import requests
import base64
# New imports for PDF QA functionality
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

app = Flask(__name__)

# Set your Groq API key
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    api_key = "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"
    print("Warning: Using hardcoded API key. It's recommended to use environment variables instead.")

if not api_key or api_key == "your_api_key_here":
    raise ValueError("Please set the GROQ_API_KEY environment variable or replace the placeholder in the code.")

# Ensure the uploads directory exists
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

output_format = """
[
    {
        "Name": "....",
        "Image": ".....",
        "Question": "....",
        "Answer": ".....",
        "Sound": ".....",
        "Story": "...",
        "Character": "..."
    },
    {
        "Name": ".....",
        "Image": ".....",
        "Question": ".....",
        "Answer": ".....",
        "Sound": "....",
        "Story": "...",
        "Character": "..."
    }
]
"""

template = """
Create 5 Flash Cards. Each card must contain:
Name: The topic of the content for the flash card (Thai).
Image: A detailed explanation prompt caption of the image that will be used to generate the image (English).
Question: A question that is simple and easy to understand for children (Thai).
Answer: The answer to the question with descriptions that helps enhance the understanding of the content (Thai).
Sound: A sound, keyword, catchphrase that help children remembers and stay engaged (Thai).
Story: The story that connects the contents of each of the flash card together. This allows children to follow the story continuously and have fun (Thai).
Character: A character that is related to the content of the flash card. The character should be fun and engaging for children (Thai Language).

Use simple and clear language.
Here are the contents:
{pdf_text}

Output Format:
```
{output_format}
```
"""

url = "https://api.hyperbolic.xyz/v1/image/generation"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJqb3A0NDUwQGdtYWlsLmNvbSJ9.vgCJB7yxFiJOpsIxQdDkPtq5DDGph9YgxpNo4myZjWQ"
}

prompt_template = PromptTemplate(
    template=template,
    input_variables=["pdf_text", "output_format"]
)

# Initialize the ChatGroq model
llm = ChatGroq(
    groq_api_key=api_key,
    model_name="llama-3.1-70b-versatile",
    max_tokens=8000,
    temperature=0.5,
)

# New variables and functions for PDF QA functionality
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

store = {}
current_rag_chain = None

def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_kwargs={"device": "cpu", "trust_remote_code": True}, model_name="Alibaba-NLP/gte-multilingual-base")

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as file:
        pdf = PyPDF2.PdfReader(file)
        text = '\n'.join([i.extract_text() for i in pdf.pages])
        text = clean(text)
    return text

def extract_and_parse_json(input_text):
    start_index = input_text.find('[')
    end_index = input_text.rfind(']')
    if start_index != -1:
        json_text = input_text[start_index:end_index+1]
        try:
            json_data = json.loads(json_text)
            return json_data
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return None
    else:
        print("No JSON array found in the input text.")
        return None

# Function to process PDF and create RAG chain
def process_pdf_for_chat(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    with open(pdf_path.split(".")[0] + ".txt", "w", encoding="utf-8") as f:
        f.write(text)
    loader = TextLoader(file_path=pdf_path.split(".")[0] + ".txt", encoding="utf-8")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(texts, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("user", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
    )
    qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("user", "{input}"),
    ]
    )
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    return conversational_rag_chain

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/extract', methods=['POST'])
def handle_extract():
    if 'pdf' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['pdf']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        filename = secure_filename(file.filename)
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(pdf_path)
        
        # Process the PDF and create RAG chain
        global current_rag_chain
        current_rag_chain = process_pdf_for_chat(pdf_path)
        
        text = extract_text_from_pdf(pdf_path)
        
        summarize = llm.invoke(f"""
        Make the content easier to understand but the key content should remains the same, Use the language that the original content is written in. If the content is in Thai, the output should be in Thai. If the content is in English, the output should be in English. If the content is in both Thai and English, the output should be in both Thai and English.
        Content:
        {text}

        Output only the text.:
        [Summarization Text]
        """)
        
        values = {
            "pdf_text": summarize.content,
            "output_format": output_format 
        }
        prompt = prompt_template.format(**values)
        response = llm.invoke(prompt)
        
        card = extract_and_parse_json(response.content)
        
        # Save the JSON file
        json_filename = "flashcard.json"
        with open(os.path.join(app.config['UPLOAD_FOLDER'], json_filename), "w", encoding="utf-8") as f:
            json.dump(card, f, ensure_ascii=False, indent=4)
        
        # Generate images
        img_arr = []
        for v in card:
            data = {
                "model_name": "SDXL1.0-base",
                "prompt": "A cartoony image of " + v["Image"],
                "steps": 30,
                "cfg_scale": 5,
                "enable_refiner": False,
                "height": 1024,
                "width": 1024,
                "backend": "auto"
            }
            response = requests.post(url, headers=headers, json=data)
            img_arr.append(response.json()["images"][0]["image"])
        
        for i in range(len(card)):
            card[i]['Image'] = img_arr[i]
        
        # Save the updated JSON file with image data
        with open(os.path.join(app.config['UPLOAD_FOLDER'], json_filename), "w", encoding="utf-8") as f:
            json.dump(card, f, ensure_ascii=False, indent=4)
        
        return redirect(url_for('show_characters'))

@app.route('/characters')
def show_characters():
    # Load the flashcards from the JSON file
    json_path = os.path.join(app.config['UPLOAD_FOLDER'], 'flashcard.json')
    if not os.path.exists(json_path):
        return "No flashcards available. Please upload a PDF first.", 404
    
    with open(json_path, 'r', encoding='utf-8') as f:
        flashcards = json.load(f)
    
    # Extract unique characters
    characters = list(set(card['Character'] for card in flashcards))
    
    return render_template('characters.html', characters=characters)

@app.route('/story/<int:index>')
def show_story(index):
    # Load the flashcards from the JSON file
    json_path = os.path.join(app.config['UPLOAD_FOLDER'], 'flashcard.json')
    if not os.path.exists(json_path):
        return "No flashcards available. Please upload a PDF first.", 404
    
    with open(json_path, 'r', encoding='utf-8') as f:
        flashcards = json.load(f)
    
    if index < 0 or index >= len(flashcards):
        return redirect(url_for('show_characters'))
    
    card = flashcards[index]
    return render_template('story.html', card=card, index=index, total=len(flashcards))

@app.route('/chat', methods=['POST'])
def chat():
    if current_rag_chain is None:
        return jsonify({"error": "No PDF processed. Please upload a PDF first."}), 400
    
    question = request.form.get('question')
    if not question:
        return jsonify({"error": "No question provided"}), 400
    
    response = current_rag_chain.invoke(
        {"input": question},
        config={
            "configurable": {"session_id": "flashcard_qa"}
        },
    )
    
    return jsonify({"answer": response['answer']})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.getenv("PORT", default=5000)))