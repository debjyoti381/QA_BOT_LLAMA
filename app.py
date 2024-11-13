from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import CTransformers
import pandas as pd
import os

# Load multiple CSV files and convert to a combined string text
def get_combined_csv_text(directory_path):
    all_text = []
    
    # Loop through each file in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory_path, filename)
            df = pd.read_csv(file_path)
            text = df.to_string(index=False)
            all_text.append(text)
    
    # Combine all text into one large string
    combined_text = "\n".join(all_text)
    return combined_text
# Split text into manageable chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
    chunks = text_splitter.split_text(text)
    return chunks
# Create a FAISS vector store from text chunks
def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
from huggingface_hub import hf_hub_download

# https://huggingface.co/docs/huggingface_hub/v0.16.3/en/package_reference/file_download#huggingface_hub.hf_hub_download
hf_hub_download(
    repo_id="TheBloke/Llama-2-7B-Chat-GGML",
    filename="llama-2-7b-chat.ggmlv3.q4_0.bin",
    local_dir="./models"
)
def get_conversational_chain():
    # Initialize the LLM
    llm = CTransformers(
        model="models/llama-2-7b-chat.ggmlv3.q4_0.bin",
        model_type="llama",
        config={'max_new_tokens': 512, 'temperature': 0.8}
    )
    
    # Create the prompt template
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in the context, say, "answer is not available in the context".
    
    Context: {context}
    Question: {question}
    
    Answer: """
    
    PROMPT = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question"]
    )
    
    # Load the vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    # Create and return the chain
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    return chain
# Process user question and return answer
def process_user_input(user_question):
    # Get the conversational chain and process the answer
    chain = get_conversational_chain()
    response = chain.invoke({"query": user_question})
    
    print("Reply:", response["result"])
# Main function to load CSVs, prepare vector store, and handle user questions
def main():
    print("Chat with CSV Data")
    
    # Directory containing CSV files
    csv_directory = "csv_files"
    
    # Load and combine CSV data into text format
    raw_text_nodes = get_combined_csv_text(csv_directory)
    
    # Create text chunks
    text_chunks = get_text_chunks(raw_text_nodes)
    
    # Build the FAISS vector store
    get_vector_store(text_chunks)
    
    while True:
        try:
            # Continuously ask questions based on user input
            user_question = input("\nAsk a question about the CSV data (or type 'exit' to quit): ")
            
            if user_question.lower() == 'exit':
                print("Goodbye!")
                break
                
            process_user_input(user_question)
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()