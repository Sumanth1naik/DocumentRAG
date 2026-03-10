import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from dotenv import load_dotenv
from langchain_core.documents import Document


load_dotenv()

def load_documents(docs_path="doc"):
    """load all txt froms the txt files from the doc directory"""
    print(f"loading the documents from the {docs_path} ...")
    
    #check if the directory exists or not
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The {docs_path} does not exist, Please create it and add the comany data")
    
    #Load all the .txt files data
    
    loader = DirectoryLoader(
        path = docs_path,
        glob ="*.txt",
        loader_cls = TextLoader,
        loader_kwargs = {"encoding": "utf-8"}
    )
    
    documents = loader.load()
    
    if documents == 0:
        raise FileNotFoundError(f"No .txt files found at {docs_path}, Please add the data...")
    
    for i,doc in enumerate(documents[:2]):
        print(f"\n Documents {i+1}....")
        print(f"Source {doc.metadata['source']}")
        print(f"document length {len(doc.page_content)} characters")
    
    return documents

def split_documents(documents, chunk_size, chunk_overlap):
    """Split documents into smaller chunks with overlap"""
    print("Splitting documents into chunks...")
    
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    
    chunks = text_splitter.split_documents(documents)
    
    # Enforce maximum chunk size by further splitting oversized chunks
    final_chunks = []
    for chunk in chunks:
        if len(chunk.page_content) > chunk_size:
            # Split oversized chunks into smaller pieces
            words = chunk.page_content.split()
            current_chunk = ""
            for word in words:
                if len(current_chunk) + len(word) + 1 <= chunk_size:
                    current_chunk += word + " "
                else:
                    if current_chunk:
                        final_chunks.append(Document(
                            page_content=current_chunk.strip(),
                            metadata=chunk.metadata
                        ))
                    current_chunk = word + " "
            if current_chunk:
                
                final_chunks.append(Document(
                    page_content=current_chunk.strip(),
                    metadata=chunk.metadata
                ))
        else:
            final_chunks.append(chunk)
    
    chunks = final_chunks
    
    if chunks:
    
        for i, chunk in enumerate(chunks[:5]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"Content:")
            print(chunk.page_content)
            print("-" * 50)
        
        if len(chunks) > 5:
            print(f"\n... and {len(chunks) - 5} more chunks")
    
    return chunks

def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """Create and persist ChromaDB vector store"""
    print("Creating embeddings and storing in ChromaDB...")
    
    embedding_model = OllamaEmbeddings(
        model="nomic-embed-text"
    )
    
    # Create ChromaDB vector store
    print("--- Creating vector store ---")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory, 
        collection_metadata={"hnsw:space": "cosine"}
    )
    print("--- Finished creating vector store ---")
    
    print(f"Vector store created and saved to {persist_directory}")
    return vectorstore


def main():
    print("Main Function")
    #1. Loading the files
    documents =  load_documents(docs_path="doc")
    #2. Chunking the files
    chunks = split_documents(documents, chunk_size=64, chunk_overlap=10)
    #3. Storing the chunk in vector DB
    vector_store = create_vector_store(chunks)
    




if __name__ == "__main__":
    main()