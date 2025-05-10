import chromadb
from sentence_transformers import SentenceTransformer
import re
from langchain_community.document_loaders import PDFPlumberLoader
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer('BAAI/bge-base-en-v1.5')

# run locally: chroma run --host localhost --port 8000
chroma_client = chromadb.HttpClient(host= "localhost", port = 8000)

try:
    chroma_client.delete_collection(name = 'contracts')
except:
    pass

# Create a list of file paths
file_paths = [
    'documents/Agreement - Global Health and PharmaTech.pdf',
    'documents/Marketing Agreement - BrightAd Agency and FashionFiesta.pdf',
    'documents/Distribution Agreement - OrganicFoods Co. and RetailKing.pdf',
    'documents/Supplier Agreement - HighTech Electronics and ComponentSource.pdf'
]



all_data = []

# Loop through each file path and process the file
for file_path in file_paths:
    # Initialize the PDFPlumberLoader with the current file path
    loader = PDFPlumberLoader(file_path)

    # Load the data from the current file
    data = loader.load()

    # Append the loaded data to the all_data list
    all_data.append(data)

# Initialize a list to store the combined data
combined_data = []

from langchain.text_splitter import RecursiveCharacterTextSplitter

# Text splitter
chunk_size = 250
chunk_overlap = 50
text_splitter = RecursiveCharacterTextSplitter(
    separators=[".", ".\n"],
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    length_function=len,
)
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=10000,  # No chunk size limit
#     chunk_overlap=0,  # No overlap
#     separators=["\n\n"]  # Use double newlines for paragraph separation
# )

chunks = []

# Loop through each item in all_data
for item in all_data:
    document = item[0]

    # Extract the page_content
    page_content = document.page_content

    # Extract filename and page from metadata
    filename = document.metadata['source']
    page = document.metadata['page']

    # Chunk a section
    chunks.extend(text_splitter.create_documents(
        texts=[page_content],
        metadatas=[{"source": filename, "page": page}]
    ))

#vectorize the chunks
all_chunk_vectors = []
all_chunk_contents = []
all_chunk_metadata = []
all_chunk_ids = []

for idx, content in enumerate(chunks):
    # print(dict(content.to_json()['kwargs']['metadata'])['source'])
    content_to_encode = str(content.to_json()['kwargs']['page_content'])+str(dict(content.to_json()['kwargs']['metadata'])['source'])
    # content_to_encode = str(content['page_content'])
    all_chunk_contents.append(content_to_encode)

    metadata_of_content = content.to_json()['kwargs']['metadata']
    # metadata_of_content = content['metadata']
    all_chunk_metadata.append(metadata_of_content)

    source_of_content = content.to_json()['kwargs']['metadata']['source']
    # source_of_content = content['metadata']['source']
    # Convert to lowercase
    source_of_content = source_of_content.lower()
    # Replace spaces with underscores
    source_of_content = source_of_content.replace(' ', '_')
    # Remove special characters (keep only alphanumeric and underscores)
    source_of_content = re.sub(r'[^a-zA-Z0-9_]', '', source_of_content)
    source_of_content += "_chunk_"+str(idx)
    all_chunk_ids.append(source_of_content)

    chunk_vector = embedding_model.encode(content_to_encode).tolist()
    all_chunk_vectors.append(chunk_vector)

print("Input size: ", len(all_chunk_vectors))


collection = chroma_client.create_collection(name="contracts")
collection.add(
    embeddings=all_chunk_vectors,
    documents=all_chunk_contents,
    metadatas=all_chunk_metadata,
    ids=all_chunk_ids
)
collection_retrieved = chroma_client.get_collection(name="contracts") # Get a collection object from an existing collection, by name. Will raise an exception if it's not found.
# print(collection_retrieved.peek())
print("Retrieved size: ", collection_retrieved.count())