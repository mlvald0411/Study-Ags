import os
import pandas as pd
from dotenv import load_dotenv
from llama_index.core.query_engine import PandasQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.agent import ReActAgent
from llama_index.llms.groq import Groq  # Ensure this is included
from llama_index.core import SimpleDirectoryReader
from pytube import YouTube
import whisper

# Load environment variables
load_dotenv()

### File Loaders ###

# Function to dynamically load CSV files
def load_csv():
    csv_path = input("Please enter the path to your CSV file: ").strip()
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return None
    try:
        csv_df = pd.read_csv(csv_path)
        return csv_df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

# Function to dynamically load Excel files
def load_excel():
    excel_path = input("Please enter the path to your Excel file: ").strip()
    if not os.path.exists(excel_path):
        print(f"File not found: {excel_path}")
        return None
    try:
        excel_df = pd.read_excel(excel_path)
        return excel_df
    except Exception as e:
        print(f"Error loading Excel: {e}")
        return None

# Function to dynamically load PDF files
def load_pdf():
    pdf_path = input("Please enter the path to your PDF file: ").strip()
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        return None
    try:
        #from llama_index.core import PDFReader
        pdf_reader = SimpleDirectoryReader(input_files=[pdf_path])
        pdf_data = pdf_reader.load_data()
        return pdf_data
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return None

# Function to download and transcribe YouTube video
def download_and_transcribe_youtube(video_url, save_path, transcription_model="base"):
    yt = YouTube(video_url)
    video = yt.streams.filter(only_audio=True).first()
    video.download(output_path=save_path, filename="audio.mp4")
   
    model = whisper.load_model(transcription_model)
    audio_path = os.path.join(save_path, "audio.mp4")
    transcription_result = model.transcribe(audio_path)
   
    transcription_path = os.path.join(save_path, "transcript.txt")
    with open(transcription_path, 'w') as f:
        f.write(transcription_result['text'])
   
    return transcription_path

# Function to create an index from a PDF file
def get_pdf_index(pdf_data, index_name):
    index = None
    if not os.path.exists(index_name):
        print("Building index", index_name)
        index = VectorStoreIndex.from_documents(pdf_data, show_progress=True)
        index.storage_context.persist(persist_dir=index_name)
    else:
        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=index_name)
        )
    return index

# Function to create an index from YouTube transcript
def get_youtube_index(transcript_path, index_name):
    index = None
    if not os.path.exists(index_name):
        print("Building index", index_name)
        with open(transcript_path, 'r') as f:
            data = f.readlines()
        index = VectorStoreIndex.from_documents(data, show_progress=True)
        index.storage_context.persist(persist_dir=index_name)
    else:
        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=index_name)
        )
    return index


### Main Function ###
def main():
    tools = []
   
    # Ask the user what type of file they want to query
    file_type = input("What type of file would you like to query (CSV, Excel, PDF, or YouTube)? ").strip().lower()
   
    if file_type == "csv":
        csv_df = load_csv()
        if csv_df is not None:
            csv_query_engine = PandasQueryEngine(df=csv_df, verbose=True)
            tools.append(
                QueryEngineTool(
                    query_engine=csv_query_engine,
                    metadata=ToolMetadata(
                        name="csv_data",
                        description="This provides information from the CSV file.",
                    ),
                )
            )
   
    elif file_type == "excel":
        excel_df = load_excel()
        if excel_df is not None:
            excel_query_engine = PandasQueryEngine(df=excel_df, verbose=True)
            tools.append(
                QueryEngineTool(
                    query_engine=excel_query_engine,
                    metadata=ToolMetadata(
                        name="excel_data",
                        description="This provides information from the Excel file.",
                    ),
                )
            )
   
    elif file_type == "pdf":
        pdf_data = load_pdf()
        if pdf_data is not None:
            print(pdf_data)
            pdf_index = get_pdf_index(pdf_data, "pdf_data")
            pdf_engine = pdf_index.as_query_engine()
            tools.append(
                QueryEngineTool(
                    query_engine=pdf_engine,
                    metadata=ToolMetadata(
                        name="pdf_data",
                        description="This provides information from the PDF file.",
                    ),
                )
            )
   
    elif file_type == "youtube":
        video_url = input("Please enter the YouTube video URL: ").strip()
        save_path = "data"  # Folder to save the audio and transcript
        transcription_path = download_and_transcribe_youtube(video_url, save_path)
        youtube_index = get_youtube_index(transcription_path, "youtube_video")
        youtube_engine = youtube_index.as_query_engine()
        tools.append(
            QueryEngineTool(
                query_engine=youtube_engine,
                metadata=ToolMetadata(
                    name="youtube_data",
                    description="This provides information from the transcribed YouTube video.",
                ),
            )
        )
   
    else:
        print("Invalid file type selected. Please choose CSV, Excel, PDF, or YouTube.")
        return

    # Initialize the LLM (Language Model)
    llm = Groq(model="llama3-70b-8192")

    # Initialize the ReActAgent with all tools
    agent = ReActAgent.from_tools(tools, llm=llm, verbose=True)

    # Run the query loop
    while (prompt := input("Enter a prompt (q to quit): ")) != "q":
        result = agent.query(prompt)
        print(result)

# Run the main function
if __name__ == "__main__":
    main()