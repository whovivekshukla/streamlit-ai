import streamlit as st
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import io
import wave
import os
from openai import OpenAI
import tempfile
import base64
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import UpstashRedisChatMessageHistory
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import UpstashRedisChatMessageHistory
from langchain.agents import AgentExecutor, create_tool_calling_agent

import faiss

from langchain_community.docstore import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain.memory import VectorStoreRetrieverMemory
from langchain_openai import OpenAIEmbeddings


from utils.tools import book_appointment

from langchain.tools.retriever import create_retriever_tool



URL = "https://popular-hawk-40003.upstash.io"
TOKEN = "AZxDAAIjcDE2ZGQ4YmQ0N2Q5NTQ0YjcwOWNmZjcyN2FjNjBjMGU1ZXAxMA"

# Initialize Whisper model
@st.cache_resource
def load_whisper_model():
    return WhisperModel("tiny", device="cpu", compute_type="int8")

model = load_whisper_model()

# Initialize OpenAI client
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

# Initialize Langchain components
llm = ChatOpenAI(model="gpt-4o")


def get_session_history(session_id: str):
    return UpstashRedisChatMessageHistory(
        url=URL,
        token=TOKEN,
        ttl=3600,  # Set time-to-live in seconds (e.g., 1 hour)
        session_id=session_id
    )

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are Azodha, an intelligent virtual assistant for a healthcare appointment booking application. Your primary functions are to:

1. Guide patients towards booking appointments with appropriate healthcare professionals rather than attempting to answer medical questions directly.
2. Assist with the appointment booking process by providing available time slots and helping patients schedule with the right type of healthcare provider.
3. Offer general information about the healthcare facility, its services, and policies.
4. Provide reminders for upcoming appointments.
5. Help patients navigate the app and access basic, non-medical information.

Key points to remember:
- Always maintain a professional, empathetic, and patient-friendly tone.
- Prioritize patient privacy and confidentiality.
- Do not attempt to diagnose conditions, answer specific medical questions, or provide medical advice.
- For any medical concerns or questions, consistently recommend booking an appointment with an appropriate healthcare professional.
- For emergencies, always direct patients to call emergency services immediately.
- Each response should be at max 50 words, but try to to not hit it.

Your role is to facilitate connections between patients and healthcare providers, not to serve as a source of medical information."""),
   ("placeholder", "{chat_history}"),
    ("human", "{input}"),
     ("placeholder", "{agent_scratchpad}"),
])

chain = prompt | llm

embedding_size = 1536  # Dimensions of the OpenAIEmbeddings
index = faiss.IndexFlatL2(embedding_size)
embedding_fn = OpenAIEmbeddings().embed_query
vectorstore = FAISS(embedding_fn, index, InMemoryDocstore({}), {})


retriever = vectorstore.as_retriever(search_kwargs=dict(k=1))
# memory = VectorStoreRetrieverMemory(retriever=retriever)

tool = create_retriever_tool(
    retriever,
    "convert_response_to_vector",
    "this tool is used when you call another tools but the context token for them is a lot, so first you convert it to vector and then use it!.",
)

# Initialize tools
tools = [book_appointment, tool]



# Create tool calling agent
agent = create_tool_calling_agent(llm, tools, prompt)

def get_llm_response(user_input, session_id):
    history = get_session_history(session_id)
    
    # Add user message to history
    history.add_user_message(user_input)
    
    # Prepare the full conversation history
    chat_history = history.messages
    
    # Generate LLM response
    response = prompt.format_prompt(
        chat_history=chat_history,
        input=user_input
    )
    ai_message = llm.invoke(response.to_string())
    
    # Add AI message to history
    history.add_ai_message(ai_message)
    
    return ai_message


def get_agent_response(user_input, session_id):
    history = get_session_history(session_id)
    
    # Add user message to history
    history.add_user_message(user_input)
    
    # Prepare the full conversation history
    chat_history = history.messages
    
    # Create an agent executor by passing in the agent and tools
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    result = agent_executor.invoke({"input": user_input, "chat_history": chat_history})
    
    # Add AI message to history
    history.add_ai_message(result['output'])
    
    return result['output']

def main():
    st.title("Azodha AI Assistant")

    # Initialize session state
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    # Fetch conversation history from Redis if available
    session_id = "112"  # Example session ID, you might want to generate or fetch this dynamically
    history = get_session_history(session_id)
    chat_history = history.messages

    # Update session state with fetched history
    if chat_history:
        st.session_state.conversation_history = [{"role": msg.type, "content": msg.content} for msg in chat_history]

    # Audio recording parameters
    duration = 5  # Duration of each recording chunk in seconds
    fs = 16000  # Sample rate
    channels = 1  # Number of audio channels

    # Function to record audio
    def record_audio():
        audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=channels)
        sd.wait()
        return audio_data

    # Function to convert numpy array to WAV bytes
    def numpy_to_wav_bytes(audio_np):
        byte_io = io.BytesIO()
        with wave.open(byte_io, 'wb') as wave_file:
            wave_file.setnchannels(channels)
            wave_file.setsampwidth(2)  # 2 bytes per sample for int16
            wave_file.setframerate(fs)
            wave_file.writeframes((audio_np * 32767).astype(np.int16).tobytes())
        byte_io.seek(0)  # Reset the BytesIO object to the beginning
        return byte_io

    # Function to transcribe audio
    def transcribe_audio(audio_data):
        audio_bytes_io = numpy_to_wav_bytes(audio_data)
        segments, _ = model.transcribe(audio_bytes_io, language="en")
        transcription = " ".join([segment.text for segment in segments])
        return transcription

    # Function to convert text to speech using OpenAI's TTS
    def text_to_speech(text):
        try:
            response = client.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=text
            )
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
                temp_filename = temp_audio.name
                response.stream_to_file(temp_filename)
            
            return temp_filename
        except Exception as e:
            st.error(f"Error in TTS conversion: {str(e)}")
            return None

    # Function to auto-play audio
    def autoplay_audio(file_path: str):
        with open(file_path, "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()
            md = f"""
                <audio autoplay="true" id="audio-player">
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                </audio>
                """
            st.markdown(md, unsafe_allow_html=True)

    # Function to stop audio
    def stop_audio():
        st.markdown("""
            <script>
            var audio = document.getElementById('audio-player');
            if (audio) {
                audio.pause();
                audio.currentTime = 0;
            }
            </script>
            """, unsafe_allow_html=True)

    # Create a chat-like UI
    for message in st.session_state.conversation_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Create a button to start recording
    if st.button("Start Recording"):
        with st.spinner("Recording... Speak now!"):
            audio_data = record_audio()
        
        with st.spinner("Recording finished. Transcribing..."):
            try:
                transcription = transcribe_audio(audio_data)
                
                # Display user message
                with st.chat_message("user"):
                    st.write(transcription)
                
                # Get AI response
                with st.spinner("AI is thinking..."):
                    ai_response = get_agent_response(transcription, session_id)
                    
                    print(ai_response)
                
                # Display AI response
                with st.chat_message("ai"):
                    st.write(ai_response)
                
                # Update conversation history
                st.session_state.conversation_history.append({"role": "user", "content": transcription})
                st.session_state.conversation_history.append({"role": "ai", "content": ai_response})
                
                # Convert AI response to speech and auto-play
                with st.spinner("Converting AI response to speech..."):
                    audio_file = text_to_speech(ai_response)
                    if audio_file:
                        autoplay_audio(audio_file)
                        if st.button("Stop Playing"):
                            stop_audio()
                        os.unlink(audio_file)  # Delete the temporary file after playing
            except Exception as e:
                st.error(f"Error: {str(e)}")

    # Button to clear the conversation history
    if st.button("Clear Conversation"):
        st.session_state.conversation_history = []
        history.clear()  # Clear the history in Redis as well

if __name__ == "__main__":
    main()