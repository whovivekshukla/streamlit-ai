import streamlit as st
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import io
import wave
import os
from openai import OpenAI
import tempfile
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_openai import ChatOpenAI
import base64
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# Initialize Whisper model
@st.cache_resource
def load_whisper_model():
    return WhisperModel("tiny", device="cpu", compute_type="int8")

model = load_whisper_model()

# Initialize OpenAI client
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

# Initialize Langchain components
llm = ChatOpenAI(temperature=0.7)
memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# Initialize message history store
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


with_message_history = RunnableWithMessageHistory(conversation, get_session_history)

def main():
    st.title("AI Assistant with Speech Interface")

    # Initialize session state
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

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
                <audio autoplay="true">
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                </audio>
                """
            st.markdown(md, unsafe_allow_html=True)

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
                    ai_response = with_message_history.invoke(
                        [HumanMessage(content=transcription)],
                        config={"configurable": {"session_id": "default"}}
                    )
                
                # Display AI response
                with st.chat_message("ai"):
                    st.write(ai_response['response'])
                
                # Update conversation history
                st.session_state.conversation_history.append({"role": "user", "content": transcription})
                st.session_state.conversation_history.append({"role": "ai", "content": ai_response['response']})
                
                # Convert AI response to speech and auto-play
                with st.spinner("Converting AI response to speech..."):
                    audio_file = text_to_speech(ai_response['response'])
                    if audio_file:
                        autoplay_audio(audio_file)
                        os.unlink(audio_file)  # Delete the temporary file after playing
            except Exception as e:
                st.error(f"Error: {str(e)}")

    # Button to clear the conversation history
    if st.button("Clear Conversation"):
        st.session_state.conversation_history = []
        memory.clear()

if __name__ == "__main__":
    main()