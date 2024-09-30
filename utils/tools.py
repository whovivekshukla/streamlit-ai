# Standard library imports
import os
import json
from pathlib import Path

# Load environment variables from .env file

# Third-party imports

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain import hub
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool
import requests

UPSTASH_VECTOR_REST_URL = os.getenv("UPSTASH_VECTOR_REST_URL")
UPSTASH_VECTOR_REST_TOKEN = os.getenv("UPSTASH_VECTOR_REST_TOKEN")
REDIS_URL = os.getenv("REDIS_URL")

embeddings = OpenAIEmbeddings()

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    return RedisChatMessageHistory(session_id, url=REDIS_URL)


class BookAppointmentInput(BaseModel):
    patient_name: str = Field(description="The name of the patient booking the appointment")
    doctor_name: str = Field(description="The name of the doctor with whom the appointment is being booked")
    appointment_time: str = Field(description="The time of the appointment in ISO format")
    appointment_type_id: int = Field(description="The type ID of the appointment")
    start_time: str = Field(description="The start time of the appointment")
    end_time: str = Field(description="The end time of the appointment")
    duration_in_minutes: int = Field(description="The duration of the appointment in minutes")
    # related_patient_id: str = Field(description="The ID of the related patient")
    note: str = Field(description="A note for the appointment")
    event_title: str = Field(description="The title of the event")
    status: str = Field(description="The status of the appointment")
    booking_category: str = Field(description="The booking category of the appointment")
    override_slots: bool = Field(description="Whether to override slots")

@tool("book_appointment", args_schema=BookAppointmentInput)
def book_appointment(patient_name: str, doctor_name: str, appointment_time: str, appointment_type_id: int, start_time: str, end_time: str, duration_in_minutes: int, note: str, event_title: str, status: str, booking_category: str, override_slots: bool) -> str:
    """Make a POST request to book an appointment.
    
    Example payload:
    {
      "appointmentTypeId": 1145,
      "date": "2024-08-13T00:00:00.000Z",
      "startTime": "03:30:00",
      "endTime": "04:00:00",
      "overrideSlots": false,
      "durationInMinutes": 30,
      "relatedPatients": [
        {
          "id": "323jjje2je333"
        }
      ],
      "note": "test",
      "attributes": {
        "eventTitle": "test",
        "status": "SCHEDULED",
        "bookingCategory": "Event",
        "overrideSlots": false
      }
    }
    """

    url = "https://platform-api-development.azo.dev/api/service-provider-scheduling/appointments"
    payload = {
        "appointmentTypeId": 3720,
        "date": appointment_time,
        "startTime": start_time,
        "endTime": end_time,
        "overrideSlots": True,
        "durationInMinutes": duration_in_minutes,
         "patientId": "e9ba687a25ae4c3cb7dd",
        "attributes": {
            "eventTitle": event_title,
            "status": status,
            "bookingCategory": booking_category,
            "overrideSlots": override_slots
        }
    }
    headers = {
        'authorization': 'Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6InFWMDZzQzkzVS1UNWRIdXhRSWY1TyJ9.eyJuaWNrbmFtZSI6Im1hbmlzaGJhc2FyZ2VrYXIrc3BwIiwibmFtZSI6Im1hbmlzaGJhc2FyZ2VrYXIrc3BwQGF6b2RoYS5jb20iLCJwaWN0dXJlIjoiaHR0cHM6Ly9zLmdyYXZhdGFyLmNvbS9hdmF0YXIvMjhlOWRiYzZjNjNjNzRlOWRlMDA3NjlhZjkxMWY2OTE_cz00ODAmcj1wZyZkPWh0dHBzJTNBJTJGJTJGY2RuLmF1dGgwLmNvbSUyRmF2YXRhcnMlMkZtYS5wbmciLCJ1cGRhdGVkX2F0IjoiMjAyNC0wOS0zMFQxNDoxMTozMy4yNTBaIiwiZW1haWwiOiJtYW5pc2hiYXNhcmdla2FyK3NwcEBhem9kaGEuY29tIiwiZW1haWxfdmVyaWZpZWQiOmZhbHNlLCJpc3MiOiJodHRwczovL2Rldi1jYXJlMnUudXMuYXV0aDAuY29tLyIsImF1ZCI6ImdVNzhLaTNDRmVzaUZHYlR6ZUR6RHpCTDE5MDFobkhWIiwiaWF0IjoxNzI3NzA1NDk2LCJleHAiOjE3Mjc3NDE0OTYsInN1YiI6ImF1dGgwfDY1MzZhZjkyOWFlZDhjZjFiMjM0Njc0OSIsInNpZCI6Imx6RE1HNGt3Ty1EQmdjOWk5REo1VmZ1aTBoSDBvZHVqIiwibm9uY2UiOiJjbGwyYzFWbFlrWlRWbWM1TURSVllrRXlSekpqY0RaelYxWnhjbFpDVGtWc2RGUkxUMHQwWTAwemRnPT0ifQ.GADBoe7ncZJlXctj8dDPCi_CFVs16nmzqSRXCEsyuCO7wu-i1uc5AwJQ205axSn8Lc5DZQxbiVhmVSEXKfH2IMJcm5qbMdPb_9t0knGC4jTzm6x1rpCwIzAv8mikMs5EATKXifq389J-mLkUGKiVYEK8dhVcyHWTZkFqzwxDrf9oY7VDyj1HcT3AlH4JS_6L8AcRkD1Mw_CL_seRKWO6Brb1TEy4ZMTcg3z22FrUfl7L4k6Xt3_iHMoaMNS93M8iVFHo_zsX3urPbClUo2RAN7srjUOYMrQvFMAugHa8KUGR1_eVdjHk7WgY65d1695dij8cypV2TvvUXPvHbjS8oQ',
        'organization_code': 'dev-care2u',
    }
    
    print(f"curl -X POST {url} -H 'authorization: {headers['authorization']}' -H 'organization_code: {headers['organization_code']}' -H 'Content-Type: application/json' -d '{json.dumps(payload)}'")

    
    response = requests.post(url, json=payload, headers=headers)
    print(response.text)
    print(response.headers)
    
    if response.status_code == 200:
        return "Appointment successfully booked."
    else:
        return f"Failed to book appointment. Status code: {response.status_code}, Response: {response.text}"


# class CheckAvailableSlotsInput(BaseModel):
#     from_date: str = Field(description="The start date for checking available slots, example: 2024-08-13")
#     to_date: str = Field(description="The end date for checking available slots, example: 2024-08-16")
#     # appointment_type_template_id: int = Field(description="The template ID of the appointment type")
#     include_booked_slots: bool = Field(description="Whether to include booked slots in the response")

# @tool("check_available_slots", args_schema=CheckAvailableSlotsInput)
# def check_available_slots(from_date: str, to_date: str, include_booked_slots: bool) -> dict:
#     """Check available slots for a given date range and appointment type and then show the user all the options asking which slot they want to book, also ask them for a particular time in the day, so the results are few to choose, example query: fromDate=2024-08-13&toDate=2024-08-15&appointmentTypeTemplateId=705&includeBookedSlots=true"""

#     url = f"https://platform-api-development.azo.dev/api/service-provider-scheduling/available-slots-service-provider-info?fromDate={from_date}&toDate={to_date}&appointmentTypeTemplateId={1309}&includeBookedSlots={str(include_booked_slots).lower()}"
#     headers = {
#         'authorization': 'Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6ImJhYTdyUWZPY2dSQkd0clQ1WTNRUiJ9.eyJuaWNrbmFtZSI6InNodWJoYW0rZXhwYW5kIiwibmFtZSI6InNodWJoYW0rZXhwYW5kQGF6b2RoYS5jb20iLCJwaWN0dXJlIjoiaHR0cHM6Ly9zLmdyYXZhdGFyLmNvbS9hdmF0YXIvOTk5Y2QwMTUyZGI5MTJlYTEyNGY2NDBmMTY4ODc0OWQ_cz00ODAmcj1wZyZkPWh0dHBzJTNBJTJGJTJGY2RuLmF1dGgwLmNvbSUyRmF2YXRhcnMlMkZzaC5wbmciLCJ1cGRhdGVkX2F0IjoiMjAyNC0wOS0zMFQwNDo1MTo0OS4wNzFaIiwiZW1haWwiOiJzaHViaGFtK2V4cGFuZEBhem9kaGEuY29tIiwiZW1haWxfdmVyaWZpZWQiOmZhbHNlLCJpc3MiOiJodHRwczovL3N0YWdpbmctZXhwYW5kLmV1LmF1dGgwLmNvbS8iLCJhdWQiOiJBS0Vja0R2bjh1bXZMMDlzZXhBWnlRY3lGa1FobmJsSiIsImlhdCI6MTcyNzY3MTkxMiwiZXhwIjoxNzI3NzA3OTEyLCJzdWIiOiJhdXRoMHw2NjBhNzJjNzZlYTdiZmY0NTAxNGRiMTciLCJzaWQiOiJhVnBreExQWE80TG5pWW9zQWw2eXkxaldfTHJQdDQ5RCIsIm5vbmNlIjoiZVZvNFRWaGlNVTF2V0c0NFNYUTRXRmwzWDB4SlkwdGlRazl1UkRsV1NVRTRVRTFmTkV0SlVGbEJMZz09In0.O4xmecM7tDmC6tYweVjR_u0930Huqu5Frwkhd7TTu_6rUc_aHBv04ojkHQLosHRT3Rb3d3sNaXKRrMzQBWND2x-QeP6uaBon5R33txVoU4PydSxRibn5w8RLqsac7oNZUKdKXXYM0evBpDePdtd-7mpg-KETgEBx4A-g-K6OEqNou3CRrg9fw4qJ4nUzuXvRzOr4gdQQdwyoUuJhOcR97Y_3T0Mwb_GMRAHDQTuN9xQUjK55aC7VUCV8rp_djgmlQ5MNG-QJ9DdmDbwlPMEl3pNlNC2OzK9OGeDDa9Vfkrp4JxLpR6w5L9dayH254m3qsEVBUdASSTR0pZhFAtfayg',
#         'organization_code': 'dev-expandhealth',
#     }

#     print(url)
#     response = requests.get(url, headers=headers)
    
#     if response.status_code == 200:
#         return response.json()
#     else:
#         return {"error": f"Failed to check available slots. Status code: {response.status_code}, Response: {response.text}"}
