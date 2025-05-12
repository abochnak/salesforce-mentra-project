#!/usr/bin/env python3

import os
import arrow
import logging
import json
import time
import asyncio
import gradio as gr
from typing import Optional
from google import genai
from google.genai import types
from pydantic import BaseModel

zip_codes = {
    "30912": "Augusta, GA",
    "30673": "Washington, GA",
    "30030": "Decatur, GA",
    "30342": "Atlanta, GA",
    "30742": "Fort Oglethorpe, GA",
    "30701": "Calhoun, GA",
    "30303": "Atlanta, GA",
    "39828": "Cairo, GA",
    "31021": "Dublin, GA",
    "31643": "Quitman, GA",
    "30474": "Vidalia, GA",
    "31093": "Warner Robins, GA",
    "31069": "Perry, GA",
    "30329": "Atlanta, GA",
    "30165": "Rome, GA",
    "31730": "Camilla, GA",
    "31768": "Moultrie, GA",
    "30046": "Lawrenceville, GA",
    "30909": "Augusta, GA",
    "31792": "Thomasville, GA",
    "31901": "Columbus, GA",
    "30097": "Duluth, GA",
    "30901": "Augusta, GA",
    "30705": "Chatsworth, GA",
    "31501": "Waycross, GA",
    "30125": "Cedartown, GA",
    "30635": "Elberton, GA",
    "30458": "Statesboro, GA",
    "30720": "Dalton, GA",
    "30322": "Atlanta, GA",
    "31029": "Forsyth, GA",
    "31404": "Savannah, GA",
    "30058": "Lithonia, GA",
    "30033": "Decatur, GA",
    "31024": "Eatonton, GA",
    "30308": "Atlanta, GA"
}

class ProcedureInsuranceLocationSchema(BaseModel):
    health_procedure_name: Optional[str]
    health_insurance_name: Optional[str]
    location_for_procedure: Optional[str]

    def missing_items(self):
        missing = []
        if not self.health_procedure_name:
            missing.append("health procedure name")
        if not self.health_insurance_name:
            missing.append("health insurance name")
        if not self.location_for_procedure:
            missing.append("location for procedure")
        return missing

geminiClient = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL = "gemini-2.0-flash"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger(__name__)


with gr.Blocks() as demo:
    output = gr.Markdown(render=False)
    procedure = gr.State(None)
    insurance = gr.State(None)
    location = gr.State(None)
    confirmed = gr.State(None)

    def build_output():
        return gr.Markdown(f"""
# Your selection:

{f"**Procedure:** {procedure.value}" if procedure.value else ""}

{f"**Insurance:** {insurance.value}" if insurance.value else ""}

{f"**Location:** {location.value}" if location.value else ""}
""")


    def respond(message, history):
        response = None
        _yes = "Looks good!"
        _no = "That's not right."
        if not history or not procedure.value or not insurance.value or not location.value or confirmed.value is False:
            # Initial message for getting the fields
            yield "Using Gemini...", build_output()
            gemini_response = geminiClient.models.generate_content(
                model=MODEL,
                config=types.GenerateContentConfig(
                    system_instruction="You are HealthPT: GPT-powered Healthcare Cost Analysis. Given a health procedure, insurance provider, and location, HealthPT performs an estimated cost analysis from its dataset. You need to perform the first step which is just processing the user's input.",
                    response_mime_type="application/json",
                    response_schema=ProcedureInsuranceLocationSchema
                ),
                contents=message,
            )
            print(f"Gemini response: {gemini_response.text}")
            gemini_response: ProcedureInsuranceLocationSchema = gemini_response.parsed
            if gemini_response.health_procedure_name:
                procedure.value = gemini_response.health_procedure_name
            if gemini_response.health_insurance_name:
                insurance.value = gemini_response.health_insurance_name
            if gemini_response.location_for_procedure:
                location.value = gemini_response.location_for_procedure
            missing = gemini_response.missing_items()
            if missing:
                response = f"I didn't quite get all of that. I need information for {" and ".join(missing)}. Can you provide those again?"
            else:
                response = gr.ChatMessage(
                    content = f"Got it! You are looking for a '{procedure.value}' procedure located within '{location.value}' using '{insurance.value}' insurance. Is that right?",
                    options = [
                        {"value": _yes},
                        {"value": _no},
                    ]
                )
                confirmed.value = None
        
        elif confirmed.value is None:
            if history[-1]["content"] == _yes:
                confirmed.value = True
                response = "Great!"

            elif history[-1]["content"] == _no:
                confirmed.value = False
                response = "OK, let's try again... what needs to be corrected?"
        yield response, build_output()
        



    gr.Markdown("""
# HealthPT: GPT-powered Healthcare Cost Analysis
Given a health procedure, insurance provider, and location, HealthPT provides a cost estimate from our database of 1,000,000+ procedures in the state of Georgia.
    """)
    with gr.Row():
        with gr.Column():
            gr.ChatInterface(
                respond,
                type="messages",
                textbox=gr.Textbox(placeholder="What health procedure are you seeking information for? Make sure to include your location and name of your insurance provider."),
                examples=[
                    "I need heart surgery in Atlanta with Cigna insurance.",
                    "How expensive is a flu shot in Augusta with BlueCross BlueShield insurance?"
                ],
                additional_outputs=[output],
            )
        with gr.Column():
            output.render()

demo.launch(server_port=7860)