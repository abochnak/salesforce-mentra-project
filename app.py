#!/usr/bin/env python3

import os
import arrow
import logging
import json
import requests
import time
import asyncio
import collections
import gradio as gr
from typing import Optional
from google import genai
from google.genai import types
from pydantic import BaseModel
from utils.metadata import zip_to_city, insurance_names, procedure_summaries_map, procedure_primary_summary, substr_list_filter, substr_map_filter

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

BACKEND_ENDPOINT = os.getenv("BACKEND_ENDPOINT")

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
    result = gr.State(None)

    def build_output(no_markdown=False):
        #  ({substr_map_filter(procedure_summaries_map, procedure.value)})
        #  ({substr_list_filter(insurance_names, insurance.value)})
        #  ({substr_map_filter(zip_to_city, location.value)})
        _ret = f"""
# Your selection:

{" | ".join(filter(None, [
    f"**Procedure:** {procedure.value}" if procedure.value else None, 
    f"**Insurance:** {insurance.value}" if insurance.value else None,
    f"**Location:** {location.value}" if location.value else None
]))}

{_result_md_output() if result.value is not None and confirmed.value else ""}
"""
        return _ret if no_markdown else gr.Markdown(_ret)
    
    def _result_md_output():
        ret = f"# Results:\n\n"
        _sorted_items = sorted(result.value.items(), key=lambda x: -1*(x[1]["hospital_overall_rating"] or 0))
        for hosp_name, r in _sorted_items:
            ret += f"""
## {hosp_name}
**Location:** {r["street_address"]}, {zip_to_city.get(r["zip_code"])}, GA {r["zip_code"]}

**Hospital Rating:** {r["hospital_overall_rating"]}
            """
            for proc, proc_prices in r["prices"].items():
                ret += f"""
**Procedure {proc}**: {proc_prices[0] if len(proc_prices) == 1 else f"{proc_prices[0]} - {proc_prices[-1]}"}
                """

        return ret
        

    def fetch_dataset_rows():
        _procedures = substr_map_filter(procedure_summaries_map, procedure.value)
        _procedure_filter = f"service={_procedures[0]}" if len(_procedures) == 1 else ""
        _insurances = substr_list_filter(insurance_names, insurance.value)
        _insurance_filter = f"plan_raw={_insurances[0]}" if len(_insurances) == 1 else ""
        _locations = [str(i) for i in substr_map_filter(zip_to_city, location.value)]
        _zip_filter = f"zip_code={_locations[0]}" if len(_locations) == 1 else ""
        _filter = "&".join(filter(None, [_procedure_filter, _insurance_filter, _zip_filter]))
        logger.info(f"Querying: {BACKEND_ENDPOINT}?{_filter}")
        full_data = requests.get(f"{BACKEND_ENDPOINT}?{_filter}").json()
        ret = []
        for item in full_data.get("data"):
            if not item["service"] in _procedures:
                continue
            if not item["plan_raw"] in _insurances:
                continue
            if not str(item["zip_code"]) in _locations:
                continue
            ret.append(item)
        _deduped = _dedup_rows(ret)
        logger.info(f"Query for {_filter} returned {len(ret)} rows, {len(_deduped)} after dedup: {_deduped}")
        return _deduped

    def _dedup_rows(pre_result):
        prov_to_rows = collections.defaultdict(list)
        ret = {}
        for item in pre_result:
            if item["provider"] not in ret:
                ret[item["provider"]] = {
                    "zip_code": item["zip_code"],
                    "hospital_overall_rating": item["hospital_overall_rating"],
                    "street_address": item["street_address"]
                }
            prov_to_rows[item["provider"]].append(item)
        for prov in prov_to_rows.keys():
            _prices = collections.defaultdict(list)
            for i in prov_to_rows[prov]:
                r = i["rate"]
                s = procedure_primary_summary.get(i["service"], i["service"])
                if " - " in r:
                    l, h = r.split(" - ", 1)
                    if l == h:
                        _prices[s].append(l)
                        continue
                    else:
                        _prices[s].append(l)
                        _prices[s].append(h)
                        continue
                _prices[s].append(r)
            
            _s_prices = {}
            for k,v in _prices.items():
                _s_prices[k] = list(sorted(_prices[k], key=lambda x: float(x[1:])))
                

            ret[prov]["prices"] = _s_prices
        return ret


    def history_to_gemini_fmt(gr_history, message=None, result_md=None):
        gem_history = []
        for item in gr_history:
            if item.get("content"):
                if item.get("role") == "user":
                    gem_history.append({"role": "user", "parts": [{"text": item["content"]}]})
                    continue
                elif item.get("role") == "assistant":
                    gem_history.append({"role": "model", "parts": [{"text": item["content"]}]})
        if message:
            gem_history.append({"role": "user", "parts": [{"text": message}]})
        if result_md:
            gem_history.append({"role": "model", "parts": [{"text": result_md}]})
        return gem_history

    def respond(message, history):
        logger.info(f"respond(message={message}, history={history})")
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
            no_match = []
            gemini_response: ProcedureInsuranceLocationSchema = gemini_response.parsed
            if gemini_response.health_procedure_name:
                procedure.value = gemini_response.health_procedure_name
                if not substr_map_filter(procedure_summaries_map, procedure.value):
                    no_match.append(f"health procedure '{procedure.value}'")
                    procedure.value = None
            if gemini_response.health_insurance_name:
                insurance.value = gemini_response.health_insurance_name
                if not substr_list_filter(insurance_names, insurance.value):
                    no_match.append(f"health insurance '{insurance.value}")
                    insurance.value = None
            if gemini_response.location_for_procedure:
                location.value = gemini_response.location_for_procedure
                if not substr_map_filter(zip_to_city, location.value):
                    no_match.append(f"location '{location.value}")
                    location.value = None

            missing = gemini_response.missing_items()
            if missing:
                response = f"I didn't quite get all of that. I need information for {" and ".join(missing)}. Can you provide those again?"
            elif no_match:
                response = f"Sorry, I wasn't able to find matches in the database for {" and ".join(no_match)}. Can you provide those again? Note that the dataset only includes data for a limited number of procedures within the state of Georgia. Can you try something else?"
            else:
                yield "Checking the dataset...", build_output()
                _fetched = fetch_dataset_rows()
                if len(_fetched) == 0:
                    response = f"Sorry, but while I understand you're looking for a '{procedure.value}' procedure located within '{location.value}' using '{insurance.value}' insurance, I wasn't able to find matches in the database with this criteria. The dataset only includes data for a limited number of procedures within the state of Georgia. Can you try something else?"
                else:
                    result.value = _fetched
                    response = gr.ChatMessage(
                        content = f"Got it! You are looking for a '{procedure.value}' procedure located within '{location.value}' using '{insurance.value}' insurance. I found price details for {len(_fetched)} medical providers. Is this what you were looking for?",
                        options = [
                            {"value": _yes},
                            {"value": _no},
                        ]
                    )
                    confirmed.value = None
        
        elif confirmed.value is None:
            if message == _yes:
                confirmed.value = True
                yield "Great! I'm summarizing this with Gemini...", build_output()
                _gemini_input = f"Summarize the following. Do NOT use bullet points. Write a single paragraph. My initial question was: {history_to_gemini_fmt(history)[0].get("parts")[0].get("text")}\n\n{build_output(no_markdown=True)}"
                logger.info(f"Gemini input: {_gemini_input}")
                #A summary of what I found is on the right. Do you have any questions about this I can help answer?"
                gemini_response = geminiClient.models.generate_content(
                    model=MODEL,
                    config=types.GenerateContentConfig(
                        system_instruction="You are HealthPT: GPT-powered Healthcare Cost Analysis. Given a health procedure, insurance provider, and location, HealthPT analyzes the cost information from the provided data in a conversational interface, allowing the user to ask follow-up questions. Answer the user's question directly."
                    ),
                    contents=_gemini_input
                )
                logger.info(f"Gemini response: {gemini_response}")
                response = gemini_response.text

            elif message == _no:
                confirmed.value = False
                response = "OK, let's try again... what needs to be corrected?"
        else:
            yield "Asking Gemini...", build_output()
            _gemini_input = history_to_gemini_fmt(history, message=f"{message}: {build_output(no_markdown=True)}")[-6:]
            logger.info(f"Gemini followup input: {_gemini_input}")
            was_changed = [False]
            def change_search_parameters(health_procedure_name: Optional[str], health_insurance_name: Optional[str], location_for_procedure: Optional[str]):
                was_changed[0] = True
                
            gemini_response = geminiClient.models.generate_content(
                model=MODEL,
                config=types.GenerateContentConfig(
                    system_instruction="You are HealthPT: GPT-powered Healthcare Cost Analysis. Given a health procedure, insurance provider, and location, HealthPT analyzes the cost information from the provided data in a conversational interface, allowing the user to ask follow-up questions. Answer the user's question directly.",
                    tools=[change_parameters]
                ),
                contents=_gemini_input
            )
            logger.info(f"Gemini followup response: {gemini_response}")
            response = gemini_response.text
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
                textbox=gr.Textbox(placeholder="Enter a query about healthcare costs..."),
                examples=[
                    "I need an EKG in Atlanta with Cigna insurance.",
                    "How expensive is an ER visit in Augusta with WellCare insurance?"
                ],
                additional_outputs=[output],
            )
        with gr.Column():
            output.render()

demo.launch(server_port=7860)