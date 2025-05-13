#!/usr/bin/env python3

import os
import arrow
import logging
import math
import json
import requests
import time
import asyncio
import collections
import gradio as gr
import urllib.parse
from typing import List, Optional, Union
from google import genai
from google.genai import types
from pydantic import BaseModel
from utils.metadata import zip_to_city, zip_to_latlng, insurance_names, procedure_summaries_map, procedure_primary_summary, substr_list_filter, substr_map_filter

class TrueFalseOrNone(BaseModel):
    response: Optional[bool]

class ProcedureInsuranceLocationSchema(BaseModel):
    '''
    One or more of health_procedure_name, health_insurance_name, or location_for_procedure.
    '''
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
        

    def raw_fetch_dataset(procedure_val, insurance_val, location_val):
        _procedures = substr_map_filter(procedure_summaries_map, procedure_val)
        _procedure_filter = f"service={_procedures[0]}" if _procedures and len(_procedures) == 1 else ""
        _insurances = substr_list_filter(insurance_names, insurance_val)
        _insurance_filter = f"plan_raw={_insurances[0]}" if _insurances and len(_insurances) == 1 else ""
        _locations = [str(i) for i in substr_map_filter(zip_to_city, location_val)]
        _zip_filter = f"zip_code={_locations[0]}" if _locations and len(_locations) == 1 else ""
        _filter = "&".join(filter(None, [_procedure_filter, _insurance_filter, _zip_filter]))
        logger.info(f"Querying: {BACKEND_ENDPOINT}?{_filter}")
        full_data = requests.get(f"{BACKEND_ENDPOINT}?{_filter}").json()
        ret = []
        for item in full_data.get("data"):
            if _procedures:
                if not item["service"] in _procedures:
                    continue
            if _insurances:
                if not item["plan_raw"] in _insurances:
                    continue
            if _locations:
                if not str(item["zip_code"]) in _locations:
                    continue
            ret.append(item)
        _deduped = _dedup_rows(ret)
        logger.info(f"Query for {_filter} returned {len(ret)} rows, {len(_deduped)} after dedup: {_deduped}")
        return _deduped

    def fetch_dataset_rows():
        return raw_fetch_dataset(procedure.value, insurance.value, location.value)
    
    def calculate_distance(latlng_a, latlng_b):
        def _calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
            # Haversine formula, returning miles
            R_km = 6371.0
            dlat = math.radians(lat2 - lat1)
            dlon = math.radians(lon2 - lon1)

            a = (math.sin(dlat / 2) ** 2 +
                math.cos(math.radians(lat1)) *
                math.cos(math.radians(lat2)) *
                math.sin(dlon / 2) ** 2)

            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            distance_km = R_km * c

            return distance_km * 0.6213712
        return _calculate_distance(latlng_a["latitude"], latlng_a["longitude"], latlng_b["latitude"], latlng_b["longitude"])

    def closest_other_locations_to(location_name: str) -> List[str]:
        loc_latlng = None
        for z, c in zip_to_city.items():
            if c.lower() == location_name.lower():
                loc_latlng = zip_to_latlng.get(z)
        if not loc_latlng:
            return []
        dists = []
        for z, c in zip_to_city.items():
            if c.lower() == location_name.lower():
                continue
            cur_latlng = zip_to_latlng.get(z)
            _dist = calculate_distance(loc_latlng, cur_latlng)
            _skip = None
            for oo in dists:
                if oo[1].lower() == c.lower():
                    if oo[0] <= _dist:
                        _skip = True
                    else:
                        del oo
            
            if _skip:
                continue
                    
            dists.append((_dist, c))
        dists.sort()
        return [i[1] for i in dists[:3]]
        
    def all_locations_in_dataset() -> List[str]:
        return list(set(zip_to_city.values()))

    def get_travel_directions_to(street_address: str) -> str:
        """Returns travel directions to the given street address for a location."""
        return f"[Get travel directions to {street_address}](https://www.google.com/maps/place/{urllib.parse.quote(street_address)})"

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
        if result_md:
            gem_history.append({"role": "model", "parts": [{"text": result_md}]})
        if message:
            gem_history.append({"role": "user", "parts": [{"text": message}]})
        return gem_history
    
    def gemini_yn_intent(message):
        gemini_response = geminiClient.models.generate_content(
            model=MODEL,
            config=types.GenerateContentConfig(
                system_instruction="Interpret the user's response as one of: True, False, or None.",
                response_mime_type="application/json",
                response_schema=TrueFalseOrNone
            ),
            contents=message,
        )

        p = gemini_response.parsed
        if p:
            return p.response
        return None

    def respond(message, history):
        logger.info(f"respond(message={message}, history={history})")
        response = None
        _yes = "Looks good!"
        _no = "That's not right."
    
        def process_procedure_request(gemini_response):
            no_match = []
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
            return response

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
            response = yield from process_procedure_request(gemini_response)
        
        elif confirmed.value is None:
            if message == _yes or (message != _no and gemini_yn_intent(message) is True):
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
                response = gr.ChatMessage(
                    content = response,
                    options = [
                        {"value": "What other options are nearby?"},
                        {"value": "Give travel directions to the cheapest option."}
                    ]
                )

            elif message == _no or gemini_yn_intent(message) is False:
                confirmed.value = False
                if message != _no:
                    response = "OK, let's try again... what needs to be corrected?"
                else:
                    response = "OK, let's try again... what needs to be corrected?"
        else:
            yield "Asking Gemini...", build_output()
            _gemini_input = history_to_gemini_fmt(history, message=f"{message}", result_md=build_output(no_markdown=True))[-6:]
            logger.info(f"Gemini followup input: {_gemini_input}")
            was_changed = [None]
            def change_search_parameters(change_schema: ProcedureInsuranceLocationSchema):
                logger.info(f"Gemini change search parameters: {change_schema}")

                was_changed[0] = change_schema

                return 'done!'
            gemini_response = None
            while gemini_response is None:
                def change_procedure(procedure: str):
                    return change_search_parameters(ProcedureInsuranceLocationSchema(health_procedure_name=procedure, health_insurance_name=insurance.value, location_for_procedure=location.value))
                def change_insurance(insurance: str):
                    return change_search_parameters(ProcedureInsuranceLocationSchema(health_insurance_name=insurance, health_procedure_name=procedure.value, location_for_procedure=location.value))
                def change_location(location: str):
                    return change_search_parameters(ProcedureInsuranceLocationSchema(location_for_procedure=location, health_procedure_name=procedure.value, health_insurance_name=insurance.value))
                gemini_response = geminiClient.models.generate_content(
                    model=MODEL,
                    config=types.GenerateContentConfig(
                        system_instruction="""You are HealthPT: GPT-powered Healthcare Cost Analysis. Given a health procedure, insurance provider, and location, HealthPT analyzes the cost information from the provided data in a conversational interface, allowing the user to ask follow-up questions.

                        Call 'change_procedure' to change the health procedure, 'change_insurance' to change insurance, or 'change_location' to change location.
                        Call 'closest_other_locations_to' to identify other nearby locations and do so without prompting. 
                        Call 'get_travel_directions_to' to get travel directions which should then be shared with the user.
                        
                        ALWAYS continue the conversation after a function call, and call functions without asking for confirmation.
                        """,
                        tools=[change_procedure, change_insurance, change_location, all_locations_in_dataset, closest_other_locations_to, get_travel_directions_to],
                        
                    ),
                    contents=_gemini_input
                )
                logger.info(f"Gemini followup response: {gemini_response.text=} {gemini_response=}")
                if was_changed[0]:
                    response = yield from process_procedure_request(was_changed[0])
                else:
                    response = gemini_response.text
                
                break
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