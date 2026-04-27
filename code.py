import os
import sys

WORKSPACE_DIRECTORY = "/workspace"
if os.path.exists(WORKSPACE_DIRECTORY) and WORKSPACE_DIRECTORY not in sys.path:
    sys.path.append(WORKSPACE_DIRECTORY)
    print(f"Added {WORKSPACE_DIRECTORY} to the Python path")

%pip install -q json-repair==0.47.1 numexpr==2.11.0 openai==1.74.0 pandas==2.3.0 pydantic==2.11.7 python-dotenv==1.1.0

from openai import OpenAI

client = OpenAI(
    base_url="https://openai.vocareum.com/v1",
    api_key="voc-3118232301607364820801699237949ce600.39107985",
)

from enum import Enum

class OpenAIModel(str, Enum):
    GPT_41 = "gpt-4.1"  # Strong default choice for development tasks, particularly those requiring speed, responsiveness, and general-purpose reasoning. 
    GPT_41_MINI = "gpt-4.1-mini"  # Fast and affordable, good for brainstorming, drafting, and tasks that don't require the full power of GPT-4.1.
    GPT_41_NANO = "gpt-4.1-nano"  # The fastest and cheapest model, suitable for lightweight tasks, high-frequency usage, and edge computing.

MODEL = OpenAIModel.GPT_41_MINI  # Default model for this project


VACATION_INFO_DICT = {
    "travelers": [
        {
            "name": "Yuri",
            "age": 30,
            "interests": ["tennis", "cooking", "comedy", "technology"],
        },
        {
            "name": "Hiro",
            "age": 25,
            "interests": ["reading", "music", "theatre", "art"],
        },
    ],
    "destination": "AgentsVille",
    "date_of_arrival": "2025-06-10",
    "date_of_departure": "2025-06-12",
    "budget": 130,
}


from project_lib import Interest
from typing import List
from pydantic import BaseModel
import datetime
from pprint import pprint

class Traveler(BaseModel):
    """A traveler with a name, age, and list of interests.
    
    Attributes:
        name (str): The name of the traveler.
        age (int): The age of the traveler.
        interests (List[Interest]): A list of interests of the traveler.
    """
    name: str
    age: int
    interests: List[Interest]

class VacationInfo(BaseModel):
    """Vacation information including travelers, destination, dates, and budget.
    Attributes:
        travelers (List[Traveler]): A list of travelers.
        destination (str): The vacation destination.
        date_of_arrival (datetime.date): The date of arrival.
        date_of_departure (datetime.date): The date of departure.
        budget (int): The budget for the vacation in fictional currency units.
    """
    travelers: List[Traveler]
    destination: str
    date_of_arrival: datetime.date
    date_of_departure: datetime.date
    budget: int


vacation_info = VacationInfo.model_validate(VACATION_INFO_DICT)

pprint(vacation_info.model_dump())

# Check that VacationInfo contains the expected data
assert "travelers" in vacation_info.model_dump().keys(), "VacationInfo should contain 'travelers' key"
assert "destination" in vacation_info.model_dump().keys(), "VacationInfo should contain 'destination' key"
assert "date_of_arrival" in vacation_info.model_dump().keys(), "VacationInfo should contain 'date_of_arrival' key"
assert "date_of_departure" in vacation_info.model_dump().keys(), "VacationInfo should contain 'date_of_departure' key"
assert "budget" in vacation_info.model_dump().keys(), "VacationInfo should contain 'budget' key"
assert isinstance(vacation_info.travelers, list), "Travelers should be a list"
assert all(isinstance(traveler, Traveler) for traveler in vacation_info.travelers), "All travelers should be instances of Traveler class"
assert isinstance(vacation_info.date_of_arrival, datetime.date), "date_of_arrival should be a date"
assert isinstance(vacation_info.date_of_departure, datetime.date), "date_of_departure should be a date"
assert isinstance(vacation_info.budget, int), "budget should be an integer"

print("✅ VacationInfo data structure is valid!")

from project_lib import call_weather_api_mocked
import pandas as pd

pd.set_option("display.max_colwidth", None)  # Show full content in DataFrame cells

weather_for_dates = [
    call_weather_api_mocked(
        date=ts.strftime("%Y-%m-%d"), city=vacation_info.destination
    )
    for ts in pd.date_range(
        start=vacation_info.date_of_arrival,
        end=vacation_info.date_of_departure,
        freq="D",
    )
]

weather_for_dates_df = pd.DataFrame(weather_for_dates)

weather_for_dates_df


from project_lib import call_activities_api_mocked

activities_for_dates = [
    activity
    for ts in pd.date_range(
        start=vacation_info.date_of_arrival,
        end=vacation_info.date_of_departure,
        freq="D",
    )
    for activity in call_activities_api_mocked(
        date=ts.strftime("%Y-%m-%d"), city=vacation_info.destination
    )
]

activities_for_dates_df = pd.DataFrame(activities_for_dates)

activities_for_dates_df

class Weather(BaseModel):
    temperature: float
    temperature_unit: str
    condition: str


class Activity(BaseModel):
    activity_id: str
    name: str
    start_time: datetime.datetime
    end_time: datetime.datetime
    location: str
    description: str
    price: int
    related_interests: List[Interest]


class ActivityRecommendation(BaseModel):
    activity: Activity
    reasons_for_recommendation: List[str]


class ItineraryDay(BaseModel):
    date: datetime.date
    weather: Weather
    activity_recommendations: List[ActivityRecommendation]


class TravelPlan(BaseModel):
    city: str
    start_date: datetime.date
    end_date: datetime.date
    total_cost: int
    itinerary_days: List[ItineraryDay]


import json
from project_lib import ChatAgent
from typing import Optional

ITINERARY_AGENT_SYSTEM_PROMPT = f"""
You are an expert travel planning agent specializing in creating personalized day-by-day vacation itineraries.

## Task

Create a detailed day-by-day itinerary for the travelers based on their vacation details, interests, and constraints.
Follow these steps:
1. Review the traveler profiles, interests, destination, dates, and budget.
2. Check the weather for each day and avoid scheduling outdoor-only activities on rainy days.
3. Select activities that match the interests of the travelers.
4. Ensure the total cost of all activities does not exceed the budget.
5. Include at least one activity per day.
6. Provide reasons for each activity recommendation based on traveler interests.

## Output Format

Respond using two sections (ANALYSIS AND FINAL OUTPUT) in the following format:

    ANALYSIS:
    * Step-by-step reasoning about traveler interests, weather conditions, budget constraints, and activity selection for each day.


    FINAL OUTPUT:

    ```json
    {json.dumps(TravelPlan.model_json_schema(), indent=2)}
    ```

## Context

Vacation Info:
{vacation_info.model_dump_json(indent=2)}

Weather data for the trip:
{json.dumps(weather_for_dates, indent=2, default=str)}

Available activities for the trip:
{json.dumps(activities_for_dates, indent=2, default=str)}
"""  


assert "TASK" in ITINERARY_AGENT_SYSTEM_PROMPT.upper(), "ITINERARY_AGENT_SYSTEM_PROMPT should contain a 'TASK' section"
assert "OUTPUT FORMAT" in ITINERARY_AGENT_SYSTEM_PROMPT.upper(), "ITINERARY_AGENT_SYSTEM_PROMPT should contain a 'OUTPUT FORMAT' section"


class ItineraryAgent(ChatAgent):
    """An agent that plans itineraries based on vacation information, weather, and activities."""
    system_prompt = ITINERARY_AGENT_SYSTEM_PROMPT

    def get_itinerary(self, vacation_info: VacationInfo, model: Optional[OpenAIModel] = None) -> TravelPlan:
        """Generates a travel itinerary based on the provided vacation information."""
        from project_lib import print_in_box
        response = (self.chat(
            user_message=vacation_info.model_dump_json(indent=2),
            add_to_messages=False,
            model=model or self.model,
        ) or "").strip()

        print_in_box(response, "Raw Response")


        json_text = response.strip()

        if "```json" in json_text:
            json_text = json_text.split("```json")[1].split("```")[0].strip()

        try:
            travel_plan = TravelPlan.model_validate_json(json_text)
            return travel_plan
        except Exception as e:
            print("Error validating the following text as TravelPlan JSON:")
            print(json_text)
            raise

itinerary_agent = ItineraryAgent(client=client, model=MODEL)


travel_plan_1 = itinerary_agent.get_itinerary(
    vacation_info=vacation_info,
    model=MODEL,
)

if travel_plan_1 is not None:
    print("✅ Initial itinerary generated successfully. Congratulations!")


class AgentError(Exception):
    pass


class EvaluationResults(BaseModel):
    success: bool
    failures: List[str]
    eval_functions: List[str]


def get_eval_results(vacation_info, final_output, eval_functions) -> EvaluationResults:
    """
    Evaluates the final output of the itinerary agent against a set of evaluation functions.
    Args:
        vacation_info (VacationInfo): The vacation information used to generate the itinerary.
        final_output (TravelPlan): The final output from the itinerary agent.
        eval_functions (List[callable]): A list of evaluation functions to apply.
    Returns:
        EvaluationResults: An object containing the success status, any failures, and the names of the evaluation functions used.
    """
    from project_lib import print_in_box
    if not isinstance(vacation_info, VacationInfo):
        raise ValueError("vacation_info must be an instance of VacationInfo")
    if not isinstance(final_output, TravelPlan):
        raise ValueError("final_output must be an instance of TravelPlan")
    if not isinstance(eval_functions, list) or not all(
        callable(fn) for fn in eval_functions
    ):
        raise ValueError("eval_functions must be a list of callable functions")
    eval_results = []
    for eval_fn in eval_functions:
        try:
            eval_fn(vacation_info, final_output)
        except AgentError as e:
            error_msg = str(e)
            print_in_box(error_msg, title="Evaluation Error")
            print("\n\n")

            eval_results.append(error_msg)
    return EvaluationResults(
        success=len(eval_results) == 0,
        failures=eval_results,
        eval_functions=[fn.__name__ for fn in eval_functions],
    )


def eval_start_end_dates_match(vacation_info: VacationInfo, final_output: TravelPlan):
    """Verifies that the arrival and departure dates in vacation_info match the start and end dates in final_output.

    Args:
        vacation_info (dict): Contains the vacation details including arrival and departure dates
        final_output (dict): Contains the itinerary details including start and end dates

    Raises:
        AgentError: If either the arrival date doesn't match the start date or the departure date doesn't match the end date
    """
    if (
        vacation_info.date_of_arrival != final_output.start_date
        or vacation_info.date_of_departure != final_output.end_date
    ):
        raise AgentError(
            f"Dates do not match: {vacation_info.date_of_arrival} != {final_output.start_date} or {vacation_info.date_of_departure} != {final_output.end_date}"
        )

    if final_output.start_date > final_output.end_date:
        raise AgentError(
            f"Start date is after end date: {final_output.start_date} > {final_output.end_date}"
        )


get_eval_results(
    vacation_info=vacation_info,
    final_output=travel_plan_1,
    eval_functions=[eval_start_end_dates_match],
)


def eval_total_cost_is_accurate(vacation_info: VacationInfo, final_output: TravelPlan):
    """Verifies that the total cost stated in final_output matches the sum of all activity prices.

    Args:
        vacation_info (dict): Contains the vacation details
        final_output (dict): Contains the itinerary details including activities with prices and total cost

    Raises:
        AgentError: If the calculated total cost doesn't match the stated total cost
    """
    actual_total_cost = 0

    for itinerary_day in final_output.itinerary_days:
        for activity_recommendation in itinerary_day.activity_recommendations:
            actual_total_cost += activity_recommendation.activity.price

    stated_total_cost = int(final_output.total_cost)

    if actual_total_cost != stated_total_cost:
        raise AgentError(
            f"Stated total cost does not match calculated total cost: {actual_total_cost} != {stated_total_cost}"
        )
    
def eval_total_cost_is_within_budget(vacation_info: VacationInfo, final_output: TravelPlan):
    """Verifies that the total cost stated in final_output is within the budget specified in vacation_info.

    Args:
        vacation_info (dict): Contains the vacation details including budget
        final_output (dict): Contains the itinerary details including total cost

    Raises:
        AgentError: If the total cost exceeds the budget
    """
    stated_total_cost = int(final_output.total_cost)
    if stated_total_cost > vacation_info.budget:
        raise AgentError(
            f"Total cost exceeds budget: {stated_total_cost} > {vacation_info.budget}"
        )

get_eval_results(
    vacation_info=vacation_info,
    final_output=travel_plan_1,
    eval_functions=[eval_total_cost_is_accurate, eval_total_cost_is_within_budget],
)


def eval_itinerary_events_match_actual_events(
    vacation_info: VacationInfo, final_output: TravelPlan
):
    """Verifies that the events listed in the itinerary match the actual events

    Args:
        vacation_info (dict): Contains the vacation details including traveler information and their interests
        final_output (dict): Contains the itinerary details including daily activities

    Raises:
        AgentError: If any traveler has no matching activities or if one traveler has more than twice
                   the number of matching activities compared to another traveler
    """
    from project_lib import call_activity_by_id_api_mocked
    event_ids_not_matching = []
    event_ids_missing = []

    for itinerary_day in final_output.itinerary_days:
        for activity_recommendation in itinerary_day.activity_recommendations:
            event_id = activity_recommendation.activity.activity_id
            reference_event = call_activity_by_id_api_mocked(event_id)

            if reference_event is None:
                event_ids_missing.append(event_id)

            elif Activity(**reference_event) != activity_recommendation.activity:
                print(
                    "---\n"
                    f"Event ID {event_id} does not match the reference event:\n"
                    f"Reference Event: {reference_event}\n"
                    f"Activity Event: {activity_recommendation.activity.model_dump()}"
                )
                event_ids_not_matching.append(event_id)
            else:
                pass

    if event_ids_missing or event_ids_not_matching:
        raise AgentError(
            f"Event IDs missing: {event_ids_missing}\nEvent IDs not matching: {event_ids_not_matching}"
        )


get_eval_results(
    vacation_info=vacation_info,
    final_output=travel_plan_1,
    eval_functions=[eval_itinerary_events_match_actual_events],
)


def eval_itinerary_satisfies_interests(
    vacation_info: VacationInfo, final_output: TravelPlan
):
    """Verifies that the itinerary includes activities matching each traveler's interests.

    This function checks that each traveler has at least one activity in the itinerary that matches their interests.

        Args:
        vacation_info (dict): Contains the vacation details including traveler information and their interests
        final_output (dict): Contains the itinerary details including daily activities

    Raises:
        AgentError: If any traveler has no matching activities or if one traveler has more than twice
                   the number of matching activities compared to another traveler
    """
    traveler_to_interests = {}
    traveler_to_interest_hit_counts = {}

    for traveler in vacation_info.travelers:
        traveler_to_interests[traveler.name] = traveler.interests
        traveler_to_interest_hit_counts[traveler.name] = 0

    for traveler_name, interests in traveler_to_interests.items():
        for itinerary_day in final_output.itinerary_days:
            for activity_recommendation in itinerary_day.activity_recommendations:
                matching_interests = set(traveler_to_interests[traveler_name]) & set(
                    activity_recommendation.activity.related_interests
                )

                if matching_interests:
                    traveler_to_interest_hit_counts[traveler_name] += 1
                    print(
                        f"✅ Traveler {traveler_name} has a match with interest {matching_interests} at {activity_recommendation.activity.name}"
                    )

    travelers_with_no_interest_hits = [
        traveler
        for traveler, interest_hit_count in traveler_to_interest_hit_counts.items()
        if interest_hit_count == 0
    ]

    if travelers_with_no_interest_hits:
        raise AgentError(
            f"Travelers {travelers_with_no_interest_hits} has no matches with the itinerary."
        )


get_eval_results(
    vacation_info=vacation_info,
    final_output=travel_plan_1,
    eval_functions=[eval_itinerary_satisfies_interests],
)



ACTIVITY_AND_WEATHER_ARE_COMPATIBLE_SYSTEM_PROMPT = """
You are an expert travel advisor who determines whether a planned activity is suitable given the current weather conditions.

## Task
Given an activity name, its description, and the current weather condition, determine whether the activity is compatible with the weather.
When there is not enough information to decide, assume the activity IS_COMPATIBLE with the weather.
Also, look out for backup options mentioned in the activity description (e.g. "indoor alternative available") — if a backup exists, the activity IS_COMPATIBLE.

## Output format

    REASONING:
    * Step-by-step reasoning about whether the activity is suitable for the given weather condition.

    FINAL ANSWER:
    [IS_COMPATIBLE, IS_INCOMPATIBLE]

## Examples
`
Activity: Outdoor Tennis Tournament
Description: A competitive tennis tournament held on outdoor courts. No indoor alternative available.
Weather Condition: Heavy Rain

REASONING:
* Tennis is an outdoor sport.
* The tournament is held on outdoor courts with no indoor alternative.
* Heavy rain makes outdoor tennis unplayable and potentially dangerous.

FINAL ANSWER:
IS_INCOMPATIBLE

---

Activity: City Art Museum Tour
Description: A guided tour through the AgentsVille Museum of Modern Art, exploring local and international exhibits.
Weather Condition: Heavy Rain

REASONING:
* The art museum tour is an indoor activity.
* Rain has no impact on indoor activities.

FINAL ANSWER:
IS_COMPATIBLE
""".strip()



def eval_activities_and_weather_are_compatible(
    vacation_info: VacationInfo, final_output: TravelPlan
):
    """Verifies that no outdoor-only activities are scheduled during inclement weather conditions.

    Args:
        vacation_info (dict): Contains the vacation details
        final_output (dict): Contains the itinerary details including daily activities and weather conditions

    Raises:
        AgentError: If any outdoor activities are scheduled during weather conditions that could ruin them
    """
    from project_lib import do_chat_completion

    activities_that_are_incompatible = []

    for itinerary_day in final_output.itinerary_days:
        weather_condition = itinerary_day.weather.condition

        for activity_recommendation in itinerary_day.activity_recommendations:
            resp = do_chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": ACTIVITY_AND_WEATHER_ARE_COMPATIBLE_SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": f"Activity: {activity_recommendation.activity.name}\nDescription: {activity_recommendation.activity.description}\nWeather Condition: {weather_condition}",
                    },
                ],
                client=client,
                model=OpenAIModel.GPT_41_NANO,
            )
    


            if "IS_COMPATIBLE" in (resp or ""):
                is_compatible = True
            elif "IS_INCOMPATIBLE" in (resp or ""):
                is_compatible = False
            else:
                raise RuntimeError(
                    f"Unexpected response from the model: {resp}. Expected 'IS_COMPATIBLE' or 'IS_INCOMPATIBLE'."
                )

            if is_compatible:
                print(
                    f"✅ Activity {activity_recommendation.activity.name} (on {itinerary_day.date}) and weather '{weather_condition}' are compatible."
                )

            else:
                activities_that_are_incompatible.append(
                    activity_recommendation.activity.name
                )
                print(
                    f"❌ Activity {activity_recommendation.activity.name} (on {itinerary_day.date}) and weather '{weather_condition}' are incompatible."
                )

    if activities_that_are_incompatible:
        raise AgentError(
            f"Activities that may be ruined by inclement weather: {activities_that_are_incompatible}"
        )


eval_results = get_eval_results(
    vacation_info=vacation_info,
    final_output=travel_plan_1,
    eval_functions=[
        eval_activities_and_weather_are_compatible
    ],
)

eval_results

ALL_EVAL_FUNCTIONS = [
    eval_start_end_dates_match,
    eval_total_cost_is_accurate,
    eval_itinerary_events_match_actual_events,
    eval_itinerary_satisfies_interests,
    eval_total_cost_is_within_budget,
    eval_activities_and_weather_are_compatible,
]

eval_results = get_eval_results(
    vacation_info=vacation_info,
    final_output=travel_plan_1,
    eval_functions=ALL_EVAL_FUNCTIONS,
)

eval_results.model_dump()

def get_tool_descriptions_string(fns):
    """Generates a tool description from a function's docstring.
    Args:
        fns (list): List of functions to generate descriptions for.
    Returns:
        str: A formatted string containing the function names and their descriptions."""
    resp = ""
    for fn in fns:
        function_name = fn.__name__
        function_doc = fn.__doc__ or "No description provided."

        resp += f"* `{function_name}`: {function_doc}\n"

    return resp


def calculator_tool(input_expression) -> float:
    """Evaluates a mathematical expression and returns the result as a float.

    Args:
        input_expression (str): A string containing a valid mathematical expression to evaluate.

    Returns:
        float: The result of the evaluated expression.

    Example:
        >>> calculator_tool("1 + 1")
        2.0
    """
    import numexpr as ne
    return float(ne.evaluate(input_expression))


assert calculator_tool("1 + 1") == 2.0

print(get_tool_descriptions_string([calculator_tool]))


def get_activities_by_date_tool(date: str, city: str) -> List[dict]:
    """Retrieves a list of available activities for a given date and city.

    Args:
        date (str): The date in 'YYYY-MM-DD' format.
        city (str): The name of the city to retrieve activities for.

    Returns:
        List[dict]: A list of activity dictionaries available on the specified date.
    """
    from project_lib import call_activities_api_mocked
    resp = call_activities_api_mocked(date=date, city=city)

    return [Activity.model_validate(activity).model_dump() for activity in resp]



assert len(get_activities_by_date_tool("2025-06-10", "AgentsVille")) > 0

print(get_tool_descriptions_string([get_activities_by_date_tool]))

def run_evals_tool(travel_plan: TravelPlan) -> dict:
    """Runs all evaluation tools on the provided travel plan and vacation info.

    Args:
        travel_plan (TravelPlan): The travel plan to evaluate.

    Returns:
        EvaluationResults: The results of the evaluations.
    """
    if isinstance(travel_plan, dict):
        travel_plan = TravelPlan.model_validate(travel_plan)

    resp = get_eval_results(
        vacation_info=vacation_info,
        final_output=travel_plan,
        eval_functions=ALL_EVAL_FUNCTIONS,
    )
    return {
        "success": resp.success,
        "failures": resp.failures,
    }


print(get_tool_descriptions_string([run_evals_tool]))

run_evals_tool(travel_plan=travel_plan_1)


def final_answer_tool(final_output: TravelPlan) -> TravelPlan:
    """Returns the final travel plan

    Args:
        final_output (TravelPlan): The final travel plan to return.

    Returns:
        TravelPlan: The final travel plan.
    """
    return final_output


print(get_tool_descriptions_string([final_answer_tool]))

ALL_TOOLS = [
    calculator_tool,
    get_activities_by_date_tool,
    run_evals_tool,
    final_answer_tool,
]
print(get_tool_descriptions_string(ALL_TOOLS))


TRAVELER_FEEDBACK = "I want to have at least two activities per day."


def eval_traveler_feedback_is_incorporated(
    vacation_info: VacationInfo, final_output: TravelPlan
):
    """Checks if the traveler's feedback was incorporated into the revised travel plan.

    Args:
        vacation_info (VacationInfo): The vacation information.
        final_output (TravelPlan): The revised travel plan.

    Raises:
        AgentError: If the traveler's feedback was not successfully incorporated.
    """

    agent = ChatAgent(
        system_prompt="""You are an expert in evaluating whether a travel plan incorporates traveler feedback.

    ## Output Format

    Respond using two sections (ANALYSIS AND FINAL OUTPUT) in the following format:

        ANALYSIS:
        * [step-by-step analysis]


        FINAL OUTPUT:
        [FULLY_INCORPORATED, PARTIALLY_INCORPORATED, NOT_INCORPORATED, or UNKNOWN]
        REASON: [reasoning for the final output]

    """,
        client=client,
        model=OpenAIModel.GPT_41,  # Use a powerful model for checking traveler feedback
    )

    resp = agent.chat(
        f"""Traveler Feedback: {TRAVELER_FEEDBACK}
    Revised Travel Plan: {final_output.model_dump_json()}
    """,
    )
    if "FINAL OUTPUT:" not in resp:
        raise RuntimeError(
            f"Unexpected response from the model: {resp}. Expected 'FINAL OUTPUT:'."
        )
    if "FULLY_INCORPORATED" not in resp:
        final_output = resp.split("FINAL OUTPUT:")[-1].strip()
        raise AgentError(
            f"Traveler feedback was not successfully incorporated into the revised travel plan. Response: {final_output}"
        )

ALL_EVAL_FUNCTIONS = [
    eval_start_end_dates_match,
    eval_total_cost_is_accurate,
    eval_itinerary_events_match_actual_events,
    eval_itinerary_satisfies_interests,
    eval_total_cost_is_within_budget,
    eval_activities_and_weather_are_compatible,
    eval_traveler_feedback_is_incorporated,
]

get_eval_results(
    vacation_info=vacation_info,
    final_output=travel_plan_1,
    eval_functions=ALL_EVAL_FUNCTIONS,
)

from project_lib import print_in_box

ITINERARY_REVISION_AGENT_SYSTEM_PROMPT = f"""
You are an expert itinerary revision agent. Your job is to revise a travel itinerary based on traveler feedback and evaluation results, using a THOUGHT → ACTION → OBSERVATION cycle.

## Task

You will be given an existing travel itinerary to revise. Follow these steps:
1. Analyze the current itinerary and the traveler's feedback: "{TRAVELER_FEEDBACK}"
2. Use the available tools to check activities by date, calculate costs, and run evaluations.
3. Revise the itinerary to incorporate the traveler's feedback (at least 2 activities per day).
4. Ensure the itinerary stays within budget, matches the correct dates, and uses only real available activities.
5. Before providing the final answer, you MUST run the run_evals_tool to confirm all checks pass. Do NOT call final_answer_tool until run_evals_tool returns success.
6. Once all evaluations pass, call the final_answer_tool with the revised travel plan.

## Available Tools

{get_tool_descriptions_string(ALL_TOOLS)}

## Output Format

    THOUGHT:
    * [Your step-by-step reasoning about what to do next and why]

    ACTION:
    {{"tool_name": "[tool_name]", "arguments": {{"arg1": "value1"}}}}

Only output ONE action per response. Wait for the OBSERVATION before taking the next action.

## Context

Traveler feedback: "{TRAVELER_FEEDBACK}"

TravelPlan schema:
{json.dumps(TravelPlan.model_json_schema(), indent=2)}

Activity schema:
{json.dumps(Activity.model_json_schema(), indent=2)}

Weather data for the trip:
{json.dumps(weather_for_dates, indent=2, default=str)}
"""  


class ItineraryRevisionAgent(ChatAgent):
    system_prompt = ITINERARY_REVISION_AGENT_SYSTEM_PROMPT
    tools = ALL_TOOLS

    def get_observation_string(self, tool_call_obj) -> str:
        """Extracts the observation from the thought-action response."""

        if "tool_name" not in tool_call_obj:
            return "OBSERVATION: No tool name specified."

        if "arguments" not in tool_call_obj:
            return "OBSERVATION: No arguments specified."

        # If the arguments are not a dictionary, state the error
        if not isinstance(tool_call_obj["arguments"], dict):
            return f"OBSERVATION: Arguments should be a dictionary, got {type(tool_call_obj['arguments'])} instead."

        # If the tool name is not a string, state the error
        if not isinstance(tool_call_obj["tool_name"], str):
            return f"OBSERVATION: Tool name should be a string, got {type(tool_call_obj['tool_name'])} instead."

        tool_name = tool_call_obj["tool_name"]
        arguments = tool_call_obj["arguments"]

        tool_fn = None

        for tool in self.tools:
            if tool.__name__ == tool_name:
                tool_fn = tool
                break

        if tool_fn is None:
            return f"OBSERVATION: Unknown tool name '{tool_name}' in action string."

        try:
            tool_response = tool_fn(**arguments)
            return f"OBSERVATION: Tool {tool_name} called successfully with response: {tool_response}"
        except Exception as e:
            return f"OBSERVATION: Error occurred while calling tool {tool_name}: {e}"

    def run_react_cycle(
        self, original_travel_plan: TravelPlan, max_steps: int = 10, model: Optional[OpenAIModel] = None, client = None,
    ) -> TravelPlan:
        """Runs the ReAct cycle to revise the itinerary based on the evaluation results."""
        from json_repair import repair_json

        self.add_message(
            role="user",
            content=f"Here is the itinerary for revision:\n{original_travel_plan.model_dump_json()}",
        )
        resp = None

        for step in range(max_steps):
            resp = self.get_response(model=model, client=client) or ""

            if "ACTION:" not in resp:
                self.add_message(role="user", content="No action found in response.")
                continue

            action_string = resp.split("ACTION:")[1].strip()

            try:
                action_string = repair_json(action_string)
                tool_call_obj = json.loads(action_string)
            except json.JSONDecodeError:
                print(f"Invalid JSON in action string: {action_string}")
                self.add_message(
                    role="user",
                    content=f"Invalid JSON in action string: {action_string}",
                )
                continue

            tool_name = tool_call_obj.get("tool_name", None)

            if tool_name == "final_answer_tool":
                try:
                    new_travel_plan = TravelPlan.model_validate(
                        tool_call_obj["arguments"].get("final_output", tool_call_obj["arguments"])
                    )
                    return new_travel_plan
                except Exception as e:
                    self.add_message(
                        role="user", content=f"Error validating final answer: {e}"
                    )
                    continue

            else:
                observation_string = self.get_observation_string(
                    tool_call_obj=tool_call_obj
                )
                self.add_message(role="user", content=observation_string)

        raise RuntimeError(
            f"ReAct cycle did not complete within {max_steps} steps. Last response: {resp}"
        )

itinerary_revision_agent = ItineraryRevisionAgent()

resp = itinerary_revision_agent.chat(
    user_message=f"Here is the itinerary for revision: {travel_plan_1.model_dump_json(indent=2)}",
    add_to_messages=False,
    model=MODEL,
    client=client,
) or ""

print_in_box(resp, "Raw Response")
if "THOUGHT:" in resp:
    print("✅ `THOUGHT:` found in raw the response, as expected.")
else:
    print("❌ Expected `THOUGHT:` in raw the response. Please check the system prompt (output format).")
if "ACTION:" in resp:
    print("✅ `ACTION:` found in raw the response, as expected.")
else:
    print("❌ Expected `ACTION:` in raw the response. Please check the system prompt (output format).")
if "\"tool_name\"" in resp:
    print("✅ `\"tool_name\":` found in raw the response, as expected.")
else:
    print("❌ Expected `\"tool_name\":` in raw the response. Please check the system prompt (output format).")


itinerary_revision_agent = ItineraryRevisionAgent()
travel_plan_2 = itinerary_revision_agent.run_react_cycle(
    original_travel_plan=travel_plan_1, max_steps=15,
    model=MODEL,
    client=client,
)

print("✅ Revised itinerary generated successfully. Congratulations!")

eval_results_2 = get_eval_results(
    vacation_info=vacation_info,
    final_output=travel_plan_2,
    eval_functions=ALL_EVAL_FUNCTIONS,
)

assert eval_results_2.success, f"❌ Read the traces above and modify the system prompt.\n\nFailures: {eval_results_2.failures}"

print("✅ All evaluation functions passed successfully for the revised travel plan.")

eval_results_2


from IPython.display import display

for itinerary_day in travel_plan_2.itinerary_days:
    print(f"Date: {itinerary_day.date}")
    print(
        f"Weather: {itinerary_day.weather.condition} ({itinerary_day.weather.temperature}°{itinerary_day.weather.temperature_unit})"
    )

    activities_df = pd.DataFrame(
        [
            activity_recommendation.activity.model_dump()
            for activity_recommendation in itinerary_day.activity_recommendations
        ]
    )
    display(activities_df)



from project_lib import narrate_my_trip

narrate_my_trip(
    vacation_info=vacation_info,
    itinerary=travel_plan_2,
    client=client,
    model=MODEL,
)
