# AgentsVille Trip Planner - Project Assignment

In this project, you'll implement an AI system to help you plan a trip--to the wonderful city of AgentsVille!

Your AI system will demonstrate advanced LLM reasoning techniques including:

1. **Role-Based Prompting** - Your agent will act as a specialized travel planner
2. **Chain-of-Thought Reasoning** - Step-by-step planning of itineraries
3. **ReAct Prompting** - Thought → Action → Observation cycles
4. **Feedback Loops** - Self-evaluation using tools in the ReAct loop to find mistakes and improve plans

You'll simulate external API calls to gather weather data and activities. Then, process this information to create personalized travel itineraries based on interests and other constraints. Last, you'll implement a feedback loop to refine the plan.

Your task is to build a travel agent that can plan the perfect AgentsVille vacation!

## Project Instructions

Here are the steps you'll follow:

1. **Define Vacation Details**
    - Specify the trip duration, interests, and constraints.
    - Use Pydantic to structure and verify this information in a class called `VacationInfo`.
2. **Review Weather and Activity schedules**
    - Simulate API calls to gather weather data and available activities in bulk
    - Review the data manually to understand the available options
3. **The ItineraryAgent**
    - Implement an agent that generates a day-by-day itinerary based on the vacation details
    - The system prompt will guide the agent's reasoning through a step-by-step planning process to take travel preferences (e.g. destination, dates, interests) and generate a detailed day-by-day itinerary
    - Craft the components of the prompt (including the role, task/instructions, output format, examples, and context) to elicit the best possible itinerary in one LLM call
4. **Evaluating the Itinerary**
    - Evaluate the itinerary using a set of criteria to ensure a high-quality travel plan
        - For instance, does the itinerary match the city and the dates requested?
        - Or, is the total cost calulation accurate and is it within budget?
        - Or, does the agent hallucinate any activities that are not available?
        - Or, does the agent suggest activities that are not suitable for the weather? This specific evaluation function will require the use of an LLM to compare the event description against the weather data.
5. **Defining the Tools**
    - We will define four tools to assist the agent
        - **calculator_tool**: to accurately calculate costs
        - **get_activities_by_date_tool**: to retrieve activities for a specific date
        - **run_evals_tool**: to evaluate the itinerary against the criteria
        - **final_answer_tool**: to provide the final answer in a structured format
6. **The ItineraryRevisionAgent**
    - We will implement a second agent that revises the itinerary based on feedback using the ReAct THOUGHT → ACTION → OBSERVATION cycle.
        - The LLM will first generated a THOUGHT / ACTION message, which contains reasoning steps and a tool call invocation.
        - The Python code will parse the tool call and execute it, returning the result as a string to the LLM in an OBSERVATION message.
        - After this cycle repeats n number of times, the LLM will invoke the final_answer_tool to signal to the Python code to end the loop and return the final answer.
    - This agent will also **incorporate feedback on the initial itinerary** from the travelers to ensure the final plan has **at least 2 activities per day**. A new evaluation function using a powerful LLM will be created to check this user feedback.
    - The agent will use the tools above to refine the plan iteratively, checking the weather and available activities, and ensuring the itinerary meets all constraints.
7. **Something just for fun!**
    - To wrap things up we'll create a fun, narrative summary of the trip, highlighting the best activities and experiences!