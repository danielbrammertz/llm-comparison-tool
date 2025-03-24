import datetime
import json

import streamlit as st

import llm_openrouter as llm

st.set_page_config(page_title="Multi LLM Test Tool", layout="wide")
st.title("Multi LLM Test Tool")


@st.cache_data
def get_models():
    models = llm.available_models()
    models = sorted(models, key=lambda x: x.name)
    return models


def prepare_session_state():
    session_vars = {  # Default values
        "prompt": "",
        "temperature": 0.0,
        "max_tokens": 2048,
        "models": [],
        "response": {},
        "cost_and_stats": {},
    }
    for var in session_vars:
        if var not in st.session_state:
            st.session_state[var] = session_vars[var]


def configuration():
    models = get_models()
    st.session_state.prompt = st.text_area(
        "System prompt",
        value="You are a helpful assistant.",
        placeholder="Enter here the system prompt",
        height=120,
    )
    cols = st.columns([5, 2, 1])
    with cols[0]:
        st.session_state.models = st.multiselect("Model(s)", models, placeholder="Select one or more models")
    with cols[1]:
        st.session_state.temperature = st.slider("Temperature", 0.0, 1.0, 0.0)
    with cols[2]:
        st.session_state.max_tokens = st.number_input("Max completion tokens", 1, 20_480, 2048, step=10)

    # Models are listed in the order the user selected them
    # Sort the selected list by name to make them easier to find in the results
    st.session_state.models = sorted(st.session_state.models, key=lambda x: x.name)

    # Show all modes in a markdown table
    with st.expander("Click to to show/hide model details"):
        model_list = (
            "| Model | ID | Prompt price | Completion price |"
            "Context length | Max completion tokens | Tokenizer | Instruct type\n"
            "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |\n"
        )
        for model in models:
            model_list += (
                f"| {model.name} | {model.id} | {model.pricing_prompt:.10f} | {model.pricing_completion:.10f} |"
                f"{model.context_length:,} | {model.max_completion_tokens:,} | {model.tokenizer} |"
                f"{model.instruct_type} |\n"
            )
        st.markdown(model_list)


def get_llm_response(user_input: str) -> dict[llm.Model, llm.LLMResponse]:
    with st.spinner("Sending request..."):
        models = st.session_state.models
        if not isinstance(models, list):
            models = [models]
        response = llm.chat_completion_multiple(
            models, st.session_state.prompt, user_input, st.session_state.temperature, st.session_state.max_tokens
        )
    return response


def get_cost_and_stats(response: dict[llm.Model, llm.LLMResponse]) -> dict[llm.Model, llm.LLMCostAndStats]:
    with st.spinner("Calculating cost and stats..."):
        cost_and_stats = llm.cost_and_stats_multiple(response)
        # Filter out None values
        return {model: stats for model, stats in cost_and_stats.items() if stats is not None}


def show_response(response: dict[llm.Model, llm.LLMResponse], cost_and_stats: dict[llm.Model, llm.LLMCostAndStats]):
    # Sort the response by model name to keep the order consistent
    response = dict(sorted(response.items(), key=lambda x: x[0].name))

    # Show the response side by side by model
    for model, llm_response in response.items():
        with st.expander(f"{model.name} Response", expanded=True):
            st.markdown(llm_response.response)
            
            # Show cost and stats if available
            if model in cost_and_stats:
                stats = cost_and_stats[model]
                st.markdown("---")
                st.markdown("**Cost and Statistics:**")
                st.markdown(f"- Total Cost: ${stats.cost:.4f}")
                st.markdown(f"- GPT Tokens (Prompt/Completion): {stats.gpt_tokens_prompt}/{stats.gpt_tokens_completion}")
                st.markdown(f"- Native Tokens (Prompt/Completion): {stats.native_tokens_prompt}/{stats.native_tokens_completion}")
                st.markdown(f"- Response Time: {stats.elapsed_time:.2f}s")
            else:
                st.markdown("---")
                st.markdown("⚠️ Cost and statistics not available for this response")


prepare_session_state()
configuration()

user_input = st.text_area("Enter your request", placeholder="Enter here the user request", height=100)
send_button = st.button("Send Request")

if send_button:
    if not st.session_state.models:
        st.error("Please select at least one model")
        st.stop()
    if not user_input:
        st.error("Please enter a request")
        st.stop()

    # Because the download button (below) reruns the entire page (as all of the Streamlit widgets do), we need to
    # save the results in the session state to show them again after the download button is clicked
    # References:
    #  - https://github.com/streamlit/streamlit/issues/3832
    #  - https://discuss.streamlit.io/t/download-button-reloads-app-and-results-output-is-gone-and-need-to-re-run/51467
    st.session_state.response = get_llm_response(user_input)
    st.session_state.cost_and_stats = get_cost_and_stats(st.session_state.response)

if st.session_state.response:
    show_response(st.session_state.response, st.session_state.cost_and_stats)

    # Let the users download the results in JSON format
    response_json = {}
    for model, response in st.session_state.response.items():
        response_json[model.name] = {
            "response": response.to_dict(),
            "cost_and_stats": st.session_state.cost_and_stats.get(model, {}).to_dict() if model in st.session_state.cost_and_stats else None
        }
    response_json = json.dumps(response_json, indent=4)
    st.download_button(
        label="Download JSON File",
        data=response_json,
        file_name=f"llm-comparison-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}",
        mime="application/json",
    )
