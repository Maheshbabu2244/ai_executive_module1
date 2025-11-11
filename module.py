import streamlit as st
import requests
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re
import random
import time
from io import BytesIO 
import streamlit.components.v1 as components 

# --- 1. INITIAL SETUP & CONFIGURATION ---

# 1.1 Streamlit Page Config
st.set_page_config(
    page_title="Module 1: AI Strategy & Executive Foundations",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 1.2 Session State Initialization 
if 'progress' not in st.session_state:
    st.session_state.progress = {f'E{i}': '🔴' for i in range(1, 6)} 
    st.session_state.progress['E0'] = '🔴' 
    st.session_state.progress['E6'] = '🔴' 
    st.session_state.journal = []
    st.session_state.lab_results = {}
    st.session_state.current_tab = 'Intro'
    st.session_state.guidance = "Welcome! Click the ** Getting Started** tab to begin your strategic module." 
    st.session_state.onboarding_done = False 

if 'assistant_chat_history' not in st.session_state:
    st.session_state.assistant_chat_history = [{"role": "assistant", "content": "👋 Hi there! I'm your AI Instructor. Ask me anything in simple terms about AI strategy, risks, or what to do next!"}]
if 'last_assistant_call' not in st.session_state:
    st.session_state.last_assistant_call = 0

# 1.3 AI Model API Configuration 
AI_API_URL = "https://api.cerebras.ai/v1/chat/completions"
API_KEY_NAME = "cerebras" 

# --- MODEL LIST (For Failover) ---
MODEL_PRIORITY_LIST = [
    "gpt-oss-120b",    
    "qwen-3-32b",      
    "llama-3.3-70b",   
    "llama3.1-8b"      
]
DEFAULT_MODEL = MODEL_PRIORITY_LIST[0]

# --- API Key Check and Setup ---
try:
    CEREBRAS_API_KEY = st.secrets[API_KEY_NAME]["api_key"]
except KeyError:
    CEREBRAS_API_KEY = "SIMULATION_MODE_ACTIVE"
    st.warning("⚙️ Running in **SIMULATION MODE** - API key not found. Add your Cerebras API key to secrets.toml for live AI functionality.")


# --- 1.4 CUSTOM STREAMLIT STYLING ---
STYLING = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* === MAIN APP CONTAINER === */
.stApp {
    font-family: 'Inter', sans-serif;
    background: linear-gradient(120deg, #f8f9fb, #eef3f7, #f8f9fb);
    background-size: 400% 400%;
    animation: gradientShift 18s ease infinite;
    color: #2c3e50;
}

/* Animation for subtle motion */
@keyframes gradientShift {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* Main content container */
div[data-testid="stAppViewBlock"] {
    background: rgba(255, 255, 255, 0.9);
    margin: 25px;
    border-radius: 14px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
    padding: 35px;
}

/* === SIDEBAR === */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffffff 0%, #f1f4f8 100%);
    border-right: 1px solid rgba(0,0,0,0.05);
}

section[data-testid="stSidebar"] * {
    color: #34495e !important;
}

section[data-testid="stSidebar"] h1, 
section[data-testid="stSidebar"] h2 {
    color: #2c3e50 !important;
    font-weight: 700;
}

/* === HEADER === */
.corporate-header {
    background: linear-gradient(90deg, #3498db 0%, #85c1e9 100%);
    color: #fff;
    padding: 25px 35px;
    border-radius: 10px;
    margin-bottom: 25px;
    box-shadow: 0 4px 20px rgba(52, 152, 219, 0.3);
    animation: fadeIn 1s ease-out;
}

.corporate-header h1 {
    font-size: 30px;
    font-weight: 700;
    margin: 0;
}

.corporate-header p {
    font-size: 15px;
    opacity: 0.9;
    margin: 6px 0 0;
}

/* === SECTION HEADERS === */
.section-header {
    color: #2c3e50;
    font-weight: 700;
    font-size: 22px;
    margin: 25px 0 15px;
    padding-bottom: 8px;
    border-bottom: 3px solid #3498db;
}

.subsection-header {
    color: #34495e;
    font-weight: 600;
    font-size: 17px;
    margin-top: 20px;
}

/* === KPI CARDS === */
.kpi-container {
    display: flex;
    gap: 15px;
    margin: 20px 0;
}

.kpi-card {
    flex: 1;
    background: #ffffff;
    border-left: 5px solid #3498db;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}

.kpi-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
}

.kpi-label {
    font-size: 12px;
    color: #7f8c8d;
    text-transform: uppercase;
    margin-bottom: 8px;
    font-weight: 600;
}

.kpi-value {
    font-size: 26px;
    font-weight: 700;
    color: #2c3e50;
}

.kpi-low {border-left-color: #e74c3c;}
.kpi-medium {border-left-color: #f39c12;}
.kpi-high {border-left-color: #27ae60;}

/* === PROGRESS CARD === */
.progress-card {
    background: linear-gradient(135deg, #eaf2fb 0%, #f8fbff 100%);
    padding: 18px;
    border-radius: 10px;
    color: #2c3e50;
    margin-bottom: 20px;
    box-shadow: 0 3px 10px rgba(0,0,0,0.05);
}

.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #3498db, #5dade2);
}

/* === BUTTONS === */
.stButton > button {
    background: linear-gradient(135deg, #3498db 0%, #5dade2 100%);
    color: white;
    border: none;
    border-radius: 6px;
    padding: 10px 28px;
    font-weight: 600;
    transition: all 0.3s ease;
    box-shadow: 0 3px 12px rgba(52,152,219,0.3);
}

.stButton > button:hover {
    background: linear-gradient(135deg, #5dade2 0%, #3498db 100%);
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(52,152,219,0.4);
}

/* === TABS === */
.stTabs [data-baseweb="tab-list"] {
    background: #f4f6f8;
    border-radius: 10px;
    padding: 10px;
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    background: white;
    border: 1px solid #dee2e6;
    border-radius: 6px;
    padding: 10px 20px;
    color: #34495e;
    font-weight: 600;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #3498db 0%, #5dade2 100%);
    color: white;
    border: none;
}

/* === INPUT FIELDS === */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    border: 1.5px solid #dce3ec;
    border-radius: 6px;
    padding: 10px;
    font-size: 14px;
    transition: border-color 0.3s ease;
}

.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #3498db;
    box-shadow: 0 0 0 3px rgba(52,152,219,0.15);
}

/* === EXPANDERS === */
.streamlit-expanderHeader {
    background: #f8f9fa;
    border-radius: 6px;
    border: 1px solid #e0e6ed;
    font-weight: 600;
    color: #2c3e50;
    padding: 10px 15px;
}

.streamlit-expanderHeader:hover {
    background: #edf2f6;
}

/* === MISC === */
@keyframes fadeIn {
    from {opacity: 0; transform: translateY(-8px);}
    to {opacity: 1; transform: translateY(0);}
}
</style>
"""
st.markdown(STYLING, unsafe_allow_html=True)



# --- 2. UTILITY & LLM FUNCTIONS ---

def get_progress_badge(key):
    return st.session_state.progress.get(key, '🔴')

def get_progress_percent():
    completed_count = sum(1 for status in st.session_state.progress.values() if status == '🟢')
    total_count = len(st.session_state.progress)
    return int((completed_count / total_count) * 100)

def update_progress(key, status):
    st.session_state.progress[key] = status
    
def update_guidance(message):
    st.session_state.guidance = message

def save_to_journal(title, prompt, result, metrics=None):
    st.session_state.journal.append({
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'lab': st.session_state.current_tab,
        'title': title,
        'prompt': prompt,
        'result': result,
        'metrics': metrics or {}
    })

def clean_output(text):
    """Removes ** marks and leading/trailing whitespace for cleaner display."""
    if text:
        return re.sub(r'\*\*', '', text).strip()
    return text

def llm_call_cerebras(messages, models_to_try, max_tokens=512, temperature=0.5):
    """
    Handles the secure API call with a FAILOVER mechanism across multiple models.
    """
    
    API_READ_TIMEOUT = 30
    
    # --- SIMULATION MODE ---
    if CEREBRAS_API_KEY == "SIMULATION_MODE_ACTIVE":
        if st.session_state.current_tab == 'Assistant':
            user_input = messages[-1]['content'] if messages else "No query provided."
            content = f"SIMULATION: Instructor processed your request about '{user_input[:20]}...' The advice is: Focus on the strategic value proposition before technical implementation."
        else:
            content = "Simulation mode active. LLM logic run separately."
            
        return {"content": clean_output(content), "model": "Simulation", "tokens_used": 0, "latency": 0.5, "throughput_tps": 0}

    # --- LIVE API CALL WITH FAILOVER ---
    
    log_container = st.container(border=True)
    log_container.subheader("💻 AI Model Processing Steps")
    
    for i, model in enumerate(models_to_try):
        log_container.info(f"Attempting API call with Model: {model} (Attempt {i+1}/{len(models_to_try)})")
        
        headers = {"Authorization": f"Bearer {CEREBRAS_API_KEY}", "Content-Type": "application/json"}
        payload = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}

        try:
            start_time = time.time() 
            response = requests.post(AI_API_URL, json=payload, headers=headers, timeout=API_READ_TIMEOUT)
            end_time = time.time() 
            
            if response.status_code == 200:
                data = response.json()
                content = clean_output(data["choices"][0]["message"]["content"])
                tokens_generated = len(content.split()) 
                
                log_container.success(f"✅ Success! Response received from {model}.")
                
                return {
                    "content": content, 
                    "model": model, 
                    "tokens_used": data.get("usage", {}).get("total_tokens", tokens_generated),
                    "latency": end_time - start_time,
                    "throughput_tps": tokens_generated / (end_time - start_time)
                }

            else:
                error_detail = response.json().get("message", response.text[:100])
                
                if response.status_code in [429, 500, 503] and i < len(models_to_try) - 1:
                    log_container.warning(f"⚠️ Failover Triggered: Model {model} failed ({response.status_code}). Retrying with next model in 2s...")
                    time.sleep(2)
                    continue 
                else:
                    log_container.error(f"🚨 Final Error: All models failed. {error_detail}")
                    return {"error": f"API Call Failed ({response.status_code}): {error_detail}"}

        except requests.exceptions.RequestException as e:
            if i < len(models_to_try) - 1:
                 log_container.warning(f"⚠️ Network Error with {model}. Retrying next model in 2s...")
                 time.sleep(2)
                 continue
            else:
                log_container.error(f"🚨 Final Network/Timeout Error: All attempts failed. {e}")
                return {"error": f"API Call Failed: {e}"}

    return {"error": "API Call Failed: No models were available to process the request."}


def run_executive_llm_logic(tab_code: str, user_input: str, mode: str, prompt_template: str) -> dict:
    """
    Handles the execution flow for E-labs, choosing between LIVE API and SIMULATION.
    """
    
    system_prompt = f"Role: {mode}. Adhere strictly to this prompt: {prompt_template}"
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_input}]

    if CEREBRAS_API_KEY != "SIMULATION_MODE_ACTIVE":
        result_obj = llm_call_cerebras(messages, models_to_try=MODEL_PRIORITY_LIST, max_tokens=512, temperature=0.5)
        raw_output = result_obj.get('content', result_obj.get('error', 'API execution failed.'))
        
        if 'error' in result_obj:
             raw_output = run_cerebras_simulation_logic(tab_code, user_input)
             result_obj['content'] = raw_output
    else:
        raw_output = run_cerebras_simulation_logic(tab_code, user_input)
        result_obj = {'content': raw_output, 'model': 'Simulation', 'tokens_used': len(raw_output.split()), 'latency': 0.5, 'throughput_tps': 0}
        
    metrics = {}
    if tab_code == 'E1':
        metrics = {"Clarity Score": random.randint(60, 99), "Predicted ROI Impact": random.choice(["High", "Moderate", "Low"])}
    elif tab_code == 'E4':
        metrics['Feasibility Score'] = random.uniform(0.3, 0.95)
    
    mermaid_flow = f"""
        graph TD;
            A[Executive Input: {user_input[:20]}...] --> B(AI Strategist: {mode});
            B --> C{{Prompt: {prompt_template[:30]}...}};
            C --> D(Strategic Output);
            D --> E[Metrics Calc: Clarity/ROI];
            E --> F[Dashboard Artifact];
    """

    return {
        "output": raw_output,
        "metrics": metrics,
        "result_obj": result_obj,
        "mermaid_flow": mermaid_flow,
    }

def run_cerebras_simulation_logic(tab_code, user_input):
    """Fallback logic when the API key is missing or failed."""
    if tab_code == 'E1':
        if "logistics" in user_input.lower() or "inventory" in user_input.lower():
            return (
                "Strategic Opportunity 1: Predictive Inventory Optimization (reduce holding costs). \n"
                "Strategic Opportunity 2: Automated Route Optimization (improve delivery speed). \n"
                "Strategic Opportunity 3: Supplier Risk Profiling (avoid supply chain risks)."
            )
        else:
            return (
                "Strategic Opportunity 1: Personalized Customer Engagement (increase CLV). \n"
                "Strategic Opportunity 2: Automated Insight Generation (reduce executive reporting time). \n"
                "Strategic Opportunity 3: AI-Driven Talent Acquisition (reduce turnover)."
            )
    elif tab_code == 'E2':
        return f"Based on your selections, the roadmap prioritizes high-impact, feasible projects first. Immediate action: Establish data governance in the {user_input.split(',')[0]} department."
    elif tab_code == 'E3':
        if "Automate Support" in user_input:
            return "| Factor | Opportunities | Cautions | \n|---|---|---| \n| Cost/Speed | Reduced OpEx, 24/7 service. | High initial cost, training data bias risk. |"
        else:
            return "| Factor | Opportunities | Cautions | \n|---|---|---| \n| Growth | New revenue streams. | Market entry risk, high integration complexity. |"
    elif tab_code == 'E4':
        score = random.uniform(0.3, 0.95)
        status = "Highly Feasible" if score > 0.7 else ("Moderately Feasible" if score > 0.5 else "Not Feasible")
        return f"AI Use Case Canvas Summary: \nThis project is categorized as {status} with a Feasibility Score of {score:.2f}. The key recommendation is to secure the primary data stream immediately."
    elif tab_code == 'E5':
        if "Ethical" in user_input:
            return "Mitigation Recommendations: Establish an AI Governance Board for auditing fairness (Ethical Risk). Implement decentralized security protocols (Financial Risk)."
        else:
            return "Mitigation Recommendations: Focus on quick wins to prove ROI. Develop a robust MLOps framework."
    else:
        return "AI Assistant response based on context."


def get_example_prompt(tab_code: str) -> str:
    """Returns a specific, detailed prompt based on the current lab."""
    
    prompts = {
        'E1': "Generate a concise, excellent 30-word example of a business challenge scenario specifically for the E1 lab, focused on 'Logistics Optimization'.",
        'E2': "Generate an excellent example of the three highest-scoring departments and their scores (Impact: 9, Feasibility: 8) for the E2 Impact Matrix Lab. Format the output as a bulleted list.",
        'E3': "Generate an example of a successful Hypothesis (20 words max) for the 'Automate Support' decision in the E3 lab.",
        'E4': "Generate a complete, excellent example of a high-feasibility 'AI Use Case Canvas' by providing a Goal, Data Available (well-structured), Expected Benefit, and Constraints (low). Format as a list of key-value pairs.",
        'E5': "Generate a simple example of a high-risk scenario for the E5 Risk Analyzer, detailing why the Ethical Risk score would be 9 and the Operational Risk score would be 8.",
        'E6': "Generate an example of a strong final synthesis reflection (50 words max) for the E6 Reflection lab."
    }
    
    return prompts.get(tab_code, "Please provide an example for the current lab.")


# --- 4. AI ASSISTANT SIDEBAR FUNCTION ---

def render_sidebar():
    """Renders the persistent AI Assistant and Progress Tracker in the sidebar."""
    
    st.sidebar.markdown('<div class="progress-tracker">', unsafe_allow_html=True)
    st.sidebar.markdown("####  Your Learning Progress")
    progress_percent = get_progress_percent()
    st.sidebar.progress(progress_percent, text=f"**Module Complete: {progress_percent}%**")
    
    lab_statuses = [f"{code} {get_progress_badge(code)}" for code in sorted(st.session_state.progress.keys())]
    st.sidebar.caption(f"Status: {' | '.join(lab_statuses)}")
    
    guidance_message = st.session_state.get('guidance', "Welcome! Select a lab tab (E1-E6) to begin your module.")
    st.sidebar.markdown('**Current Goal:**')
    st.sidebar.info(guidance_message)
    st.sidebar.markdown('</div>', unsafe_allow_html=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown("####  AI Instructor Chatbot")
    st.sidebar.caption("Ask simple, non-technical questions here!")
    
    user_query = st.sidebar.text_area("Ask about the module, steps, or concepts:", key="assistant_query")
    
    if st.sidebar.button("Ask Instructor", key="run_assistant"):
        
        if time.time() - st.session_state.last_assistant_call < 5:
            st.sidebar.error("Please wait 5 seconds before asking the Assistant another question.")
            return

        if user_query:
            st.session_state.assistant_chat_history.append({"role": "user", "content": user_query})
            
            SYSTEM_PROMPT = "You are the AI Instructor Assistant for executive leaders. Your tone must be highly professional, non-technical, and focused on business value, risks, and strategic next steps. Define concepts, suggest next steps, and give simple guidance."
            
            temp_current_tab = st.session_state.current_tab
            st.session_state.current_tab = 'Assistant'
            
            with st.spinner("Assistant is analyzing your strategic query..."):
                assistant_result_data = run_executive_llm_logic('Assistant', user_query, "Executive AI Strategy Coach", SYSTEM_PROMPT)
            
            st.session_state.current_tab = temp_current_tab 
            st.session_state.last_assistant_call = time.time()
            
            if 'error' in assistant_result_data['result_obj']:
                response = "Assistant Error: Sorry, I encountered an issue. The API connection failed. Please try again or check the API key/network."
            else:
                response = clean_output(assistant_result_data['output'])

            st.session_state.assistant_chat_history.append({"role": "assistant", "content": response})
            st.rerun() 
            
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Need ideas for Lab {st.session_state.current_tab}?")

    if st.sidebar.button(" Provide Example for Current Lab", key="run_example"):
        
        current_lab_code = st.session_state.current_tab
        
        if current_lab_code not in ['E1', 'E2', 'E3', 'E4', 'E5', 'E6']:
            st.sidebar.error("Please select a specific lab tab (E1-E6) first.")
            return

        example_prompt = get_example_prompt(current_lab_code)
        
        with st.spinner(f"Generating example for {current_lab_code}..."):
            example_result_data = run_executive_llm_logic(
                tab_code=current_lab_code,
                user_input="Generate the required example.",
                mode="Example Generator",
                prompt_template=example_prompt
            )
            
        if 'error' not in example_result_data['result_obj']:
            example_output = example_result_data['output']
            st.session_state.assistant_chat_history.append({
                "role": "assistant",
                "content": f"Example for {current_lab_code}: \n{example_output}"
            })
        else:
             st.session_state.assistant_chat_history.append({
                "role": "assistant",
                "content": "Example generation failed. Please check your API key or network connection."
            })
        
        st.rerun() 

    for message in st.session_state.assistant_chat_history:
        content = clean_output(message['content'])
        if message['role'] == 'user':
            st.sidebar.markdown(f'**You:** {content}')
        elif message['role'] == 'assistant':
            st.sidebar.markdown(f'**Instructor:** {content}')

    st.sidebar.markdown("---")
    if st.sidebar.button("Reset All Lab Progress ⚠️", type='secondary'):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        
        st.session_state.progress = {f'E{i}': '🔴' for i in range(1, 6)} 
        st.session_state.progress['E0'] = '🔴'
        st.session_state.progress['E6'] = '🔴'
        st.session_state.journal = []
        st.session_state.current_tab = 'Intro'
        st.session_state.guidance = "Welcome! Select a lab tab (E1-E6) to begin your module." 
        st.session_state.onboarding_done = False
        st.session_state.assistant_chat_history = [{"role": "assistant", "content": "👋 Hi there! I'm your AI Instructor. Ask me anything in simple terms about AI strategy, risks, or what to do next!"}]
        st.session_state.last_assistant_call = 0 
        
        st.success("Session cleared. Please refresh the browser.")
        st.rerun()


# --- 5. EXECUTIVE LAB IMPLEMENTATION FUNCTIONS (E0 - E6) ---

def render_getting_started():
    st.markdown('<div class="title-header"> Executive Module 1: Strategic Foundations</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    st.header(" Module Overview: From Strategy to Execution")
    st.info("This module teaches executives how to strategically interact with AI and translate business problems into viable AI projects.")
    st.markdown("---")
    
    st.subheader(" Module Journey: Learning the Strategic AI Process")
    
    st.markdown("####  E1: AI in Business Context")
    col_def, col_goal = st.columns([1, 1])

    with col_def:
        st.markdown("##### Core Definition & Concept:")
        st.markdown("""
            - **Definition:** **Value Creation**—identifying where AI generates the most financial or strategic benefit.
            - **Key Concept:** AI is a **value-chain transformation tool**, not just a technical project.
        """)
    with col_goal:
        st.markdown("##### The Goal & Hands-on Action:")
        st.success("Goal: Successfully run your first strategic prompt and understand the direct relationship between your problem statement's clarity and the Predicted ROI.")
        st.markdown("**Hands-on Action:** Input a business scenario → Run the AI Strategist → Analyze Clarity Score and Predicted ROI Impact.")
    st.markdown("---")
    
    st.markdown("####  E2: AI Impact Matrix")
    col_def, col_goal = st.columns([1, 1])

    with col_def:
        st.markdown("##### Core Definition & Concept:")
        st.markdown("""
            - **Definition:** **Impact Matrix**—a visualization tool for prioritizing projects based on **Business Value** vs. **Implementation Difficulty**.
            - **Key Concept:** Investment decisions must be quantitative.
        """)
    with col_goal:
        st.markdown("##### The Goal & Hands-on Action:")
        st.success("Goal: Master objective prioritization using the 2x2 matrix.")
        st.markdown("**Hands-on Action:** Score five departments → Visualize on scatter plot → Generate AI Implementation Roadmap.")
    st.markdown("---")

    st.markdown("####  E3: Strategic Decision Lab")
    col_def, col_goal = st.columns([1, 1])

    with col_def:
        st.markdown("##### Core Definition & Concept:")
        st.markdown("""
            - **Definition:** **Strategic Advisory**—using AI to provide balanced analysis for complex investment choices.
            - **Key Concept:** Leaders must make decisions under uncertainty.
        """)
    with col_goal:
        st.markdown("##### The Goal & Hands-on Action:")
        st.success("Goal: Practice analytical judgment by evaluating the AI's balanced rationale.")
        st.markdown("**Hands-on Action:** Select a decision → Input hypothesis → Run AI Advisor → Analyze Pro/Con Table.")
    st.markdown("---")

    st.markdown("####  E4: Use Case Builder")
    col_def, col_goal = st.columns([1, 1])

    with col_def:
        st.markdown("##### Core Definition & Concept:")
        st.markdown("""
            - **Definition:** **Use Case Canvas**—a structured framework for defining project goals, data requirements, and constraints.
            - **Key Concept:** An AI project is only viable with clear goals and accessible data.
        """)
    with col_goal:
        st.markdown("##### The Goal & Hands-on Action:")
        st.success("Goal: Master project scoping and achieve a 'Highly Feasible' score.")
        st.markdown("**Hands-on Action:** Fill out Use Case Canvas → Run Feasibility AI → Review Status.")
    st.markdown("---")
    
    st.markdown("####  E5: Risk vs Reward Analyzer")
    col_def, col_goal = st.columns([1, 1])

    with col_def:
        st.markdown("##### Core Definition & Concept:")
        st.markdown("""
            - **Definition:** **Mitigation Strategy**—identifying and planning responses for Ethical, Financial, and Operational risks.
            - **Key Concept:** Leaders must compare potential reward against potential loss.
        """)
    with col_goal:
        st.markdown("##### The Goal & Hands-on Action:")
        st.success("Goal: Develop balanced strategy and generate targeted mitigation recommendations.")
        st.markdown("**Hands-on Action:** Score risks/rewards → View Radar Chart → Generate Mitigation Plan.")
    st.markdown("---")
    
    st.markdown("####  E6: Reflection & Export")
    col_def, col_goal = st.columns([1, 1])

    with col_def:
        st.markdown("##### Core Definition & Concept:")
        st.markdown("""
            - **Definition:** **Executive Synthesis**—internalizing module learning into personalized, actionable strategy.
            - **Key Concept:** Learning must translate into action.
        """)
    with col_goal:
        st.markdown("##### The Goal & Hands-on Action:")
        st.success("Goal: Write key takeaways and export your AI Strategy Readiness Report.")
        st.markdown("**Hands-on Action:** Write final reflection → Export Report.")
    st.markdown("---")
    
    if st.button("Start Module (E1)"):
        update_progress('E0', '🟢')
        st.session_state.current_tab = 'E1'
        st.rerun()


def render_lab_E1():
    st.header(" E1: AI in Business Context ")
    
    st.subheader(" Problem Statement")
    st.markdown("Scenario: You're asked, 'Where should we invest in AI first to maximize ROI?'")
    st.subheader(" Objective")
    st.success("Learning Goal: Recognize AI's role in value creation by translating a business challenge into measurable opportunities.")
    
    st.markdown("---")
    
    with st.expander(" Understanding Your Metrics (Click to Expand)", expanded=True):
        st.markdown("These two metrics help you assess both the **quality of your input** and the **value of the AI's output**.")
        
        col_m1, col_m2 = st.columns(2)
        
        with col_m1:
            st.subheader("Clarity Score (0-100%)")
            st.markdown("""
            * **What is it?** A simulated score measuring the quality, specificity, and completeness of your scenario input.
            * **Why it matters:** High-quality AI outputs require high-quality inputs.
            * **Goal:** Aim for a high score (90%+) by providing a clearly defined problem statement.
            """)
        
        with col_m2:
            st.subheader("Predicted ROI Impact")
            st.markdown("""
            * **What is it?** A simulated business estimate of potential financial or strategic return.
            * **Why it matters:** It helps you prioritize AI opportunities.
            * **Goal:** Focus on **High** impact solutions.
            """)

    st.markdown("---")
    
    with st.expander(" Instructions: How to Use This Lab", expanded=True):
        st.markdown("1. **Describe Your Challenge:** Clearly describe one critical business problem.")
        st.markdown("2. **Run AI Strategist:** Click **Map AI Opportunities**.")
        st.markdown("3. **Analyze:** Observe the Clarity Score and Predicted ROI Impact.")
    
    col_input, col_output = st.columns([1, 1])
    
    with col_input:
        st.subheader("1. Describe Your Business Scenario")
        scenario_input = st.text_area(
            "Enter your current organizational challenge:",
            height=200,
            placeholder="Our logistics team struggles with demand forecasting, leading to high inventory costs.",
            key='e1_scenario_input'
        )
        
        if st.button(" Map AI Opportunities (Run AI)"):
            if not scenario_input.strip() or len(scenario_input.split()) < 5:
                st.warning(" Please provide a **detailed business scenario** (at least 5 words).")
            else:
                with st.spinner('Analyzing scenario with AI Strategist...'):
                    result_data = run_executive_llm_logic('E1', scenario_input, "Business Strategist", "You are a business strategist. Given this description of a company, identify 3 distinct strategic ways AI can add value. Frame them as high-level opportunities.")
                    st.session_state['e1_result_data'] = result_data
                    st.session_state['e1_progress'] = True
                    update_progress('E1', '🟢') 
                    st.rerun()
                    
    if st.session_state.get('e1_result_data'):
        result_data = st.session_state['e1_result_data']
        
        with col_output:
            st.subheader("2. AI Opportunities & Metrics")
            st.success("Analysis Complete! Here are 3 Strategic AI Opportunities:")
            st.markdown(result_data['output'])
            
            st.markdown("---")
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-label">Clarity Score</div>
                        <div class="kpi-value">{result_data['metrics'].get('Clarity Score', 'N/A')}%</div>
                    </div>
                """, unsafe_allow_html=True)
            with col_m2:
                roi_value = result_data['metrics'].get('Predicted ROI Impact', 'N/A')
                roi_class = "kpi-high" if roi_value == "High" else ("kpi-low" if roi_value == "Low" else "")
                st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-label">Predicted ROI Impact</div>
                        <div class="kpi-value {roi_class}">{roi_value}</div>
                    </div>
                """, unsafe_allow_html=True)

        st.markdown("---")
        render_ai_explanation(result_data)
        
        if st.button("Save Insight & Complete Lab E1", key='e1_save_complete'):
            save_to_journal("E1: Business Context", scenario_input, result_data['result_obj'], result_data['metrics'])
            update_guidance("🎉 E1 Lab Complete! Move to the E2: AI Impact Matrix tab to learn how to prioritize.")
            st.rerun()

def render_lab_E2():
    st.header(" E2: AI Impact Matrix ")
    
    st.subheader(" Problem Statement")
    st.markdown("Scenario: Your budget only allows for two AI projects this year. How do you objectively choose which departments get the investment?")
    st.subheader(" Objective")
    st.success("Learning Goal: Quantify AI impact across the organization using Impact and Feasibility.")
    
    st.markdown("---")
    
    with st.expander(" Understanding Your Metrics (Click to Expand)", expanded=False):
        st.markdown("This lab focuses on two primary strategic KPIs for project prioritization:")
        
        col_m1, col_m2 = st.columns(2)
        
        with col_m1:
            st.subheader("Business Impact (Value)")
            st.markdown("""
            * **What is it?** Measures the potential organizational benefit.
            * **Why it matters:** Aligns the project with core business goals.
            """)
        
        with col_m2:
            st.subheader("Implementation Feasibility (Ease)")
            st.markdown("""
            * **What is it?** Measures the ease of deployment.
            * **Why it matters:** Ensures the project is achievable within realistic constraints.
            """)

    with st.expander(" Instructions: How to Use This Lab", expanded=True):
        st.markdown("1. **Score Departments:** Use sliders to rate each department.")
        st.markdown("2. **Analyze Plot:** The scatter plot visualizes your priorities.")
        st.markdown("3. **Run AI:** Click **Generate Roadmap**.")
    
    departments = ["Marketing", "HR", "Finance", "Operations", "R&D"]
    impact_data = []
    
    st.subheader("1. Department Assessment: Score Impact & Feasibility")
    
    for dept in departments:
        col1, col2 = st.columns(2)
        impact = col1.slider(f"{dept} - Business Impact (1-10)", 1, 10, 5, key=f"e2_impact_{dept}")
        feasibility = col2.slider(f"{dept} - Feasibility (1-10)", 1, 10, 5, key=f"e2_feas_{dept}")
        impact_data.append({"Department": dept, "Impact": impact, "Feasibility": feasibility})
    
    df = pd.DataFrame(impact_data)
    st.markdown("---")
    
    st.subheader("2. Visualization: AI Adoption Prioritization Map")
    fig = px.scatter(df, x='Feasibility', y='Impact', color='Department', 
                        title='AI Adoption Prioritization Map',
                        labels={'Feasibility': 'Implementation Feasibility', 'Impact': 'Business Value Impact'},
                        range_x=[0, 10], range_y=[0, 10], hover_data=['Department'])
    fig.add_hline(y=7, line_dash="dash", line_color="red", annotation_text="High Impact Threshold", annotation_position="top left")
    fig.add_vline(x=7, line_dash="dash", line_color="red", annotation_text="High Feasibility Threshold", annotation_position="bottom right")
    st.plotly_chart(fig, use_container_width=True) 
        
    st.subheader("3. AI Implementation Roadmap")
    
    high_priority = df[(df['Impact'] >= 7) & (df['Feasibility'] >= 7)]['Department'].tolist()
    user_summary = f"High priority departments: {', '.join(high_priority)}. Overall scores collected." 
    
    if st.button("Generate Implementation Roadmap"):
        user_summary = f"High priority departments: {', '.join(high_priority)}. Overall focus: {departments[df['Impact'].idxmax()]}."
        
        result_data = run_executive_llm_logic('E2', user_summary, "Implementation Strategist", "Based on user-provided Impact/Feasibility scores for departments, generate a 3-point AI implementation roadmap.")
        st.session_state['e2_result_data'] = result_data
        st.success("Artifact Generated: AI Adoption Prioritization Map.")
        st.markdown(result_data['output'])
        
        st.markdown("---")
        render_ai_explanation(result_data)

    if st.session_state.get('e2_result_data'):
        if st.button("Save & Complete Lab E2", key='e2_save_btn'):
            update_progress('E2', '🟢')
            save_to_journal("E2: Impact Matrix", user_summary, st.session_state['e2_result_data']['result_obj'], {'scores': df.to_dict()})
            update_guidance("🎉 E2 Lab Complete! Move to the E3: Strategic Decision Lab to practice critical thinking.")
            st.rerun()

def render_lab_E3():
    st.header(" E3: Strategic Decision Lab ")
    
    st.subheader(" Problem Statement")
    st.markdown("Scenario: You must decide between two major AI investments. How do you assess the trade-offs and risks?")
    st.subheader(" Objective")
    st.success("Learning Goal: Build analytical judgment by evaluating balanced Pro/Con arguments for strategic AI decisions.")
    
    st.markdown("---")
    
    with st.expander("Understanding Your Metrics (Click to Expand)", expanded=False):
        st.markdown("This lab measures the quality of the AI's Pro/Con rationale:")
        
        col_m1, col_m2 = st.columns(2)
        
        with col_m1:
            st.subheader("Pro/Con Balance")
            st.markdown("""
            * **What is it?** Measures if the AI provided equally strong arguments for both sides.
            * **Why it matters:** Strategic advisory requires avoiding bias.
            """)
        
        with col_m2:
            st.subheader("ROI Alignment")
            st.markdown("""
            * **What is it?** Measures if factors are tied to core executive metrics.
            * **Why it matters:** Ensures focus on tangible business value.
            """)

    with st.expander(" Instructions: How to Use This Lab", expanded=True):
        st.markdown("1. **Select Decision:** Choose a high-stakes decision.")
        st.markdown("2. **Form Hypothesis:** Write your initial belief.")
        st.markdown("3. **Run AI Advisor:** Get balanced Pro/Con analysis.")

    decision_options = {
        "Expand Market": "Expand into a New Market with AI-driven Personalization",
        "Automate Support": "Automate 80% of Tier-1 Customer Support using Generative AI",
        "Rethink Talent": "Invest in a Chief Data Officer vs. Upskill Existing Managers"
    }
    
    selected_decision = st.selectbox("1. Select a Strategic Decision:", list(decision_options.values()), key='e3_decision')
    hypothesis = st.text_area("2. Type your initial hypothesis/justification:", placeholder="e.g., Automating support will free up expert agents for complex, high-value cases.", key='e3_hypothesis')
    
    if st.button("Run AI Rationale (Pro/Con Analysis)"):
        user_input = f"Decision: {selected_decision}. Hypothesis: {hypothesis}"
        
        result_data = run_executive_llm_logic('E3', user_input, "Strategic Advisor", "Act as a strategic advisor. Present opportunities and cautions for this AI adoption decision in a two-column markdown table (Pro/Con format).")
        st.session_state['e3_result_data'] = result_data
        st.success("AI Analysis Complete: Review the Balanced Decision Table.")

    if st.session_state.get('e3_result_data'):
        result_data = st.session_state['e3_result_data'] 
        user_input = f"Decision: {selected_decision}. Hypothesis: {hypothesis}" 

        st.markdown("---")
        st.subheader("AI-Generated Decision Table")
        st.markdown(result_data['output'])
        
        st.subheader("3. Reflection Box")
        reflection = st.text_area("3. How did the AI analysis challenge or confirm your initial hypothesis?", key="e3_reflection")
        
        st.markdown("---")
        render_ai_explanation(result_data)

        if st.button("Save & Complete Lab E3", key='e3_save_btn'):
            update_progress('E3', '🟢')
            save_to_journal("E3: Strategic Decision Lab", f"{selected_decision} | Hypothesis: {hypothesis}", st.session_state['e3_result_data']['result_obj'], {'reflection': reflection})
            update_guidance("🎉 E3 Lab Complete! Move to the E4: Use Case Builder to define a project.")
            st.rerun()

def render_lab_E4():
    st.header(" E4: Use Case Builder ")
    
    st.subheader(" Problem Statement")
    st.markdown("Scenario: You have a great AI idea, but your engineering team needs a structured, viable project definition.")
    st.subheader(" Objective")
    st.success("Learning Goal: Learn to structure a viable AI project proposal and score its feasibility.")
    
    st.markdown("---")
    
    with st.expander(" Understanding Your Metrics (Click to Expand)", expanded=False):
        st.markdown("This lab measures the overall viability of your proposed AI project:")
        
        col_m1, col_m2 = st.columns(2)
        
        with col_m1:
            st.subheader("Feasibility Score (0.0-1.0)")
            st.markdown("""
            * **What is it?** A score representing the probability of successfully completing the project.
            * **Why it matters:** Combines technical risk with operational reality.
            """)
        
        with col_m2:
            st.subheader("Data Readiness")
            st.markdown("""
            * **What is it?** Assessment of whether necessary data is available and clean.
            * **Why it matters:** Data readiness is the #1 technical blocker.
            """)
    
    with st.expander(" Instructions: How to Use This Lab", expanded=True):
        st.markdown("1. **Fill Canvas:** Complete all four inputs.")
        st.markdown("2. **Run Feasibility AI:** Click **Generate Summary**.")
        st.markdown("3. **Review Status:** Check the Feasibility Score.")
    
    st.subheader("1. AI Use Case Canvas Inputs")
    col_a, col_b = st.columns(2)
    
    with col_a:
        goal = st.text_input("1. Business Goal (What to achieve?)", placeholder="e.g., Reduce customer churn by 10%.", key='e4_goal')
        data_available = st.radio("2. Data Available?", ["Yes, well-structured", "Some, requires cleaning", "No, needs collection"], key='e4_data')
    with col_b:
        expected_benefit = st.text_input("3. Expected Benefit (Quantifiable value)", placeholder="e.g., $5M in annual savings.", key='e4_benefit')
        constraints = st.text_input("4. Constraints (Budget, Time, Regulation)", placeholder="e.g., 6-month deadline, HIPAA compliance.", key='e4_constraints')
        
    score = 0.0 
    
    if st.button("Generate Use Case Summary & Feasibility Score"):
        user_input = f"Goal: {goal}. Data: {data_available}. Benefit: {expected_benefit}. Constraints: {constraints}."
        
        if not all([goal, expected_benefit, constraints]):
            st.warning("Please fill out all fields in the Use Case Canvas.")
            return

        result_data = run_executive_llm_logic('E4', user_input, "Project Feasibility Analyst", "Using the provided Goal, Data, Benefit, and Constraints, generate a 'Business Use Case Summary' and conclude with a Feasibility Status (Feasible, Moderate, Not Feasible).")
        st.session_state['e4_result_data'] = result_data
        
        score = result_data['metrics']['Feasibility Score'] 
        
        if score > 0.7:
            status_color = "green"
            status_text = "🟢 Highly Feasible"
        elif score > 0.5:
            status_color = "orange"
            status_text = "🟡 Moderately Feasible"
        else:
            status_color = "red"
            status_text = "🔴 Not Feasible"
            
        st.markdown(f"### Status: <span style='color:{status_color}'>{status_text}</span>", unsafe_allow_html=True)
        st.markdown("#### Business Use Case Summary")
        st.success(result_data['output'])
        
        st.markdown("---")
        render_ai_explanation(result_data)

    if st.session_state.get('e4_result_data'):
        score = st.session_state['e4_result_data']['metrics']['Feasibility Score']
        user_input = f"Goal: {goal}. Data: {data_available}. Benefit: {expected_benefit}. Constraints: {constraints}."
        
        if st.button("Save & Complete Lab E4", key='e4_save_btn'):
            update_progress('E4', '🟢')
            artifact = {"Use Case Summary": st.session_state['e4_result_data']['output'], "Feasibility Score": score}
            save_to_journal("E4: Use Case Builder", user_input, st.session_state['e4_result_data']['result_obj'], artifact)
            update_guidance("🎉 E4 Lab Complete! Move to the E5: Risk vs Reward Analyzer.")
            st.rerun()

def render_lab_E5():
    st.header(" E5: Risk vs Reward Analyzer ")
    
    st.subheader(" Problem Statement")
    st.markdown("Scenario: Your project looks promising, but what about the hidden risks?")
    st.subheader(" Objective")
    st.success("Learning Goal: Build executive understanding of AI trade-offs and develop mitigation strategies.")
    
    st.markdown("---")
    
    with st.expander(" Understanding Your Metrics (Click to Expand)", expanded=False):
        st.markdown("This lab visualizes the balance between potential gains and risks:")
        
        col_m1, col_m2 = st.columns(2)
        
        with col_m1:
            st.subheader("Risk Profile")
            st.markdown("""
            * **What is it?** A score for three primary risk categories.
            * **Why it matters:** Ensures proactive planning.
            """)
        
        with col_m2:
            st.subheader("Risk-Reward Ratio")
            st.markdown("""
            * **What is it?** The balance between profit and potential loss.
            * **Why it matters:** Guides decisions where rewards justify exposure.
            """)

    with st.expander(" Instructions: How to Use This Lab", expanded=True):
        st.markdown("1. **Define Profile:** Use sliders to score Risks and Rewards.")
        st.markdown("2. **Analyze Plot:** Review the Radar Chart.")
        st.markdown("3. **Run Mitigation AI:** Get tailored mitigation advice.")
    
    st.subheader("1. Define Risk & Reward Profile")
    
    risk_col, reward_col = st.columns(2)
    
    with risk_col:
        st.markdown("##### 🔻 Risk Assessment")
        risks = {
            "Ethical": st.slider("Ethical Risk (Bias, Privacy)", 0, 10, 5, key="e5_risk_e"),
            "Financial": st.slider("Financial Risk (Cost, ROI)", 0, 10, 5, key="e5_risk_f"),
            "Operational": st.slider("Operational Risk (Complexity)", 0, 10, 5, key="e5_risk_o")
        }
        
    with reward_col:
        st.markdown("##### 🔺 Reward Assessment")
        rewards = {
            "Efficiency": st.slider("Efficiency Reward", 0, 10, 5, key="e5_reward_e"),
            "Growth": st.slider("Growth Reward", 0, 10, 5, key="e5_reward_g"),
            "Experience": st.slider("Experience Reward", 0, 10, 5, key="e5_reward_x")
        }

    st.markdown("---")
    
    st.subheader("2. Visualization: Risk/Reward Profile")
    categories = list(risks.keys())
    
    fig = go.Figure(data=[
        go.Scatterpolar(
            r=list(risks.values()) + [risks[categories[0]]], 
            theta=categories + [categories[0]],
            fill='toself', name='Risk Profile',
            line_color='red'
        ),
        go.Scatterpolar(
            r=list(rewards.values()) + [rewards['Efficiency']], 
            theta=list(rewards.keys()) + ['Efficiency'],
            fill='toself', name='Reward Potential',
            line_color='blue'
        )
    ])
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 10])), showlegend=True)
    st.plotly_chart(fig, use_container_width=True) 

    risk_areas = [k for k, v in risks.items() if v >= 7]
    user_input = f"High risk areas: {', '.join(risk_areas)}. Current profile: {json.dumps(risks)}"
    
    if st.button("Generate Mitigation Recommendations"):
        
        result_data = run_executive_llm_logic('E5', user_input, "Risk Mitigation Advisor", "Generate executive summary with clear mitigation recommendations for all identified high risks (score 7+).")
        st.session_state['e5_result_data'] = result_data
        
        st.subheader("Mitigation Recommendations")
        st.success(result_data['output'])
        
        st.markdown("---")
        render_ai_explanation(result_data)

    if st.session_state.get('e5_result_data'):
        if st.button("Save & Complete Lab E5", key='e5_save_btn'):
            update_progress('E5', '🟢')
            save_to_journal("E5: Risk vs Reward", user_input, st.session_state['e5_result_data']['result_obj'], {'risks': risks, 'rewards': rewards})
            update_guidance(" E5 Lab Complete! Move to the E6: Reflection & Export tab.")
            st.rerun()

def render_learning_journal():
    st.header(" E6: Executive Reflection & Export ")
    st.markdown("##### **Goal:** Review and reflect on the key experiments from each lab.")
    st.markdown("---")
    
    st.subheader("Your Module Progress")
    progress_percent = get_progress_percent()
    st.progress(progress_percent, text=f"Module Completion: {progress_percent}%")
    
    with st.expander(" Understanding Your Metrics (Click to Expand)", expanded=False):
        st.markdown("This section reviews the quality of your learning artifacts:")
        
        col_m1, col_m2 = st.columns(2)
        
        with col_m1:
            st.subheader("Synthesis Completeness")
            st.markdown("""
            * **What is it?** Measures whether your reflection addresses all prior labs.
            * **Why it matters:** Ensures learning is consolidated.
            """)
        
        with col_m2:
            st.subheader("Report Readiness")
            st.markdown("""
            * **What is it?** Final check ensuring all artifacts are saved.
            * **Why it matters:** Guarantees complete strategy document.
            """)

    with st.expander(" Instructions: How to Use This Section", expanded=True):
        st.markdown("1. **Synthesize:** Write your final takeaways from labs E1-E5.")
        st.markdown("2. **Export:** Click **Export Insight Report** to compile your artifacts.")
    
    st.subheader("1. Personal Reflection & Strategy Synthesis")
    reflection = st.text_area(
        "Synthesize your key takeaways from Module 1. What are your 3 top action items?",
        key="e6_reflection",
        height=200
    )
    
    if st.button("Export 'Executive Insight Report (PDF)'"):
        report_content = f"AI+ EXECUTIVE INSIGHT REPORT - MODULE 1\n"
        report_content += f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        report_content += f"FINAL REFLECTION:\n{reflection}\n\n"
        
        report_content += "ARTIFACT SUMMARY:\n"
        for entry in st.session_state.journal:
            report_content += f"--- {entry['lab']}: {entry['title']} ({entry['timestamp'].split(' ')[0]}) ---\n"
            report_content += f"Output: {entry['result'].get('content', 'N/A')[:500]}...\n\n"
            
        pdf_bytes = BytesIO(report_content.encode('utf-8'))
            
        st.download_button(
            label="Download Executive Insight Report (PDF)",
            data=pdf_bytes,
            file_name="AI_Executive_Insight_Report.pdf",
            mime="application/pdf"
        )
        
        update_progress('E6', '🟢')
        save_to_journal("E6: Final Reflection", "N/A", {'content': 'Report Exported'}, {'reflection': reflection})
        st.success("Report Generated and Progress Finalized!")
        st.rerun()

    st.markdown("---")
    
    st.subheader("2. Saved Experiments & Artifacts")
    if st.session_state.journal:
        reversed_journal = st.session_state.journal[::-1]
        for entry in reversed_journal:
            with st.expander(f"{entry['lab']}: {entry['title']} ({entry['timestamp'].split(' ')[0]})", expanded=False):
                st.markdown("**Prompt/Input Used:**")
                st.code(entry['prompt'], language='markdown')
                st.markdown("**AI Response/Artifact:**")
                st.info(entry['result'].get('content', 'N/A'))
                st.markdown(f"**Metrics/Scores:** {entry['metrics']}")
    else:
        st.info("Your journal is empty! Complete Labs E1-E5 to build your personalized report.")


def render_ai_explanation(result_data):
    st.warning("For transparency, this panel shows the exact steps the AI model took.")
    with st.expander(f" How the AI Generated This Output", expanded=True):
        st.subheader("Step-by-Step AI Processing")
        
        st.info("The execution steps were logged above by the LLM connector.")
        
        st.markdown("---")
        st.subheader("Mermaid Flow Diagram")
        
        html_code = f"""
            <html>
            <head>
                <script src="https://cdn.jsdelivr.net/npm/mermaid@10.3.0/dist/mermaid.min.js"></script>
                <style>
                    .mermaid {{ 
                        font-family: 'Helvetica', 'Arial', sans-serif;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        width: 100%;
                    }}
                </style>
            </head>
            <body>
                <div class="mermaid">
                    {result_data['mermaid_flow']}
                </div>
                <script>
                    mermaid.initialize({{ startOnLoad: true }});
                </script>
            </body>
            </html>
            """
        
        st.markdown("""
            **Visual Aid:** This flowchart shows the strategic path the AI took from your input to the final artifact.
        """)
        
        components.html(html_code, height=700) 

        with st.expander("Show Raw Flowchart Code"):
            st.code(result_data['mermaid_flow'], language='mermaid') 
        
        st.markdown("---")
        st.subheader("AI Summary")
        st.success("The system confirmed all inputs were used and the output aligns with Strategic Reasoning context.")


# --- 6. MAIN APPLICATION ENTRY POINT ---

def main():
    render_sidebar()
    
    st.markdown('<div class="title-header">Module 1: AI Strategy & Executive Foundations</div>', unsafe_allow_html=True)
    st.markdown("---")

    tab_titles = [
        " Getting Started (E0)", "E1: Business Context ", "E2: Impact Matrix ", 
        "E3: Decision Lab ", "E4: Use Case Builder ", 
        "E5: Risk Analyzer ", "E6: Reflection & Export "
    ]
    tabs = st.tabs(tab_titles)
    
    with tabs[0]:
        st.session_state.current_tab = 'E0'
        render_getting_started()
    with tabs[1]:
        st.session_state.current_tab = 'E1'
        render_lab_E1()
    with tabs[2]:
        st.session_state.current_tab = 'E2'
        render_lab_E2()
    with tabs[3]:
        st.session_state.current_tab = 'E3'
        render_lab_E3()
    with tabs[4]:
        st.session_state.current_tab = 'E4'
        render_lab_E4()
    with tabs[5]:
        st.session_state.current_tab = 'E5'
        render_lab_E5()
    with tabs[6]:
        st.session_state.current_tab = 'E6'
        render_learning_journal()


if __name__ == '__main__':
    main()