import os
from datetime import datetime

import streamlit as st
from crewai import Agent, Crew, Process, Task, LLM

# ============================================================
# LITELLM / GROQ COMPATIBILITY FIX
# Prevents 'cache_breakpoint is unsupported' error from Groq
# ============================================================
os.environ["DISABLE_PROMPT_CACHING"] = "TRUE"

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="ReqPilot AI",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM STYLING (CSS)
# ============================================================
st.markdown(
    """
    <style>
    .stApp {
        background:
        radial-gradient(circle at 10% 0%, rgba(37, 99, 235, 0.08), transparent 28%),
        radial-gradient(circle at 90% 10%, rgba(139, 92, 246, 0.07), transparent 25%),
        #f8fafc;
    }

    .hero {
        background: linear-gradient(135deg, #0f172a 0%, #111827 60%, #1e293b 100%);
        padding: 34px 38px;
        border-radius: 22px;
        margin-bottom: 26px;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 18px 45px rgba(15, 23, 42, 0.16);
    }

    .hero h1 {
        color: white;
        font-size: 38px;
        margin: 0;
        font-weight: 800;
        letter-spacing: -1px;
    }

    .hero p {
        color: #cbd5e1;
        font-size: 16px;
        margin-top: 10px;
        line-height: 1.6;
    }

    .agent-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 15px;
        padding: 15px;
        margin-bottom: 12px;
        box-shadow: 0 5px 18px rgba(15,23,42,0.05);
    }

    .agent-number {
        color: #2563eb;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    .agent-title {
        color: #0f172a;
        font-size: 16px;
        font-weight: 750;
        margin-top: 3px;
    }

    .agent-description {
        color: #64748b;
        font-size: 12px;
        margin-top: 4px;
    }

    .section-title {
        color: #0f172a;
        font-size: 22px;
        font-weight: 800;
        margin-top: 12px;
        margin-bottom: 8px;
    }

    .status-box {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        color: #1e40af;
        padding: 13px 16px;
        border-radius: 12px;
        margin: 12px 0;
    }

    .metric-box {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 17px;
        text-align: center;
    }

    .metric-value {
        color: #0f172a;
        font-size: 24px;
        font-weight: 800;
    }

    .metric-label {
        color: #64748b;
        font-size: 12px;
        margin-top: 4px;
    }

    div[data-testid="stTextArea"] textarea {
        border-radius: 12px;
    }

    div[data-testid="stTextInput"] input {
        border-radius: 10px;
    }

    .footer {
        color: #94a3b8;
        text-align: center;
        font-size: 12px;
        margin-top: 35px;
        padding: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "last_project" not in st.session_state:
    st.session_state.last_project = ""

if "last_run_time" not in st.session_state:
    st.session_state.last_run_time = None

# ============================================================
# STREAMLIT DEPLOYMENT SAFE API KEY RETRIEVAL
# Prevents 'bool' object has no attribute 'get' crash
# ============================================================
def get_api_key():
    # 1. Environment Variable Check
    env_key = os.getenv("GROQ_API_KEY", "")
    if env_key and isinstance(env_key, str) and len(env_key.strip()) > 0:
        return env_key.strip()

    # 2. Streamlit Cloud Secrets Check
    try:
        if "GROQ_API_KEY" in st.secrets:
            key_val = st.secrets["GROQ_API_KEY"]
            if key_val and isinstance(key_val, str) and len(key_val.strip()) > 0:
                return key_val.strip()
    except Exception:
        pass

    return ""

# ============================================================
# LLM CREATION (GROQ COMPATIBLE)
# ============================================================
def create_llm(api_key):
    os.environ["GROQ_API_KEY"] = api_key

    return LLM(
        model="groq/llama-3.3-70b-versatile",
        api_key=api_key,
        temperature=0.2,
        cache=False,  # Caching disabled for Groq API compatibility
    )

# ============================================================
# AGENTS CREATION
# ============================================================
def create_agents(llm):
    business_analyst = Agent(
        role="Senior Business Analyst",
        goal="Transform raw business ideas into structured and implementation-ready requirements.",
        backstory="Experienced in software requirements engineering, process mapping, and rule definition.",
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    technical_architect = Agent(
        role="Senior Technical Requirements Architect",
        goal="Convert business requirements into technical specs, architecture, and non-functional requirements.",
        backstory="Specializes in system architecture, APIs, data modeling, security, and scalability.",
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    product_analyst = Agent(
        role="Senior Product Analyst",
        goal="Translate requirements into user journeys, user stories, and testable acceptance criteria.",
        backstory="Expert in user story mapping, acceptance criteria formulation, and MVP defining.",
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    qa_reviewer = Agent(
        role="Senior QA Requirements Reviewer",
        goal="Audit requirements for ambiguity, missing edge-cases, and produce test scenarios.",
        backstory="Senior QA lead who ensures requirement testability and development readiness.",
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    return business_analyst, technical_architect, product_analyst, qa_reviewer

# ============================================================
# WORKFLOW EXECUTION
# ============================================================
def run_requirements_crew(project_name, business_domain, raw_requirement, llm):
    ba_agent, tech_agent, prod_agent, qa_agent = create_agents(llm)

    business_task = Task(
        description=f"""
Analyze the project requirement:
PROJECT NAME: {project_name}
BUSINESS DOMAIN: {business_domain}
RAW REQUIREMENT: {raw_requirement}

Produce a structured business requirements analysis covering:
1. Business Objective & Problem Statement
2. Stakeholders & Users
3. Functional Requirements
4. Core Business Rules & Workflows
5. Assumptions, Ambiguities & Open Questions
""",
        expected_output="Structured business requirements document.",
        agent=ba_agent,
    )

    technical_task = Task(
        description="""
Using the Business Analyst output, produce technical requirements covering:
1. System Capabilities & Architecture
2. Data & API Requirements
3. Security, Authentication & Privacy
4. Non-Functional Requirements (Performance, Scalability)
5. Technical Risks & Open Technical Questions
""",
        expected_output="Detailed technical requirements specification.",
        agent=tech_agent,
        context=[business_task],
    )

    product_task = Task(
        description="""
Based on the previous outputs, generate:
1. User Personas & Main Journeys
2. User Stories (As a... I want... So that...)
3. Testable Acceptance Criteria
4. Edge Cases & Error Scenarios
5. MVP Scope vs Future Scope
""",
        expected_output="Product requirement document with user stories and acceptance criteria.",
        agent=prod_agent,
        context=[business_task, technical_task],
    )

    qa_task = Task(
        description="""
Perform a comprehensive audit on the entire requirements suite:
1. Identify missing requirements, gaps, or contradictions
2. Testability audit
3. QA Test Scenarios
4. Final Readiness Assessment: (READY FOR DEVELOPMENT or NEEDS CLARIFICATION)
""",
        expected_output="Comprehensive QA audit report and readiness assessment.",
        agent=qa_agent,
        context=[business_task, technical_task, product_task],
    )

    crew = Crew(
        agents=[ba_agent, tech_agent, prod_agent, qa_agent],
        tasks=[business_task, technical_task, product_task, qa_task],
        process=Process.sequential,
        verbose=False,
    )

    return crew.kickoff()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## ◆ ReqPilot AI")
    st.caption("Autonomous Requirements Engineering")
    st.divider()

    st.markdown("### Multi-Agent Pipeline")
    agents_info = [
        ("01", "Business Analyst", "Business objectives, actors, workflows."),
        ("02", "Technical Architect", "Technical requirements, APIs, security."),
        ("03", "Product Analyst", "User stories & acceptance criteria."),
        ("04", "QA Reviewer", "Quality audit & readiness assessment."),
    ]

    for number, title, description in agents_info:
        st.markdown(
            f"""
            <div class="agent-card">
                <div class="agent-number">Agent {number}</div>
                <div class="agent-title">{title}</div>
                <div class="agent-description">{description}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("### AI Stack")
    st.write("**Framework:** CrewAI")
    st.write("**Provider:** Groq")
    st.write("**Model:** Llama 3.3 70B Versatile")

# ============================================================
# MAIN UI
# ============================================================
st.markdown(
    """
    <div class="hero">
        <h1>◆ ReqPilot AI</h1>
        <p>Autonomous Software Requirements Engineering System<br>
        Transform raw ideas into structured and auditable software specifications.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-title">Requirements Analysis Workspace</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    project_name = st.text_input("Project Name", placeholder="e.g. Leave Management System")
with col2:
    business_domain = st.text_input("Business Domain", placeholder="e.g. Human Resources")

raw_requirement = st.text_area(
    "Raw Requirement Description",
    height=200,
    placeholder="Describe your software system idea or business problem in detail...",
)

run_analysis = st.button("◆ Run Autonomous Requirements Analysis", type="primary", use_container_width=True)

# ============================================================
# EXECUTION LOGIC
# ============================================================
if run_analysis:
    api_key = get_api_key()

    if not api_key:
        st.error(
            "⚠️ GROQ_API_KEY is missing or invalid in Streamlit Secrets! "
            "Please check Streamlit Cloud Settings -> Secrets and ensure it is formatted as: GROQ_API_KEY = \"gsk_...\""
        )
        st.stop()

    if not project_name.strip() or not business_domain.strip() or not raw_requirement.strip():
        st.warning("Please complete all input fields before running analysis.")
        st.stop()

    st.session_state.analysis_result = None
    progress_box = st.empty()

    try:
        progress_box.markdown(
            """
            <div class="status-box">
                <strong>AI Workflow Triggered.</strong><br>
                Running Agents: BA → Technical Architect → Product Analyst → QA Reviewer...
            </div>
            """,
            unsafe_allow_html=True,
        )

        llm = create_llm(api_key)

        with st.spinner("Analyzing requirements via CrewAI agents..."):
            result = run_requirements_crew(
                project_name=project_name.strip(),
                business_domain=business_domain.strip(),
                raw_requirement=raw_requirement.strip(),
                llm=llm,
            )

        st.session_state.analysis_result = str(result)
        st.session_state.last_project = project_name.strip()
        st.session_state.last_run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        progress_box.empty()
        st.success("Analysis completed successfully!")

    except Exception as error:
        progress_box.empty()
        st.error("The CrewAI workflow encountered an unexpected error.")
        with st.expander("Technical Error Log"):
            st.code(str(error))

# ============================================================
# RESULTS DISPLAY
# ============================================================
if st.session_state.analysis_result:
    st.divider()
    st.markdown('<div class="section-title">Final Requirements Specification</div>', unsafe_allow_html=True)

    st.markdown(f"**Project:** {st.session_state.last_project}")
    if st.session_state.last_run_time:
        st.caption(f"Generated on: {st.session_state.last_run_time}")

    st.markdown(st.session_state.analysis_result)

    st.download_button(
        label="↓ Download Report (.txt)",
        data=st.session_state.analysis_result,
        file_name=f"{st.session_state.last_project.lower().replace(' ', '_')}_requirements.txt",
        mime="text/plain",
        use_container_width=True,
    )

# ============================================================
# FOOTER
# ============================================================
st.markdown(
    """
    <div class="footer">
        ReqPilot AI · CrewAI × Groq × Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
