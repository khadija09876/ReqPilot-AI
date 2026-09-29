import os
from datetime import datetime

import streamlit as st
from crewai import Agent, Crew, Process, Task, LLM

# ============================================================
# REQPILOT AI
# Autonomous Software Requirements Engineering System
# CrewAI + Groq + Streamlit
# ============================================================

st.set_page_config(
    page_title="ReqPilot AI",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """ <style>
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
# SESSION STATE
# ============================================================

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "last_project" not in st.session_state:
    st.session_state.last_project = ""

if "last_run_time" not in st.session_state:
    st.session_state.last_run_time = None

# ============================================================
# API KEY (SAFE GETTER FIX)
# ============================================================

def get_api_key():
    """
    Safely retrieves GROQ_API_KEY without attribute error on st.secrets.
    """
    # Check environment variable first
    env_key = os.getenv("GROQ_API_KEY", "")
    if env_key:
        return env_key

    # Check Streamlit secrets safely
    try:
        if hasattr(st, "secrets") and st.secrets:
            if "GROQ_API_KEY" in st.secrets:
                return str(st.secrets["GROQ_API_KEY"])
    except Exception:
        pass

    return ""

# ============================================================
# LLM (GROQ COMPATIBLE)
# ============================================================

def create_llm(api_key):
    """
    CrewAI LLM configured to use Groq.
    cache=False prevents prompt cache errors on Groq endpoints.
    """
    return LLM(
        model="groq/llama-3.3-70b-versatile",
        api_key=api_key,
        temperature=0.2,
        cache=False,
    )

# ============================================================
# AGENTS
# ============================================================

def create_agents(llm):
    business_analyst = Agent(
        role="Senior Business Analyst",
        goal=(
            "Transform raw business ideas into precise, structured and "
            "implementation-ready business requirements."
        ),
        backstory=(
            "You are a senior business analyst experienced in software "
            "requirements engineering, stakeholder analysis, process "
            "mapping and business rule definition. You identify ambiguity "
            "instead of inventing missing information."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    technical_architect = Agent(
        role="Senior Technical Requirements Architect",
        goal=(
            "Convert validated business requirements into clear technical "
            "requirements and non-functional requirements."
        ),
        backstory=(
            "You are a senior software architect who specializes in "
            "requirements analysis, system architecture, APIs, data, "
            "security, scalability, integrations and non-functional "
            "requirements."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    product_analyst = Agent(
        role="Senior Product Analyst",
        goal=(
            "Translate the analyzed requirements into user-centered "
            "product behavior, user stories and acceptance criteria."
        ),
        backstory=(
            "You are an experienced product analyst specializing in "
            "personas, user journeys, user stories, acceptance criteria, "
            "MVP definition and edge-case analysis."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    qa_reviewer = Agent(
        role="Senior QA Requirements Reviewer",
        goal=(
            "Audit the complete requirements package for ambiguity, "
            "missing requirements, contradictions, testability and "
            "implementation risks."
        ),
        backstory=(
            "You are a senior QA and requirements quality reviewer. "
            "You challenge unclear requirements, identify gaps and "
            "produce actionable corrections and test scenarios."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    return (
        business_analyst,
        technical_architect,
        product_analyst,
        qa_reviewer,
    )

# ============================================================
# CREW WORKFLOW
# ============================================================

def run_requirements_crew(
    project_name,
    business_domain,
    raw_requirement,
    llm,
):
    (
        business_analyst,
        technical_architect,
        product_analyst,
        qa_reviewer,
    ) = create_agents(llm)

    # --------------------------------------------------------
    # TASK 1 — BUSINESS ANALYSIS
    # --------------------------------------------------------

    business_task = Task(
        description=f"""
Analyze the following software project requirement.

PROJECT NAME:
{project_name}

BUSINESS DOMAIN:
{business_domain}

RAW REQUIREMENT:
{raw_requirement}

Produce a structured business requirements analysis.

Cover:
1. Business objective
2. Problem statement
3. Business value
4. Stakeholders
5. Primary users
6. Secondary users
7. System actors
8. Core business workflows
9. Functional requirements
10. Business rules
11. Inputs
12. Outputs
13. Assumptions
14. Ambiguities
15. Missing information
16. Important clarification questions

Important:
- Do not invent facts that are not supported by the raw requirement.
- Clearly mark assumptions.
- Make every requirement specific and actionable.
- Use professional requirements-engineering terminology.
""",
        expected_output=(
            "A detailed and structured business requirements analysis "
            "containing functional requirements, stakeholders, workflows, "
            "business rules, assumptions, ambiguities and clarification questions."
        ),
        agent=business_analyst,
    )

    # --------------------------------------------------------
    # TASK 2 — TECHNICAL ANALYSIS
    # --------------------------------------------------------

    technical_task = Task(
        description="""
Using the Business Analyst's output as your primary context, create a
technical requirements analysis.

Review the previous analysis and produce:
1. System capabilities
2. Technical requirements
3. Functional-to-technical mapping
4. Authentication requirements
5. Authorization requirements
6. Data requirements
7. API requirements
8. External integration requirements
9. Security requirements
10. Privacy considerations
11. Performance requirements
12. Availability requirements
13. Scalability requirements
14. Reliability requirements
15. Maintainability requirements
16. Observability requirements
17. Dependency considerations
18. Technical risks
19. Technical assumptions
20. Open technical questions

Do not select a technology merely because it is popular.
Only identify technology constraints that are actually supported by
the requirement. Clearly distinguish confirmed requirements from
recommended considerations.
""",
        expected_output=(
            "A structured technical requirements document covering "
            "functional technical requirements, NFRs, security, data, "
            "integrations, risks and open technical questions."
        ),
        agent=technical_architect,
        context=[business_task],
    )

    # --------------------------------------------------------
    # TASK 3 — PRODUCT ANALYSIS
    # --------------------------------------------------------

    product_task = Task(
        description="""
Using the Business Analyst and Technical Analyst outputs, define the
product-level requirements.

Produce:
1. User personas
2. Persona goals
3. Main user journeys
4. End-to-end workflows
5. User stories
6. Acceptance criteria
7. Edge cases
8. Error scenarios
9. Usability requirements
10. Accessibility considerations
11. MVP scope
12. Post-MVP enhancement candidates
13. Product success indicators
14. User-facing validation rules
15. Important product decisions still requiring clarification

User stories should use this format where appropriate:
As a [user],
I want [capability],
so that [outcome].

Acceptance criteria must be testable and concrete.

Do not invent business facts. Where information is missing,
identify it explicitly.
""",
        expected_output=(
            "A product requirements analysis containing personas, journeys, "
            "user stories, testable acceptance criteria, edge cases, "
            "MVP scope and unresolved product questions."
        ),
        agent=product_analyst,
        context=[business_task, technical_task],
    )

    # --------------------------------------------------------
    # TASK 4 — QA REQUIREMENTS AUDIT
    # --------------------------------------------------------

    qa_task = Task(
        description="""
Audit the complete requirements package produced by the previous agents.

Perform a rigorous requirements-quality review.

Check for:
1. Missing requirements
2. Ambiguous requirements
3. Contradictory requirements
4. Duplicate requirements
5. Untestable requirements
6. Weak acceptance criteria
7. Missing edge cases
8. Missing error handling
9. Security gaps
10. Data/privacy gaps
11. Performance gaps
12. Integration risks
13. Implementation risks
14. Unclear assumptions
15. Missing stakeholder decisions
16. Requirements that require clarification

Then produce:
A. Requirements quality findings
B. Critical gaps
C. Recommended corrections
D. QA test scenarios
E. Final clarification questions
F. Readiness assessment

The readiness assessment must be one of:
READY FOR DEVELOPMENT
or
NEEDS CLARIFICATION

Do not declare something ready if critical ambiguity remains.
Do not invent information to make the project appear complete.
""",
        expected_output=(
            "A rigorous QA audit with defects, gaps, corrections, "
            "test scenarios, clarification questions and a final "
            "development-readiness assessment."
        ),
        agent=qa_reviewer,
        context=[business_task, technical_task, product_task],
    )

    # --------------------------------------------------------
    # CREW
    # --------------------------------------------------------

    crew = Crew(
        agents=[
            business_analyst,
            technical_architect,
            product_analyst,
            qa_reviewer,
        ],
        tasks=[
            business_task,
            technical_task,
            product_task,
            qa_task,
        ],
        process=Process.sequential,
        verbose=False,
    )

    result = crew.kickoff()
    return result

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## ◆ ReqPilot AI")
    st.caption("Autonomous Requirements Engineering")

    st.divider()

    st.markdown("### Multi-Agent Pipeline")

    agents_info = [
        (
            "01",
            "Business Analyst",
            "Business objectives, actors, workflows and functional requirements.",
        ),
        (
            "02",
            "Technical Architect",
            "Technical requirements, security, data and non-functional requirements.",
        ),
        (
            "03",
            "Product Analyst",
            "Personas, user journeys, stories and acceptance criteria.",
        ),
        (
            "04",
            "QA Reviewer",
            "Quality audit, gaps, testability and readiness assessment.",
        ),
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

    st.markdown("### AI Infrastructure")
    st.write("**Framework:** CrewAI")
    st.write("**LLM Provider:** Groq")
    st.write("**Model:** Llama 3.3 70B Versatile")
    st.write("**Interface:** Streamlit")
    st.write("**Deployment:** Streamlit Community Cloud")

    st.divider()

    st.caption("API keys are read from Streamlit Secrets.")

# ============================================================
# HERO
# ============================================================

st.markdown(
    """ <div class="hero"> <h1>◆ ReqPilot AI</h1> <p>
Autonomous Software Requirements Engineering System <br>
Transform raw software ideas into structured, auditable and
implementation-ready requirements using a collaborative
multi-agent AI workflow. </p> </div>
""",
    unsafe_allow_html=True,
)

# ============================================================
# INTRO
# ============================================================

st.markdown(
    """ <div class="section-title">Requirements Analysis Workspace</div>
""",
    unsafe_allow_html=True,
)

st.write(
    "Describe the software product or business problem below. "
    "ReqPilot AI will route the requirement through four specialized "
    "CrewAI agents."
)

# ============================================================
# INPUT FORM
# ============================================================

col1, col2 = st.columns(2)

with col1:
    project_name = st.text_input(
        "Project Name",
        placeholder="e.g. Employee Leave Management System",
    )

with col2:
    business_domain = st.text_input(
        "Business Domain",
        placeholder="e.g. Human Resources / Enterprise Software",
    )

raw_requirement = st.text_area(
    "Raw Business / Software Requirement",
    height=230,
    placeholder=(
        "Example:\n\n"
        "Our company needs a centralized employee leave management "
        "system where employees can submit leave requests, managers "
        "can approve or reject requests, HR can manage policies and "
        "administrators can monitor leave records and reports."
    ),
)

# ============================================================
# RUN BUTTON
# ============================================================

run_analysis = st.button(
    "◆ Run Autonomous Requirements Analysis",
    type="primary",
    use_container_width=True,
)

# ============================================================
# EXECUTION
# ============================================================

if run_analysis:
    api_key = get_api_key()

    if not api_key:
        st.error(
            "GROQ_API_KEY is missing. Add GROQ_API_KEY to "
            "Streamlit Cloud → Settings → Secrets."
        )
        st.stop()

    if not project_name.strip():
        st.warning("Please enter a project name.")
        st.stop()

    if not business_domain.strip():
        st.warning("Please enter the business domain.")
        st.stop()

    if not raw_requirement.strip():
        st.warning("Please enter the software/business requirement.")
        st.stop()

    st.session_state.analysis_result = None

    progress_box = st.empty()

    try:
        progress_box.markdown(
            """
            <div class="status-box">
                <strong>AI pipeline started.</strong><br>
                CrewAI is executing the requirements agents sequentially.
            </div>
            """,
            unsafe_allow_html=True,
        )

        llm = create_llm(api_key)

        with st.spinner(
            "Business Analyst → Technical Architect → Product Analyst → QA Reviewer"
        ):
            result = run_requirements_crew(
                project_name=project_name.strip(),
                business_domain=business_domain.strip(),
                raw_requirement=raw_requirement.strip(),
                llm=llm,
            )

        final_report = str(result)

        st.session_state.analysis_result = final_report
        st.session_state.last_project = project_name.strip()
        st.session_state.last_run_time = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        progress_box.empty()

        st.success(
            "Requirements analysis completed successfully."
        )

    except Exception as error:
        progress_box.empty()

        st.error(
            "The CrewAI workflow could not be completed."
        )

        with st.expander("Technical error details"):
            st.code(str(error))

# ============================================================
# RESULTS
# ============================================================

if st.session_state.analysis_result:
    st.divider()

    st.markdown(
        '<div class="section-title">Final Requirements Specification</div>',
        unsafe_allow_html=True,
    )

    metric1, metric2, metric3 = st.columns(3)

    with metric1:
        st.markdown(
            """
            <div class="metric-box">
                <div class="metric-value">4</div>
                <div class="metric-label">AI Agents Executed</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with metric2:
        st.markdown(
            """
            <div class="metric-box">
                <div class="metric-value">Sequential</div>
                <div class="metric-label">CrewAI Process</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with metric3:
        st.markdown(
            """
            <div class="metric-box">
                <div class="metric-value">Groq</div>
                <div class="metric-label">LLM Infrastructure</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    st.markdown(
        f"**Project:** {st.session_state.last_project}"
    )

    if st.session_state.last_run_time:
        st.caption(
            f"Analysis completed: {st.session_state.last_run_time}"
        )

    st.markdown("### Requirements Report")

    st.markdown(st.session_state.analysis_result)

    st.divider()

    st.download_button(
        label="↓ Download Requirements Report",
        data=st.session_state.analysis_result,
        file_name="reqpilot_requirements_report.txt",
        mime="text/plain",
        use_container_width=True,
    )

# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """ <div class="footer">
ReqPilot AI · Autonomous Software Requirements Engineering System <br>
CrewAI × Groq × Streamlit </div>
""",
    unsafe_allow_html=True,
)
