```python
import os
import re
from datetime import datetime

import streamlit as st
from crewai import Agent, Crew, Process, Task, LLM


# ============================================================
# APPLICATION CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ReqPilot AI",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* =========================
       GLOBAL
       ========================= */

    .stApp {
        background: #f5f7fb;
    }

    .main .block-container {
        max-width: 1380px;
        padding-top: 28px;
        padding-bottom: 60px;
    }


    /* =========================
       SIDEBAR
       ========================= */

    section[data-testid="stSidebar"] {
        background: #0f172a;
        border-right: 1px solid #1e293b;
    }

    section[data-testid="stSidebar"] * {
        color: #e2e8f0;
    }


    /* =========================
       HERO
       ========================= */

    .hero {
        background:
            radial-gradient(
                circle at top right,
                rgba(59,130,246,0.22),
                transparent 35%
            ),
            linear-gradient(
                135deg,
                #0f172a 0%,
                #172554 100%
            );

        border-radius: 24px;
        padding: 42px 46px;
        margin-bottom: 28px;
        box-shadow:
            0 18px 45px rgba(15,23,42,0.12);
    }

    .hero-badge {
        display: inline-block;
        padding: 7px 14px;
        border-radius: 999px;
        background: rgba(255,255,255,0.10);
        color: #bfdbfe;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.8px;
        margin-bottom: 16px;
    }

    .hero-title {
        color: #ffffff;
        font-size: 48px;
        font-weight: 800;
        margin: 0;
        line-height: 1.05;
    }

    .hero-subtitle {
        color: #dbeafe;
        font-size: 20px;
        margin-top: 12px;
        font-weight: 500;
    }

    .hero-description {
        color: #94a3b8;
        font-size: 15px;
        line-height: 1.7;
        max-width: 850px;
        margin-top: 15px;
    }


    /* =========================
       SECTION TITLES
       ========================= */

    .section-title {
        font-size: 27px;
        font-weight: 800;
        color: #0f172a;
        margin-top: 20px;
        margin-bottom: 5px;
    }

    .section-subtitle {
        color: #64748b;
        font-size: 14px;
        margin-bottom: 20px;
    }


    /* =========================
       AGENT CARDS
       ========================= */

    .agent-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 21px;
        min-height: 165px;
        box-shadow:
            0 5px 16px rgba(15,23,42,0.04);
    }

    .agent-number {
        color: #94a3b8;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 1.2px;
    }

    .agent-title {
        color: #0f172a;
        font-size: 17px;
        font-weight: 750;
        margin-top: 8px;
    }

    .agent-text {
        color: #64748b;
        font-size: 13px;
        line-height: 1.55;
        margin-top: 8px;
    }


    /* =========================
       METRICS
       ========================= */

    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 15px;
        padding: 19px;
        text-align: center;
        box-shadow:
            0 5px 16px rgba(15,23,42,0.04);
    }

    .metric-number {
        color: #0f172a;
        font-size: 28px;
        font-weight: 800;
    }

    .metric-label {
        color: #64748b;
        font-size: 12px;
        margin-top: 4px;
    }


    /* =========================
       REPORT
       ========================= */

    .report-top {
        background: #0f172a;
        color: white;
        padding: 23px 27px;
        border-radius: 17px 17px 0 0;
    }

    .report-title {
        font-size: 22px;
        font-weight: 800;
    }

    .report-meta {
        color: #94a3b8;
        font-size: 12px;
        margin-top: 5px;
    }

    .report-container {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-top: none;
        padding: 28px;
        border-radius: 0 0 17px 17px;
    }


    /* =========================
       FOOTER
       ========================= */

    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 12px;
        padding-top: 35px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HERO SECTION
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-badge">
            CREWAI MULTI-AGENT REQUIREMENTS ENGINEERING
        </div>

        <div class="hero-title">
            ReqPilot AI
        </div>

        <div class="hero-subtitle">
            Autonomous Software Requirements Engineering System
        </div>

        <div class="hero-description">
            Convert raw client requirements into structured,
            development-ready specifications through a coordinated
            AI workflow powered by CrewAI and Groq.
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## ◆ ReqPilot AI")

    st.caption("Software Requirements Intelligence")

    st.markdown("---")

    st.markdown("### AI Crew")

    st.markdown(
        """
        **01 — Business Analyst**

        Identifies business objectives, actors,
        workflows and functional requirements.

        **02 — Technical Analyst**

        Defines technical, security,
        integration and non-functional requirements.

        **03 — Product Analyst**

        Creates user stories, acceptance criteria,
        workflows and edge cases.

        **04 — QA Reviewer**

        Reviews gaps, ambiguity, risks
        and requirement testability.
        """
    )

    st.markdown("---")

    st.markdown("### AI Engine")

    st.code(
        "openai/gpt-oss-120b",
        language="text"
    )

    st.markdown("---")

    st.markdown("### Platform")

    st.write("CrewAI")
    st.write("Groq API")
    st.write("Streamlit")
    st.write("Streamlit Cloud")

    st.markdown("---")

    st.caption("ReqPilot AI · v1.0")


# ============================================================
# API KEY
# ============================================================

def get_groq_api_key():
    """
    Retrieve the Groq API key.

    Priority:
    1. Streamlit Secrets
    2. Environment variable
    """

    try:

        key = st.secrets.get("GROQ_API_KEY")

        if key:
            return key

    except Exception:
        pass

    return os.getenv("GROQ_API_KEY")


# ============================================================
# GROQ LLM
# ============================================================

def create_llm():

    api_key = get_groq_api_key()

    if not api_key:
        return None

    return LLM(
        model="groq/openai/gpt-oss-120b",
        api_key=api_key,
        temperature=0.2
    )


# ============================================================
# CREATE AGENTS
# ============================================================

def create_agents(llm):

    business_agent = Agent(
        role="Senior Business Analyst",

        goal=(
            "Understand the client's actual business problem "
            "and transform the raw requirement into clear, "
            "structured and traceable business requirements."
        ),

        backstory=(
            "You are a senior business analyst working for a "
            "professional software engineering organization. "
            "You specialize in requirements discovery, business "
            "process analysis, stakeholder identification, "
            "functional requirements and business rules."
        ),

        llm=llm,
        verbose=False,
        allow_delegation=False
    )


    technical_agent = Agent(
        role="Senior Technical Requirements Analyst",

        goal=(
            "Convert business requirements into realistic "
            "technical and non-functional requirements while "
            "identifying architecture, security, data, "
            "integration and scalability considerations."
        ),

        backstory=(
            "You are a senior technical requirements analyst "
            "with experience in software architecture, APIs, "
            "databases, authentication, authorization, security, "
            "performance and system integrations."
        ),

        llm=llm,
        verbose=False,
        allow_delegation=False
    )


    product_agent = Agent(
        role="Senior Product Analyst",

        goal=(
            "Translate business and technical requirements "
            "into user-centered product specifications that "
            "developers and stakeholders can understand."
        ),

        backstory=(
            "You are an experienced product analyst specializing "
            "in user journeys, personas, product workflows, "
            "user stories, acceptance criteria, usability "
            "requirements and edge cases."
        ),

        llm=llm,
        verbose=False,
        allow_delegation=False
    )


    qa_agent = Agent(
        role="Senior QA Requirements Reviewer",

        goal=(
            "Perform a rigorous quality review of the complete "
            "requirements specification and identify missing, "
            "ambiguous, contradictory, risky or untestable "
            "requirements."
        ),

        backstory=(
            "You are a senior QA requirements specialist. "
            "You review software requirements before development "
            "to identify quality issues, edge cases, risks, "
            "testability problems and clarification needs."
        ),

        llm=llm,
        verbose=False,
        allow_delegation=False
    )


    return (
        business_agent,
        technical_agent,
        product_agent,
        qa_agent
    )


# ============================================================
# RUN CREW
# ============================================================

def run_requirements_crew(
    project_name,
    business_domain,
    requirement
):

    llm = create_llm()

    if llm is None:

        raise RuntimeError(
            "GROQ_API_KEY is not configured. "
            "Please configure it in Streamlit Secrets."
        )


    (
        business_agent,
        technical_agent,
        product_agent,
        qa_agent
    ) = create_agents(llm)


    # ========================================================
    # TASK 1 — BUSINESS ANALYSIS
    # ========================================================

    business_task = Task(

        description=f"""
        Analyze the following software project requirement.

        PROJECT NAME:
        {project_name}

        BUSINESS DOMAIN:
        {business_domain}

        ORIGINAL CLIENT REQUIREMENT:
        {requirement}

        Produce a professional business analysis containing:

        1. Business objective
        2. Problem statement
        3. Stakeholders
        4. Target users
        5. System actors
        6. Main workflows
        7. Functional requirements
        8. Business rules
        9. Assumptions
        10. Ambiguous requirements
        11. Missing information

        Important:
        Do not invent unsupported requirements.
        Clearly identify assumptions separately.
        """,

        expected_output="""
        A structured business requirements analysis containing
        objectives, stakeholders, actors, workflows, functional
        requirements, business rules, assumptions, ambiguities
        and missing information.
        """,

        agent=business_agent
    )


    # ========================================================
    # TASK 2 — TECHNICAL ANALYSIS
    # ========================================================

    technical_task = Task(

        description=f"""
        Produce a professional technical requirements analysis.

        PROJECT:
        {project_name}

        DOMAIN:
        {business_domain}

        ORIGINAL REQUIREMENT:
        {requirement}

        Use the Business Analyst's output as context.

        Analyze:

        1. Technical requirements
        2. Non-functional requirements
        3. Authentication requirements
        4. Authorization requirements
        5. Data requirements
        6. API requirements
        7. Integration requirements
        8. Security requirements
        9. Performance requirements
        10. Scalability requirements
        11. Dependencies
        12. Technical risks

        Keep recommendations directly connected to the
        original requirement.
        """,

        expected_output="""
        A detailed technical requirements specification
        derived from the business analysis.
        """,

        agent=technical_agent,

        context=[business_task]
    )


    # ========================================================
    # TASK 3 — PRODUCT ANALYSIS
    # ========================================================

    product_task = Task(

        description=f"""
        Create a complete product specification.

        PROJECT:
        {project_name}

        ORIGINAL REQUIREMENT:
        {requirement}

        Use the Business Analyst and Technical Analyst
        outputs as context.

        Produce:

        1. User personas
        2. User journeys
        3. User stories
        4. Acceptance criteria
        5. Main workflows
        6. Edge cases
        7. Usability requirements
        8. MVP features
        9. Future enhancement ideas

        User story format:

        As a [user],
        I want [feature],
        so that [benefit].

        Acceptance criteria must be specific,
        measurable and testable.
        """,

        expected_output="""
        A structured product specification containing
        personas, user journeys, user stories, acceptance
        criteria, workflows, edge cases, MVP scope and
        future enhancements.
        """,

        agent=product_agent,

        context=[
            business_task,
            technical_task
        ]
    )


    # ========================================================
    # TASK 4 — QA REVIEW
    # ========================================================

    qa_task = Task(

        description=f"""
        Perform a final requirements quality review.

        PROJECT:
        {project_name}

        ORIGINAL REQUIREMENT:
        {requirement}

        Review the complete outputs from all previous agents.

        Identify:

        1. Missing requirements
        2. Ambiguous requirements
        3. Contradictions
        4. Weak acceptance criteria
        5. Untestable requirements
        6. Security concerns
        7. Edge cases
        8. Implementation risks
        9. Required clarifications
        10. QA test scenarios

        Then produce:

        FINAL REQUIREMENTS READINESS

        Choose exactly one:

        READY FOR DEVELOPMENT

        OR

        NEEDS CLARIFICATION

        Explain the decision using evidence from the
        requirement analysis.
        """,

        expected_output="""
        A comprehensive QA requirements review containing
        missing requirements, ambiguities, contradictions,
        testability issues, risks, clarifications, test
        scenarios and a final development-readiness assessment.
        """,

        agent=qa_agent,

        context=[
            business_task,
            technical_task,
            product_task
        ]
    )


    # ========================================================
    # CREW
    # ========================================================

    crew = Crew(

        agents=[
            business_agent,
            technical_agent,
            product_agent,
            qa_agent
        ],

        tasks=[
            business_task,
            technical_task,
            product_task,
            qa_task
        ],

        process=Process.sequential,

        verbose=False
    )


    # REAL CREWAI EXECUTION
    result = crew.kickoff()

    return result


# ============================================================
# INPUT SECTION
# ============================================================

st.markdown(
    '<div class="section-title">Project Analysis</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="section-subtitle">
        Enter a real software requirement. The AI crew will
        analyze it sequentially and generate a development-ready
        requirements specification.
    </div>
    """,
    unsafe_allow_html=True
)


left, right = st.columns([1.3, 1])


with left:

    project_name = st.text_input(
        "Project Name",
        placeholder="Example: Employee Leave Management System"
    )


with right:

    business_domain = st.selectbox(
        "Business Domain",
        [
            "Software / SaaS",
            "E-Commerce",
            "FinTech",
            "Healthcare",
            "Education",
            "Human Resources",
            "Logistics",
            "Retail",
            "Other"
        ]
    )


requirement = st.text_area(
    "Client / Business Requirement",
    height=250,
    placeholder=(
        "Describe the client's requirement in natural language.\n\n"
        "Example:\n"
        "We need an employee leave management platform where "
        "employees can submit leave requests, managers can "
        "approve or reject requests, and HR administrators "
        "can view leave history and generate reports."
    )
)


# ============================================================
# RUN BUTTON
# ============================================================

st.markdown("")

run_analysis = st.button(
    "◆  Run AI Requirements Analysis",
    type="primary",
    use_container_width=True
)


# ============================================================
# EXECUTION
# ============================================================

if run_analysis:

    if not project_name.strip():

        st.warning("Please enter a project name.")
        st.stop()


    if not requirement.strip():

        st.warning("Please enter a client requirement.")
        st.stop()


    if len(requirement.strip()) < 50:

        st.warning(
            "Please provide a more detailed requirement "
            "for meaningful analysis."
        )
        st.stop()


    if not get_groq_api_key():

        st.error(
            "GROQ_API_KEY is missing. Please configure it "
            "in Streamlit Secrets."
        )
        st.stop()


    st.markdown("---")

    st.markdown(
        '<div class="section-title">AI Crew Execution</div>',
        unsafe_allow_html=True
    )


    # This is a real execution status, not fake agent output.
    status = st.status(
        "Running the CrewAI requirements workflow...",
        expanded=True
    )


    try:

        status.write(
            "CrewAI is executing the Business Analyst, "
            "Technical Analyst, Product Analyst and QA Reviewer."
        )


        result = run_requirements_crew(
            project_name=project_name,
            business_domain=business_domain,
            requirement=requirement
        )


        status.update(
            label="Requirements analysis completed",
            state="complete",
            expanded=False
        )


        # ====================================================
        # METRICS
        # ====================================================

        st.markdown("")

        m1, m2, m3, m4 = st.columns(4)


        with m1:

            st.markdown(
                """
                <div class="metric-card">
                    <div class="metric-number">4</div>
                    <div class="metric-label">
                        AI Agents
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


        with m2:

            st.markdown(
                """
                <div class="metric-card">
                    <div class="metric-number">4</div>
                    <div class="metric-label">
                        Analysis Tasks
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


        with m3:

            st.markdown(
                """
                <div class="metric-card">
                    <div class="metric-number">1</div>
                    <div class="metric-label">
                        Final Specification
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


        with m4:

            current_time = datetime.now().strftime("%H:%M")

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-number">
                        {current_time}
                    </div>
                    <div class="metric-label">
                        Completed
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


        # ====================================================
        # FINAL REPORT
        # ====================================================

        st.markdown("")

        st.markdown(
            """
            <div class="report-top">
                <div class="report-title">
                    Final Requirements Specification
                </div>

                <div class="report-meta">
                    Generated by the ReqPilot AI Crew
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


        final_report = str(result)


        st.markdown(
            '<div class="report-container">',
            unsafe_allow_html=True
        )

        st.markdown(final_report)

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


        # ====================================================
        # DOWNLOAD
        # ====================================================

        st.markdown("")

        safe_project_name = re.sub(
            r"[^a-zA-Z0-9_-]",
            "_",
            project_name.strip()
        )


        st.download_button(
            label="↓  Download Requirements Report",
            data=final_report,
            file_name=(
                f"{safe_project_name}"
                "_requirements_report.txt"
            ),
            mime="text/plain",
            use_container_width=True
        )


    except Exception as error:

        status.update(
            label="CrewAI execution failed",
            state="error",
            expanded=True
        )

        st.error(
            "The AI workflow could not be completed."
        )

        st.markdown("### Error Details")

        st.code(str(error))


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        ReqPilot AI · Autonomous Software Requirements Engineering
        <br>
        CrewAI · Groq · Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
```
