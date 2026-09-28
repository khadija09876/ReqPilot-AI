```python
import os
import re
import streamlit as st

from crewai import Agent, Task, Crew, Process, LLM


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ReqPilot AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown(
    """
    <style>
        .main {
            background-color: #f6f7fb;
        }

        .block-container {
            max-width: 1250px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .hero {
            background: linear-gradient(
                135deg,
                #111827,
                #1f2937
            );
            padding: 2.5rem;
            border-radius: 20px;
            margin-bottom: 2rem;
        }

        .hero h1 {
            color: white;
            font-size: 44px;
            margin-bottom: 8px;
        }

        .hero h3 {
            color: #d1d5db;
            font-weight: 400;
            margin-top: 0;
        }

        .hero p {
            color: #9ca3af;
            font-size: 16px;
        }

        .agent-card {
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 14px;
            padding: 18px;
            margin-bottom: 12px;
        }

        .agent-card h4 {
            margin-bottom: 5px;
        }

        .agent-card p {
            color: #6b7280;
            margin-bottom: 0;
        }

        .metric-card {
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 14px;
            padding: 20px;
            text-align: center;
        }

        .metric-value {
            font-size: 28px;
            font-weight: 700;
        }

        .metric-label {
            color: #6b7280;
            font-size: 14px;
        }

        .footer {
            text-align: center;
            color: #6b7280;
            padding: 20px;
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
        <h1>🤖 ReqPilot AI</h1>
        <h3>Autonomous Software Requirements Engineering System</h3>
        <p>
            Transform raw business requirements into structured,
            development-ready software specifications using a
            CrewAI-powered multi-agent workflow.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## ⚙️ System Architecture")

    st.success("CrewAI Multi-Agent System")

    st.markdown("---")

    st.markdown("### AI Agents")

    st.markdown(
        """
        **01 — Business Analyst**

        Understands the business problem and identifies
        functional requirements.

        **02 — Technical Analyst**

        Converts business requirements into technical and
        non-functional requirements.

        **03 — Product Analyst**

        Creates user stories, acceptance criteria,
        workflows, and edge cases.

        **04 — QA Reviewer**

        Reviews the complete specification for ambiguity,
        missing requirements, risks, and testability.
        """
    )

    st.markdown("---")

    st.markdown("### Technology")

    st.markdown(
        """
        - Python
        - CrewAI
        - Groq API
        - Streamlit
        - GitHub
        - Streamlit Cloud
        """
    )

    st.markdown("---")

    st.caption("ReqPilot AI")


# ============================================================
# GROQ CONFIGURATION
# ============================================================

def get_groq_api_key():
    """Retrieve the Groq API key from Streamlit Secrets or environment variables."""

    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass

    return os.getenv("GROQ_API_KEY")


def create_llm():
    """Create the CrewAI LLM using Groq."""

    api_key = get_groq_api_key()

    if not api_key:
        return None

    return LLM(
        model="groq/openai/gpt-oss-120b",
        api_key=api_key,
        temperature=0.2
    )


# ============================================================
# AGENT CREATION
# ============================================================

def create_agents(llm):
    """Create the four specialized CrewAI agents."""

    business_analyst = Agent(
        role="Senior Business Analyst",
        goal=(
            "Understand the client's business requirement and "
            "convert it into clear, structured business requirements."
        ),
        backstory=(
            "You are a senior business analyst working in a "
            "professional software development company. You specialize "
            "in understanding client needs, identifying actors, "
            "business rules, workflows, and functional requirements."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False
    )

    technical_analyst = Agent(
        role="Senior Technical Requirements Analyst",
        goal=(
            "Transform business requirements into implementable "
            "technical and non-functional requirements."
        ),
        backstory=(
            "You are an experienced technical requirements analyst "
            "with strong knowledge of software architecture, APIs, "
            "security, databases, integrations, scalability, and "
            "performance requirements."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False
    )

    product_analyst = Agent(
        role="Senior Product Analyst",
        goal=(
            "Translate requirements into user-centered product "
            "specifications, user stories, acceptance criteria, "
            "workflows, and edge cases."
        ),
        backstory=(
            "You are a senior product analyst who focuses on user "
            "needs, product behavior, usability, workflows, and "
            "measurable acceptance criteria."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False
    )

    qa_reviewer = Agent(
        role="Senior QA Requirements Reviewer",
        goal=(
            "Review the complete requirements specification and "
            "identify missing, ambiguous, contradictory, or "
            "untestable requirements."
        ),
        backstory=(
            "You are a senior QA engineer specializing in requirements "
            "validation. Your responsibility is to detect requirement "
            "gaps and quality risks before software development begins."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False
    )

    return (
        business_analyst,
        technical_analyst,
        product_analyst,
        qa_reviewer
    )


# ============================================================
# CREW EXECUTION
# ============================================================

def run_reqpilot(project_name, domain, requirement):

    llm = create_llm()

    if llm is None:
        raise ValueError(
            "GROQ_API_KEY is not configured. "
            "Please add it to Streamlit Secrets."
        )

    (
        business_analyst,
        technical_analyst,
        product_analyst,
        qa_reviewer
    ) = create_agents(llm)

    # --------------------------------------------------------
    # TASK 1 — BUSINESS ANALYSIS
    # --------------------------------------------------------

    business_task = Task(
        description=f"""
        Analyze the following software project requirement.

        PROJECT:
        {project_name}

        DOMAIN:
        {domain}

        CLIENT REQUIREMENT:
        {requirement}

        Produce a professional business analysis containing:

        1. Business objective
        2. Problem statement
        3. Target users
        4. System actors
        5. Main business workflows
        6. Functional requirements
        7. Business rules
        8. Assumptions
        9. Ambiguous areas
        10. Missing information

        Do not invent unnecessary features.
        Clearly distinguish stated requirements from assumptions.
        """,

        expected_output="""
        A structured business requirements analysis that can be
        consumed by the technical and product analysis agents.
        """,

        agent=business_analyst
    )

    # --------------------------------------------------------
    # TASK 2 — TECHNICAL ANALYSIS
    # --------------------------------------------------------

    technical_task = Task(
        description=f"""
        Analyze the technical implications of the business
        requirements for:

        PROJECT:
        {project_name}

        DOMAIN:
        {domain}

        REQUIREMENT:
        {requirement}

        Use the Business Analyst's findings as context.

        Produce:

        1. Functional technical requirements
        2. Non-functional requirements
        3. Authentication considerations
        4. Authorization considerations
        5. Data requirements
        6. API and integration considerations
        7. Security considerations
        8. Performance considerations
        9. Scalability considerations
        10. Dependencies
        11. Technical risks

        Keep all recommendations relevant to the actual requirement.
        """,

        expected_output="""
        A professional technical requirements specification
        derived from the business analysis.
        """,

        agent=technical_analyst,
        context=[business_task]
    )

    # --------------------------------------------------------
    # TASK 3 — PRODUCT ANALYSIS
    # --------------------------------------------------------

    product_task = Task(
        description=f"""
        Create a product-oriented specification for:

        PROJECT:
        {project_name}

        REQUIREMENT:
        {requirement}

        Use the business and technical analysis as context.

        Include:

        1. User personas
        2. User stories
        3. Acceptance criteria
        4. User workflows
        5. Edge cases
        6. Usability considerations
        7. MVP features
        8. Future enhancement suggestions

        User stories must use:

        As a [user],
        I want [function],
        so that [benefit].

        Acceptance criteria must be specific and testable.
        """,

        expected_output="""
        A structured product specification containing user personas,
        user stories, acceptance criteria, workflows, edge cases,
        MVP scope, and future enhancements.
        """,

        agent=product_analyst,
        context=[business_task, technical_task]
    )

    # --------------------------------------------------------
    # TASK 4 — QA REVIEW
    # --------------------------------------------------------

    qa_task = Task(
        description=f"""
        Perform a final requirements quality review for:

        PROJECT:
        {project_name}

        REQUIREMENT:
        {requirement}

        Review all previous analyses.

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

        Finally provide:

        FINAL REQUIREMENTS READINESS

        State either:

        READY FOR DEVELOPMENT

        OR

        NEEDS CLARIFICATION

        Explain the reasons using evidence from the provided
        requirement and previous analysis.
        """,

        expected_output="""
        A comprehensive QA requirements review with missing
        requirements, ambiguities, risks, clarifications,
        test scenarios, and a final development-readiness assessment.
        """,

        agent=qa_reviewer,
        context=[
            business_task,
            technical_task,
            product_task
        ]
    )

    # --------------------------------------------------------
    # CREW
    # --------------------------------------------------------

    crew = Crew(
        agents=[
            business_analyst,
            technical_analyst,
            product_analyst,
            qa_reviewer
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

    return crew.kickoff()


# ============================================================
# INPUT SECTION
# ============================================================

st.markdown("## 📝 Software Requirement")

col1, col2 = st.columns(2)

with col1:

    project_name = st.text_input(
        "Project Name",
        placeholder="e.g. Employee Leave Management System"
    )

with col2:

    domain = st.selectbox(
        "Business Domain",
        [
            "Software / SaaS",
            "E-Commerce",
            "FinTech",
            "Healthcare",
            "Education",
            "Logistics",
            "Human Resources",
            "Retail",
            "Other"
        ]
    )

requirement = st.text_area(
    "Client / Business Requirement",
    height=230,
    placeholder=(
        "Describe the client's software requirement here..."
    )
)


# ============================================================
# RUN BUTTON
# ============================================================

st.markdown("")

run_button = st.button(
    "🚀 Analyze Requirement",
    type="primary",
    use_container_width=True
)


# ============================================================
# APPLICATION LOGIC
# ============================================================

if run_button:

    if not project_name.strip():

        st.warning("Please enter a project name.")

        st.stop()

    if not requirement.strip():

        st.warning("Please enter a software requirement.")

        st.stop()

    if len(requirement.strip()) < 40:

        st.warning(
            "Please provide a more detailed requirement "
            "for meaningful analysis."
        )

        st.stop()

    if not get_groq_api_key():

        st.error(
            "GROQ_API_KEY is missing. "
            "Please configure it in Streamlit Secrets."
        )

        st.stop()

    st.markdown("---")

    st.markdown("## 🤖 AI Crew Execution")

    progress = st.progress(0)

    status = st.empty()

    try:

        status.info(
            "Business Analyst is analyzing the requirement..."
        )

        progress.progress(20)

        result = run_reqpilot(
            project_name=project_name,
            domain=domain,
            requirement=requirement
        )

        progress.progress(100)

        status.success(
            "All four AI agents completed successfully."
        )

        st.markdown("---")

        st.markdown("## 📋 Final Requirements Report")

        final_report = str(result)

        st.markdown(final_report)

        st.markdown("---")

        st.download_button(
            label="⬇️ Download Requirements Report",
            data=final_report,
            file_name=(
                f"{re.sub(r'[^a-zA-Z0-9_-]', '_', project_name)}"
                "_requirements_report.txt"
            ),
            mime="text/plain",
            use_container_width=True
        )

    except Exception as error:

        progress.progress(0)

        status.empty()

        st.error(
            "The AI crew could not complete the analysis."
        )

        with st.expander("Technical Error Details"):

            st.code(str(error))


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div class="footer">
        ReqPilot AI · Autonomous Software Requirements Engineering
        <br>
        Powered by CrewAI, Groq and Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
```
