"""
Streamlit Web Dashboard for the Job Hunting Agent.

Provides an interactive web interface featuring candidate resume upload,
search filter controls, job match cards, cover letter customizer, and application tracking.

Usage:
    streamlit run app.py
"""

import os
import tempfile
import pandas as pd
import streamlit as st

from core import app as workflow_app
from core.resume_parser import parse_resume_from_bytes
from core.state import ApplicationMaterials, JobMatchAnalysis, UserPreferences
from database.manager import db

# ── Page Configuration ──
st.set_page_config(
    page_title="Job Hunting Agent — AI Career Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS Styling ──
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4F46E5, #06B6D4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #9CA3AF;
        margin-bottom: 1.5rem;
    }
    .match-badge {
        display: inline-block;
        padding: 0.35rem 0.8rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.95rem;
    }
    .high-match { background-color: #065F46; color: #34D399; }
    .mid-match { background-color: #78350F; color: #FBBF24; }
    .low-match { background-color: #7F1D1D; color: #FCA5A5; }
    
    .skill-pill-green {
        display: inline-block;
        background-color: #1E293B;
        color: #10B981;
        border: 1px solid #059669;
        padding: 0.2rem 0.6rem;
        border-radius: 0.375rem;
        font-size: 0.82rem;
        margin-right: 0.3rem;
        margin-bottom: 0.3rem;
    }
    .skill-pill-red {
        display: inline-block;
        background-color: #1E293B;
        color: #EF4444;
        border: 1px solid #DC2626;
        padding: 0.2rem 0.6rem;
        border-radius: 0.375rem;
        font-size: 0.82rem;
        margin-right: 0.3rem;
        margin-bottom: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)


def main() -> None:
    """Main Streamlit web application."""
    st.markdown('<div class="main-header">🤖 Job Hunting Agent</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Autonomous AI Executive Career & Job Application Assistant</div>', unsafe_allow_html=True)

    # ── Sidebar Controls ──
    with st.sidebar:
        st.header("⚙️ Candidate Preferences")

        # 1. Resume File Upload
        uploaded_file = st.file_uploader(
            "Upload Candidate Resume (PDF, DOCX, TXT)",
            type=["pdf", "docx", "txt"],
            help="Upload your resume to extract candidate background and skills."
        )

        # Save uploaded file to temporary path if provided
        resume_path = "sample_resume.txt"
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                resume_path = tmp_file.name
            st.success(f"Uploaded: `{uploaded_file.name}`")

        # 2. Search Preferences
        target_roles_input = st.text_input(
            "Target Job Roles (comma-separated)",
            value="Machine Learning Engineer, AI Developer",
            help="Enter job titles you are actively targetting."
        )
        target_roles = [r.strip() for r in target_roles_input.split(",") if r.strip()]

        preferred_locs_input = st.text_input(
            "Preferred Locations (comma-separated)",
            value="Remote",
            help="Target cities, states, or Remote."
        )
        preferred_locations = [loc.strip() for loc in preferred_locs_input.split(",") if loc.strip()]

        remote_pref = st.selectbox(
            "Remote Preference",
            options=["Remote", "Hybrid", "Onsite", "Any"],
            index=0
        )

        min_score_filter = st.slider(
            "Minimum Match Score Filter (%)",
            min_value=0,
            max_value=100,
            value=50,
            step=5
        )

        provider_choice = st.selectbox(
            "Search Provider",
            options=["duckduckgo", "tavily"],
            index=0,
            help="DuckDuckGo is 100% free with no API key required."
        )

        search_button = st.button("🚀 Launch Job Search", use_container_width=True, type="primary")

    # Store search results in Session State
    if "final_state" not in st.session_state:
        st.session_state.final_state = None

    # Execute workflow when search button is clicked
    if search_button:
        os.environ["SEARCH_PROVIDER"] = provider_choice
        user_prefs = UserPreferences(
            target_roles=target_roles,
            preferred_locations=preferred_locations,
            remote_preference=remote_pref
        )

        inputs = {
            "resume_path": resume_path,
            "preferences": user_prefs
        }
        config = {"configurable": {"thread_id": "streamlit_session"}}

        with st.spinner("Analyzing resume, searching web, ranking jobs & generating tailored materials..."):
            try:
                st.session_state.final_state = workflow_app.invoke(inputs, config=config)
                st.toast("Job Search Complete! ✅", icon="🎉")
            except Exception as e:
                st.error(f"Execution Error: {e}")

    # ── Main Dashboard Tabs ──
    tab1, tab2, tab3 = st.tabs([
        "🎯 Top Job Recommendations",
        "✉️ Cover Letter & Tailored Bullets",
        "📊 Application History & Analytics"
    ])

    final_state = st.session_state.final_state

    # ── TAB 1: Top Job Recommendations ──
    with tab1:
        if final_state and final_state.get("ranked_matches"):
            ranked_matches: list[JobMatchAnalysis] = [
                m for m in final_state["ranked_matches"]
                if isinstance(m, JobMatchAnalysis) and m.overall_score >= min_score_filter
            ]

            # Metric Cards
            col1, col2, col3 = st.columns(3)
            col1.metric("Evaluated Matches", len(final_state["ranked_matches"]))
            col2.metric("Filtered Matches (>= " + str(min_score_filter) + "%)", len(ranked_matches))
            avg_score = (sum(m.overall_score for m in ranked_matches) / len(ranked_matches)) if ranked_matches else 0.0
            col3.metric("Average Match Score", f"{avg_score:.1f}%")

            st.divider()

            if not ranked_matches:
                st.warning(f"No job matches meet the minimum score threshold of {min_score_filter}%. Try lowering the slider in the sidebar.")
            else:
                for idx, match in enumerate(ranked_matches, 1):
                    score = match.overall_score
                    badge_class = "high-match" if score >= 80 else ("mid-match" if score >= 60 else "low-match")

                    with st.container(border=True):
                        title_col, badge_col = st.columns([4, 1])
                        with title_col:
                            st.subheader(f"#{idx}. {match.job_title}")
                            st.caption(f"🏢 **{match.company}** | 🔗 [Direct Job Link]({match.url})")
                        with badge_col:
                            st.markdown(f'<div style="text-align: right;"><span class="match-badge {badge_class}">{score:.0f}% Match</span></div>', unsafe_allow_html=True)

                        st.write(f"**Match Rationale:** {match.match_rationale}")

                        # Score distribution bar
                        st.progress(int(score) / 100, text=f"Overall Compatibility: {score:.0f}% (Skills: {match.skills_match_score:.0f}% | Experience: {match.experience_match_score:.0f}%)")

                        # Skills Pills
                        m_col, gap_col = st.columns(2)
                        with m_col:
                            if match.matching_skills:
                                st.markdown("**Matching Candidate Skills:**")
                                pills_html = "".join([f'<span class="skill-pill-green">✓ {s}</span>' for s in match.matching_skills])
                                st.markdown(pills_html, unsafe_allow_html=True)
                        with gap_col:
                            if match.missing_skills:
                                st.markdown("**Skill Gaps / Missing:**")
                                gaps_html = "".join([f'<span class="skill-pill-red">✗ {s}</span>' for s in match.missing_skills])
                                st.markdown(gaps_html, unsafe_allow_html=True)

                        if match.pros:
                            st.write(f"🟢 **Pros:** {', '.join(match.pros)}")
                        if match.cons:
                            st.write(f"🔴 **Cons:** {', '.join(match.cons)}")
        else:
            st.info("👈 Upload your resume and click **'Launch Job Search'** in the sidebar to start finding tailored opportunities!")

    # ── TAB 2: Cover Letter & Tailored Bullets ──
    with tab2:
        if final_state and final_state.get("application_materials"):
            materials: ApplicationMaterials = final_state["application_materials"]
            st.success(f"Tailored Application Materials generated for: **{materials.job_title}** at **{materials.company}**")

            col_a, col_b = st.columns([1, 1])
            with col_a:
                st.subheader("📄 Tailored Resume Bullets (Google XYZ Style)")
                for bullet in materials.tailored_bullets:
                    st.markdown(f"• {bullet}")

            with col_b:
                st.subheader("✉️ Customized Cover Letter")
                st.text_area("Cover Letter Text", value=materials.cover_letter, height=320)

                # Download Button
                export_markdown = f"# Application Materials — {materials.job_title} at {materials.company}\n\n"
                export_markdown += "## Tailored Resume Bullets\n\n"
                for b in materials.tailored_bullets:
                    export_markdown += f"* {b}\n"
                export_markdown += f"\n## Customized Cover Letter\n\n{materials.cover_letter}"

                st.download_button(
                    label="💾 Download Application Materials (.md)",
                    data=export_markdown,
                    file_name=f"Application_{materials.company}.md",
                    mime="text/markdown"
                )
        else:
            st.info("Run a job search to generate tailored resume bullet points and customized cover letters for top opportunities.")

    # ── TAB 3: Application History & Analytics ──
    with tab3:
        st.subheader("📊 Tracked Job Applications & Historical Matches")

        try:
            with db._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT job_id, job_title, company, url, overall_score, evaluated_at FROM job_match_history ORDER BY evaluated_at DESC")
                rows = cursor.fetchall()
                if rows:
                    df = pd.DataFrame([dict(r) for r in rows])
                    st.dataframe(
                        df,
                        column_config={
                            "url": st.column_config.LinkColumn("Direct Link"),
                            "overall_score": st.column_config.ProgressColumn("Match Score %", min_value=0, max_value=100, format="%.0f%%")
                        },
                        use_container_width=True
                    )

                    # Score distribution chart
                    st.bar_chart(df.set_index("job_title")["overall_score"])
                else:
                    st.info("No historical job match evaluations logged in SQLite database yet.")
        except Exception as e:
            st.warning(f"Database query error: {e}")


if __name__ == "__main__":
    main()
