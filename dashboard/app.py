"""
Masar — EduMatch Tunisia (UI Only)
A complete education-to-career matching platform for Tunisian students and job seekers.

Requirements:
    pip install streamlit plotly

Usage:
    streamlit run app.py

Note: This is a UI-only version without backend functionality.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================

# Translations for multilingual support
TRANSLATIONS = {
    "en": {
        "title": "Masar — EduMatch Tunisia",
        "subtitle": "Connecting Education to Career Opportunities",
        "education_level": "Education Level",
        "current_major": "Current Major/Field",
        "experience": "Years of Experience",
        "skills": "Your Skills (one per line)",
        "strengths": "Your Strengths (tags)",
        "weaknesses": "Areas to Improve",
        "locations": "Preferred Work Locations",
        "find_match": "Find My Perfect Match",
        "loading": "Analyzing your profile...",
        "top_jobs": "🎯 Top Job Matches for You",
        "match_score": "Match Score",
        "skill_gaps": "Skill Gaps",
        "recommended_programs": "🎓 Recommended Programs",
        "action_plan": "📋 Your Action Plan",
        "download": "Download My Plan",
        "why": "Why this suggestion?",
        "feedback": "Was this helpful?",
    },
    "fr": {
        "title": "Masar — Orientation Tunisie",
        "subtitle": "Connecter l'Éducation aux Opportunités de Carrière",
        "education_level": "Niveau d'Études",
        "current_major": "Spécialité Actuelle",
        "experience": "Années d'Expérience",
        "skills": "Vos Compétences (une par ligne)",
        "strengths": "Vos Points Forts",
        "weaknesses": "Points à Améliorer",
        "locations": "Lieux de Travail Préférés",
        "find_match": "Trouver Mon Match",
        "loading": "Analyse de votre profil...",
        "top_jobs": "🎯 Meilleurs Emplois pour Vous",
        "match_score": "Score de Correspondance",
        "skill_gaps": "Compétences Manquantes",
        "recommended_programs": "🎓 Programmes Recommandés",
        "action_plan": "📋 Votre Plan d'Action",
        "download": "Télécharger Mon Plan",
        "why": "Pourquoi cette suggestion?",
        "feedback": "Était-ce utile?",
    },
    "ar": {
        "title": "مسار — التوجيه التونسي",
        "subtitle": "ربط التعليم بفرص العمل",
        "education_level": "المستوى التعليمي",
        "current_major": "التخصص الحالي",
        "experience": "سنوات الخبرة",
        "skills": "مهاراتك (واحدة في كل سطر)",
        "strengths": "نقاط قوتك",
        "weaknesses": "مجالات للتحسين",
        "locations": "أماكن العمل المفضلة",
        "find_match": "ابحث عن تطابقي",
        "loading": "جاري تحليل ملفك الشخصي...",
        "top_jobs": "🎯 أفضل الوظائف المناسبة لك",
        "match_score": "نتيجة التطابق",
        "skill_gaps": "الفجوات في المهارات",
        "recommended_programs": "🎓 البرامج الموصى بها",
        "action_plan": "📋 خطة العمل الخاصة بك",
        "download": "تحميل خطتي",
        "why": "لماذا هذا الاقتراح؟",
        "feedback": "هل كان هذا مفيداً؟",
    }
}

# Sample data for UI demo
SAMPLE_JOBS = [
    {"title": "Software Developer", "company": "TechCorp Tunisia", "location": "Tunis", "match": 95},
    {"title": "Data Analyst", "company": "DataFlow", "location": "Sfax", "match": 88},
    {"title": "Web Developer", "company": "WebSolutions", "location": "Sousse", "match": 82},
    {"title": "Business Analyst", "company": "ConsultPro", "location": "Tunis", "match": 75},
    {"title": "Product Manager", "company": "InnovateTN", "location": "Ariana", "match": 70}
]

SAMPLE_PROGRAMS = [
    {"name": "Master in Computer Science", "university": "ENSI", "match": 92},
    {"name": "Data Science Bootcamp", "university": "GoMyCode", "match": 85},
    {"name": "Digital Marketing", "university": "ISIMS", "match": 78}
]

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Masar — EduMatch Tunisia",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #FF6B35 0%, #F7931E 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
        color: white;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #FF6B35;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .job-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .match-score {
        font-size: 1.2rem;
        font-weight: bold;
        color: #FF6B35;
    }
    .stButton > button {
        background: linear-gradient(90deg, #FF6B35 0%, #F7931E 100%);
        border: none;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    # Language selection in sidebar
    st.sidebar.title("🌍 Language / اللغة / Langue")
    language = st.sidebar.selectbox(
        "Choose language:",
        options=["en", "fr", "ar"],
        format_func=lambda x: {"en": "English", "fr": "Français", "ar": "العربية"}[x],
        index=0
    )
    
    # Get translations for selected language
    t = TRANSLATIONS[language]
    
    # Set text direction based on language
    if language == "ar":
        st.markdown('<div dir="rtl">', unsafe_allow_html=True)
    
    # Main header
    st.markdown(f"""
    <div class="main-header">
        <h1>{t['title']}</h1>
        <p style="font-size: 1.2rem; margin-bottom: 0;">{t['subtitle']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Two-column layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("📝 Your Profile")
        
        # Education level
        education_level = st.selectbox(
            t["education_level"],
            ["High School", "Bachelor's", "Master's", "PhD", "Vocational Training"]
        )
        
        # Current major
        current_major = st.text_input(t["current_major"], placeholder="e.g., Computer Science")
        
        # Experience
        experience = st.slider(t["experience"], 0, 20, 0)
        
        # Skills
        skills_text = st.text_area(
            t["skills"],
            placeholder="Python\nJavaScript\nData Analysis\nProject Management",
            height=120
        )
        
        # Strengths
        strengths = st.multiselect(
            t["strengths"],
            ["Leadership", "Problem Solving", "Communication", "Teamwork", 
             "Creativity", "Analytical Thinking", "Time Management"]
        )
        
        # Weaknesses/Areas to improve
        weaknesses = st.multiselect(
            t["weaknesses"],
            ["Public Speaking", "Technical Writing", "Networking", 
             "Foreign Languages", "Advanced Math", "Presentation Skills"]
        )
        
        # Preferred locations
        locations = st.multiselect(
            t["locations"],
            ["Tunis", "Sfax", "Sousse", "Monastir", "Bizerte", "Gabes", 
             "Kairouan", "Ariana", "Remote Work"]
        )
        
        # Find match button
        if st.button(t["find_match"], type="primary"):
            st.session_state.show_results = True
    
    with col2:
        if hasattr(st.session_state, 'show_results') and st.session_state.show_results:
            # Show loading animation
            with st.spinner(t["loading"]):
                import time
                time.sleep(2)  # Simulate processing
            
            # Results section
            st.header(t["top_jobs"])
            
            # Display job matches
            for i, job in enumerate(SAMPLE_JOBS):
                with st.container():
                    job_col1, job_col2, job_col3 = st.columns([3, 2, 1])
                    
                    with job_col1:
                        st.subheader(job["title"])
                        st.write(f"🏢 {job['company']}")
                        st.write(f"📍 {job['location']}")
                    
                    with job_col2:
                        # Match score
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = job["match"],
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': t["match_score"]},
                            gauge = {
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "#FF6B35"},
                                'steps': [
                                    {'range': [0, 50], 'color': "lightgray"},
                                    {'range': [50, 80], 'color': "yellow"},
                                    {'range': [80, 100], 'color': "lightgreen"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 90
                                }
                            }
                        ))
                        fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with job_col3:
                        st.button("View Details", key=f"job_{i}")
                        st.button("Apply", key=f"apply_{i}")
                
                st.divider()
            
            # Recommended programs section
            st.header(t["recommended_programs"])
            
            prog_cols = st.columns(3)
            for i, program in enumerate(SAMPLE_PROGRAMS):
                with prog_cols[i % 3]:
                    st.metric(
                        program["name"],
                        f"{program['match']}% match",
                        program["university"]
                    )
            
            # Action plan section
            st.header(t["action_plan"])
            
            action_tabs = st.tabs(["📚 Short Term", "🎯 Medium Term", "🚀 Long Term"])
            
            with action_tabs[0]:
                st.markdown("""
                **Next 3 months:**
                - Complete Python programming course
                - Build 2-3 portfolio projects  
                - Practice coding interview questions
                - Improve English communication skills
                """)
            
            with action_tabs[1]:
                st.markdown("""
                **6-12 months:**
                - Apply for internships in target companies
                - Attend tech meetups and networking events
                - Contribute to open source projects
                - Consider specialized certification
                """)
            
            with action_tabs[2]:
                st.markdown("""
                **1-2 years:**
                - Pursue Master's degree if needed
                - Apply for full-time positions
                - Build professional network
                - Consider entrepreneurship opportunities
                """)
            
            # Download plan button
            if st.button(t["download"]):
                st.success("📄 Plan downloaded successfully!")
            
            # Feedback
            st.subheader(t["feedback"])
            feedback_col1, feedback_col2 = st.columns(2)
            with feedback_col1:
                st.button("👍 Yes, helpful!")
            with feedback_col2:
                st.button("👎 Needs improvement")
        
        else:
            # Welcome message
            st.info("👋 Fill out your profile on the left to get personalized career recommendations!")
            
            # Sample statistics
            st.subheader("📊 Platform Statistics")
            
            stat_cols = st.columns(4)
            with stat_cols[0]:
                st.metric("Active Jobs", "1,247", "+123")
            with stat_cols[1]:
                st.metric("Universities", "45", "+3")
            with stat_cols[2]:
                st.metric("Success Stories", "892", "+67")
            with stat_cols[3]:
                st.metric("Skills Mapped", "2,156", "+89")
            
            # Sample chart
            st.subheader("🎯 Popular Career Paths")
            sectors = ["Technology", "Healthcare", "Finance", "Education", "Tourism"]
            values = [35, 20, 15, 18, 12]
            
            fig = px.pie(
                values=values, 
                names=sectors, 
                title="Job Distribution by Sector",
                color_discrete_sequence=px.colors.sequential.Oranges_r
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Close RTL div for Arabic
    if language == "ar":
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "🎯 **Masar** — Empowering Tunisian talent through smart career matching | "
        "Built with ❤️ for Tunisia's future"
    )

if __name__ == "__main__":
    main()