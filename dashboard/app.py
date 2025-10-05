"""
Masar — EduMatch Tunisia
A complete education-to-career matching platform for Tunisian students and job seekers.

Requirements:
    pip install streamlit pandas numpy sentence-transformers scikit-learn plotly torch

Usage:
    streamlit run app.py

Data files needed:
    - data/processed/jobs_clean.csv
    - data/processed/programs.csv
    - data/processed/program_skills.csv
    - data/processed/skill_dictionary.csv

Note: Uses paraphrase-multilingual-MiniLM-L12-v2 model for memory efficiency.
      Falls back to paraphrase-MiniLM-L6-v2 if memory issues occur.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import json
from datetime import datetime
import os

# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================

# Fix path to work from dashboard directory
DATA_DIR = Path("../data/processed")
EMBEDDINGS_DIR = DATA_DIR
# Using a smaller, more memory-efficient model
# Alternative: "paraphrase-multilingual-MiniLM-L12-v2" (smaller than mpnet)
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

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
        "match_score": "درجة المطابقة",
        "skill_gaps": "المهارات المفقودة",
        "recommended_programs": "🎓 البرامج الموصى بها",
        "action_plan": "📋 خطة العمل الخاصة بك",
        "download": "تحميل خطتي",
        "why": "لماذا هذا الاقتراح؟",
        "feedback": "هل كان هذا مفيداً؟",
    }
}

EDUCATION_LEVELS = ["High School", "Bachelor/License"]
TUNISIA_CITIES = ["Tunis", "Sfax", "Sousse", "Bizerte", "Gabès", "Ariana", "Kairouan", "Nabeul", "Monastir", "Ben Arous"]

# =============================================================================
# CACHING & DATA LOADING
# =============================================================================

@st.cache_resource
def load_model():
    """Load and cache the sentence transformer model."""
    # AI BACKEND COMMENTED OUT FOR UI DEVELOPMENT
    # Returning None to skip model loading
    return None
    
    # try:
    #     return SentenceTransformer(MODEL_NAME)
    # except (OSError, MemoryError) as e:
    #     st.error(f"""
    #     ⚠️ **Memory Error Loading AI Model**
    #     
    #     The model requires more memory than available. Try these solutions:
    #     1. Close other applications to free up memory
    #     2. Restart your computer to clear memory
    #     3. Use a smaller model (already attempting this)
    #     
    #     Error: {str(e)}
    #     """)
    #     # Try an even smaller model as fallback
    #     try:
    #         st.info("Attempting to load a smaller model...")
    #         return SentenceTransformer("paraphrase-MiniLM-L6-v2")
    #     except Exception as e2:
    #         st.error(f"Fallback model also failed: {str(e2)}")
    #         st.stop()

@st.cache_data
def load_data():
    """Load all CSV files and prepare data structures."""
    try:
        jobs_df = pd.read_csv(DATA_DIR / "jobs_clean.csv")
        programs_df = pd.read_csv(DATA_DIR / "programs.csv")
        program_skills_df = pd.read_csv(DATA_DIR / "program_skills.csv")
        skill_dict_df = pd.read_csv(DATA_DIR / "skill_dictionary.csv")
        
        # Clean and prepare data
        jobs_df['skills'] = jobs_df['skills'].fillna('')
        jobs_df['description'] = jobs_df['description'].fillna('')
        programs_df['description'] = programs_df['description'].fillna('')
        
        return jobs_df, programs_df, program_skills_df, skill_dict_df
    except FileNotFoundError as e:
        st.error(f"❌ Data file not found: {e}")
        st.stop()

@st.cache_data
def load_or_compute_embeddings(_model, texts, cache_file):
    """Load embeddings from cache or compute them in batches to save memory."""
    # AI BACKEND COMMENTED OUT FOR UI DEVELOPMENT
    # Return mock embeddings (random vectors)
    return np.random.rand(len(texts), 384)  # 384 is typical embedding dimension
    
    # cache_path = EMBEDDINGS_DIR / cache_file
    # 
    # if cache_path.exists():
    #     return np.load(cache_path)
    # else:
    #     # Process in smaller batches to avoid memory issues
    #     batch_size = 32
    #     all_embeddings = []
    #     
    #     for i in range(0, len(texts), batch_size):
    #         batch = texts[i:i + batch_size]
    #         batch_embeddings = _model.encode(batch, show_progress_bar=False)
    #         all_embeddings.append(batch_embeddings)
    #     
    #     embeddings = np.vstack(all_embeddings)
    #     
    #     # Create directory if it doesn't exist
    #     cache_path.parent.mkdir(parents=True, exist_ok=True)
    #     np.save(cache_path, embeddings)
    #     return embeddings

# =============================================================================
# SKILL MAPPING & PROCESSING
# =============================================================================

def map_skills_to_canonical(user_skills, skill_dict_df, model):
    """
    Map user-entered skills to canonical skills using dictionary lookup
    and semantic similarity for unknown skills.
    
    Returns: list of (original_skill, canonical_skill, confidence)
    """
    # AI BACKEND COMMENTED OUT FOR UI DEVELOPMENT
    # Simple pass-through mapping for UI testing
    mapped_skills = []
    for skill in user_skills:
        skill_clean = skill.strip()
        if skill_clean:
            mapped_skills.append((skill_clean, skill_clean, 1.0))
    return mapped_skills
    
    # mapped_skills = []
    # skill_dict_lower = skill_dict_df.copy()
    # skill_dict_lower['variant_lower'] = skill_dict_lower['variant'].str.lower()
    # 
    # # Get unique canonical skills for semantic matching
    # canonical_skills = skill_dict_df['canonical_skill'].unique()
    # 
    # for skill in user_skills:
    #     skill_clean = skill.strip()
    #     skill_lower = skill_clean.lower()
    #     
    #     # Try exact match first
    #     match = skill_dict_lower[skill_dict_lower['variant_lower'] == skill_lower]
    #     
    #     if not match.empty:
    #         canonical = match.iloc[0]['canonical_skill']
    #         mapped_skills.append((skill_clean, canonical, 1.0))
    #     else:
    #         # Semantic matching for unknown skills
    #         if skill_clean:
    #             skill_emb = model.encode([skill_clean])
    #             canonical_embs = model.encode(canonical_skills.tolist())
    #             similarities = cosine_similarity(skill_emb, canonical_embs)[0]
    #             
    #             best_idx = np.argmax(similarities)
    #             best_score = similarities[best_idx]
    #             
    #             if best_score > 0.5:  # Threshold for semantic match
    #                 mapped_skills.append((skill_clean, canonical_skills[best_idx], float(best_score)))
    #             else:
    #                 # Keep original if no good match
    #                 mapped_skills.append((skill_clean, skill_clean, 0.3))
    # 
    # return mapped_skills

def extract_job_skills(job_skills_str):
    """Extract and clean skills from job skills string."""
    if pd.isna(job_skills_str) or job_skills_str == '':
        return []
    return [s.strip() for s in str(job_skills_str).split(';') if s.strip()]

# =============================================================================
# RECOMMENDATION ALGORITHMS
# =============================================================================

def compute_user_embedding(model, user_data, mapped_skills):
    """Compute embedding for user profile."""
    # AI BACKEND COMMENTED OUT FOR UI DEVELOPMENT
    # Return a mock embedding (random vector)
    return np.random.rand(384)
    
    # # Combine all user information into a rich text representation
    # canonical_skills = [skill[1] for skill in mapped_skills]
    # 
    # user_text = f"{user_data['education_level']} {user_data['major']} "
    # user_text += " ".join(canonical_skills) + " "
    # user_text += " ".join(user_data.get('strengths', [])) + " "
    # user_text += f"{user_data['experience']} years experience"
    # 
    # return model.encode([user_text])[0]

def recommend_jobs(user_embedding, jobs_df, job_embeddings, user_canonical_skills, top_k=5):
    """
    Recommend top jobs based on cosine similarity and skill matching.
    
    Returns: DataFrame with top job recommendations and details
    """
    # Compute cosine similarities
    similarities = cosine_similarity([user_embedding], job_embeddings)[0]
    
    # Get top K indices
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    recommendations = []
    user_skills_set = set(user_canonical_skills)
    
    for idx in top_indices:
        job = jobs_df.iloc[idx]
        job_skills = extract_job_skills(job['skills'])
        job_skills_set = set(job_skills)
        
        # Calculate skill coverage
        matching_skills = user_skills_set.intersection(job_skills_set)
        missing_skills = job_skills_set - user_skills_set
        
        coverage = len(matching_skills) / len(job_skills_set) if job_skills_set else 0
        
        recommendations.append({
            'job_id': job['job_id'],
            'title': job['title'],
            'company': job['company'],
            'location': job['location'],
            'description': job['description'],
            'required_skills': list(job_skills_set),
            'matching_skills': list(matching_skills),
            'missing_skills': list(missing_skills),
            'cosine_score': float(similarities[idx]),
            'skill_coverage': coverage,
            'sector': job.get('sector', 'N/A')
        })
    
    return recommendations

def recommend_programs_for_job(job_required_skills, programs_df, program_skills_df, top_k=3):
    """
    Recommend university programs that best cover the required skills for a job.
    
    Returns: list of (program_id, program_name, university, coverage_score, matched_skills)
    """
    required_skills_set = set(job_required_skills)
    program_scores = []
    
    for program_id in programs_df['program_id'].unique():
        # Get all skills taught in this program
        program_skill_rows = program_skills_df[program_skills_df['program_id'] == program_id]
        
        if program_skill_rows.empty:
            continue
        
        # Get canonical skills with confidence weighting
        program_skills = {}
        for _, row in program_skill_rows.iterrows():
            skill = row['canonical_skill']
            confidence = row.get('confidence', 1.0)
            if skill in program_skills:
                program_skills[skill] = max(program_skills[skill], confidence)
            else:
                program_skills[skill] = confidence
        
        # Calculate coverage
        program_skills_set = set(program_skills.keys())
        matched_skills = required_skills_set.intersection(program_skills_set)
        
        if matched_skills:
            # Weighted coverage by confidence
            coverage_score = sum(program_skills[s] for s in matched_skills) / len(required_skills_set)
            
            program_info = programs_df[programs_df['program_id'] == program_id].iloc[0]
            program_scores.append({
                'program_id': program_id,
                'program_name': program_info['program_name'],
                'university': program_info['university'],
                'faculty': program_info.get('faculty', 'N/A'),
                'degree_level': program_info.get('degree_level', 'N/A'),
                'duration': program_info.get('duration', 'N/A'),
                'coverage_score': coverage_score,
                'matched_skills': list(matched_skills),
                'total_skills': len(program_skills_set)
            })
    
    # Sort by coverage score
    program_scores.sort(key=lambda x: x['coverage_score'], reverse=True)
    return program_scores[:top_k]

def calculate_final_score(cosine_score, program_coverage, skill_gap_ratio):
    """
    Calculate final recommendation score.
    
    Formula: 0.6 * cosine + 0.3 * program_coverage - 0.1 * gap_penalty
    """
    return 0.6 * cosine_score + 0.3 * program_coverage - 0.1 * skill_gap_ratio

def generate_action_plan(missing_skills, program_recommendations):
    """Generate a personalized action plan with steps and timeline."""
    plan = {
        'immediate_actions': [],
        'short_term': [],
        'long_term': [],
        'estimated_timeline_weeks': 0
    }
    
    # Immediate actions
    if missing_skills:
        plan['immediate_actions'].append(
            f"📚 Focus on learning these {len(missing_skills)} priority skills: " + 
            ", ".join(list(missing_skills)[:3])
        )
        plan['immediate_actions'].append(
            "🔍 Research online courses on Coursera, Udemy, or edX for these skills"
        )
    
    # Short-term actions (1-3 months)
    if program_recommendations:
        top_program = program_recommendations[0]
        plan['short_term'].append(
            f"🎓 Consider enrolling in '{top_program['program_name']}' at {top_program['university']}"
        )
        plan['short_term'].append(
            "💼 Start building a portfolio with projects demonstrating your acquired skills"
        )
        plan['short_term'].append(
            "🤝 Network with professionals in your target industry via LinkedIn"
        )
        plan['estimated_timeline_weeks'] = 12
    else:
        plan['estimated_timeline_weeks'] = 8
    
    # Long-term actions (3-6 months)
    plan['long_term'].append("🎯 Apply to at least 5 relevant positions per week")
    plan['long_term'].append("📊 Track your progress and update your skills regularly")
    plan['long_term'].append("🌟 Obtain certifications for your top 3 most valuable skills")
    
    # Suggested courses (synthetic placeholders)
    plan['suggested_courses'] = []
    for skill in list(missing_skills)[:5]:
        plan['suggested_courses'].append({
            'skill': skill,
            'course': f"Mastering {skill} - Online Course",
            'platform': 'Coursera/Udemy',
            'duration': '4-6 weeks'
        })
    
    return plan

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_radar_chart(user_skills, job_skills, job_title):
    """Create a radar chart comparing user skills vs job requirements."""
    # Get union of all skills
    all_skills = list(set(user_skills + job_skills))[:8]  # Limit to 8 for readability
    
    # Calculate presence (1 if skill exists, 0 otherwise)
    user_values = [1 if skill in user_skills else 0 for skill in all_skills]
    job_values = [1 if skill in job_skills else 0 for skill in all_skills]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=user_values,
        theta=all_skills,
        fill='toself',
        name='Your Skills',
        line_color='#3b82f6'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=job_values,
        theta=all_skills,
        fill='toself',
        name='Job Requirements',
        line_color='#ef4444'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        showlegend=True,
        title=f"Skill Match: {job_title}",
        height=400
    )
    
    return fig

def create_program_heatmap(programs, all_skills):
    """Create a heatmap showing program coverage of top skills."""
    if not programs or not all_skills:
        return None
    
    # Prepare data matrix
    matrix = []
    program_names = []
    
    for prog in programs[:5]:  # Top 5 programs
        program_names.append(f"{prog['university'][:20]}...")
        row = [1 if skill in prog['matched_skills'] else 0 for skill in all_skills[:10]]
        matrix.append(row)
    
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=all_skills[:10],
        y=program_names,
        colorscale='Blues',
        showscale=True
    ))
    
    fig.update_layout(
        title="Program Coverage of Top Skills",
        xaxis_title="Skills",
        yaxis_title="Programs",
        height=300
    )
    
    return fig

def create_top_skills_chart(jobs_df):
    """Create a bar chart of most demanded skills."""
    all_skills = []
    for skills_str in jobs_df['skills'].dropna():
        all_skills.extend(extract_job_skills(skills_str))
    
    # Count occurrences
    from collections import Counter
    skill_counts = Counter(all_skills)
    top_10 = skill_counts.most_common(10)
    
    skills, counts = zip(*top_10) if top_10 else ([], [])
    
    fig = go.Figure(data=[
        go.Bar(x=list(counts), y=list(skills), orientation='h', marker_color='#10b981')
    ])
    
    fig.update_layout(
        title="Top 10 Most Demanded Skills in Market",
        xaxis_title="Number of Job Posts",
        yaxis_title="Skill",
        height=400
    )
    
    return fig

# =============================================================================
# UI RENDERING FUNCTIONS
# =============================================================================

def inject_custom_css():
    """Inject custom CSS for a clean, minimalistic design."""
    st.markdown("""
        <style>
        /* Main styling */
        .main {
            background-color: #f8fafc;
        }
        
        /* Card styling */
        .card {
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
            color: #1f2937;
        }
        
        .card h3 {
            color: #111827;
            margin-top: 0;
        }
        
        .card p {
            color: #4b5563;
        }
        
        .card strong {
            color: #374151;
        }
        
        /* Header styling */
        .header {
            text-align: center;
            padding: 2rem 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 12px;
            margin-bottom: 2rem;
        }
        
        /* Score badge */
        .score-badge {
            display: inline-block;
            padding: 0.5rem 1rem;
            background: #10b981;
            color: white;
            border-radius: 20px;
            font-weight: bold;
            margin: 0.5rem;
        }
        
        /* Skill tag */
        .skill-tag {
            display: inline-block;
            padding: 0.3rem 0.8rem;
            background: #e0e7ff;
            color: #4c1d95;
            border-radius: 12px;
            margin: 0.2rem;
            font-size: 0.9rem;
        }
        
        .missing-skill-tag {
            background: #fee2e2;
            color: #991b1b;
        }
        
        /* Button styling */
        .stButton>button {
            border-radius: 8px;
            border: none;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: 600;
            padding: 0.75rem 2rem;
            transition: all 0.3s;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }
        
        /* Arabic text support */
        .rtl {
            direction: rtl;
            text-align: right;
        }
        </style>
    """, unsafe_allow_html=True)

def render_header(lang):
    """Render the app header."""
    t = TRANSLATIONS[lang]
    
    if lang == 'ar':
        st.markdown(f"""
            <div class="header rtl">
                <h1>🎓 {t['title']}</h1>
                <p style="font-size: 1.2rem; margin-top: 0.5rem;">{t['subtitle']}</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="header">
                <h1>🎓 {t['title']}</h1>
                <p style="font-size: 1.2rem; margin-top: 0.5rem;">{t['subtitle']}</p>
            </div>
        """, unsafe_allow_html=True)

def render_job_card(job, idx, lang, programs, action_plan):
    """Render a single job recommendation card."""
    t = TRANSLATIONS[lang]
    
    # Calculate final score
    program_coverage = programs[0]['coverage_score'] if programs else 0
    skill_gap_ratio = len(job['missing_skills']) / max(len(job['required_skills']), 1)
    final_score = calculate_final_score(job['cosine_score'], program_coverage, skill_gap_ratio)
    
    st.markdown(f"""
        <div class="card">
            <h3>#{idx + 1} {job['title']} at {job['company']}</h3>
            <p><strong>📍 Location:</strong> {job['location']} | <strong>🏢 Sector:</strong> {job['sector']}</p>
            <div>
                <span class="score-badge">{t['match_score']}: {final_score:.1%}</span>
                <span class="score-badge" style="background: #3b82f6;">Cosine: {job['cosine_score']:.1%}</span>
                <span class="score-badge" style="background: #f59e0b;">Coverage: {job['skill_coverage']:.1%}</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Tabs for details
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Skills Analysis", "🎓 Programs", "📋 Action Plan", "❓ Why This?"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**✅ Matching Skills:**")
            for skill in job['matching_skills'][:10]:
                st.markdown(f'<span class="skill-tag">{skill}</span>', unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"**❌ {t['skill_gaps']}:**")
            for skill in job['missing_skills'][:10]:
                st.markdown(f'<span class="skill-tag missing-skill-tag">{skill}</span>', unsafe_allow_html=True)
        
        # Radar chart
        if job['matching_skills'] or job['missing_skills']:
            user_skills = [s[1] for s in st.session_state.get('mapped_skills', [])]
            fig = create_radar_chart(user_skills[:8], job['required_skills'][:8], job['title'])
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown(f"**{t['recommended_programs']}**")
        if programs:
            for i, prog in enumerate(programs, 1):
                st.markdown(f"""
                    **{i}. {prog['program_name']}**
                    - 🏛️ {prog['university']} - {prog['faculty']}
                    - 🎓 {prog['degree_level']} ({prog['duration']})
                    - 📊 Coverage: {prog['coverage_score']:.1%} ({len(prog['matched_skills'])} skills matched)
                """)
        else:
            st.info("No specific programs found. Consider general skill development courses.")
    
    with tab3:
        st.markdown(f"**{t['action_plan']}**")
        st.markdown(f"⏱️ **Estimated Timeline:** {action_plan['estimated_timeline_weeks']} weeks")
        
        st.markdown("**Immediate Actions (Week 1-2):**")
        for action in action_plan['immediate_actions']:
            st.markdown(f"- {action}")
        
        st.markdown("**Short-term Goals (Month 1-3):**")
        for action in action_plan['short_term']:
            st.markdown(f"- {action}")
        
        st.markdown("**Long-term Goals (Month 3-6):**")
        for action in action_plan['long_term']:
            st.markdown(f"- {action}")
        
        if action_plan['suggested_courses']:
            st.markdown("**📚 Suggested Courses:**")
            for course in action_plan['suggested_courses'][:3]:
                st.markdown(f"- **{course['skill']}**: {course['course']} ({course['platform']}, {course['duration']})")
    
    with tab4:
        st.markdown("**🔍 Explanation:**")
        st.markdown(f"""
        This job was recommended based on:
        - **Semantic Similarity:** {job['cosine_score']:.1%} - Your profile description matches the job description
        - **Skill Match:** {job['skill_coverage']:.1%} - You have {len(job['matching_skills'])} out of {len(job['required_skills'])} required skills
        - **Program Support:** {program_coverage:.1%} - Available programs can help you gain missing skills
        
        **Scoring Formula:**
        - Final Score = 0.6 × Cosine + 0.3 × Program Coverage - 0.1 × Gap Penalty
        - Your Score: 0.6 × {job['cosine_score']:.2f} + 0.3 × {program_coverage:.2f} - 0.1 × {skill_gap_ratio:.2f} = **{final_score:.2%}**
        """)
        
        st.markdown("**Top Keywords from Job Description:**")
        # Simple keyword extraction (top words)
        words = job['description'].lower().split()
        from collections import Counter
        word_freq = Counter([w for w in words if len(w) > 4])
        top_words = [w for w, _ in word_freq.most_common(5)]
        st.write(", ".join(top_words))
    
    # Feedback
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        if st.button("👍", key=f"like_{idx}"):
            save_feedback(job['job_id'], "positive")
            st.success("Thanks for your feedback!")
    with col2:
        if st.button("👎", key=f"dislike_{idx}"):
            save_feedback(job['job_id'], "negative")
            st.success("Thanks for your feedback!")

def save_feedback(job_id, sentiment):
    """Save user feedback to CSV."""
    feedback_file = Path("data/feedback.csv")
    feedback_file.parent.mkdir(parents=True, exist_ok=True)
    
    feedback_data = {
        'timestamp': datetime.now().isoformat(),
        'job_id': job_id,
        'sentiment': sentiment
    }
    
    # Append to CSV
    df = pd.DataFrame([feedback_data])
    if feedback_file.exists():
        df.to_csv(feedback_file, mode='a', header=False, index=False)
    else:
        df.to_csv(feedback_file, mode='w', header=True, index=False)

def export_plan(recommendations, lang):
    """Export recommendation plan as JSON."""
    export_data = {
        'generated_at': datetime.now().isoformat(),
        'language': lang,
        'recommendations': recommendations
    }
    
    return json.dumps(export_data, indent=2, ensure_ascii=False)

# =============================================================================
# MAIN APP
# =============================================================================

def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="Masar — EduMatch Tunisia",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    inject_custom_css()
    
   
    # Load model and data
    with st.spinner("Loading AI model..."):
        model = load_model()
        jobs_df, programs_df, program_skills_df, skill_dict_df = load_data()
    
    # Sidebar - User Input Form
    with st.sidebar:
        # Logo/Brand area with emoji
        st.markdown("""
            <div style="text-align: center; padding: 1rem 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 1rem;">
                <h1 style="color: white; margin: 0; font-size: 2rem;">🎓 مسار</h1>
                <p style="color: #e0e7ff; margin: 0.5rem 0 0 0; font-size: 0.9rem;">MASAR</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Language selector
        lang = st.selectbox("🌍 Language / اللغة", ["en", "fr", "ar"], index=0)
        t = TRANSLATIONS[lang]
        
        st.markdown("---")
        st.header("Your Profile" if lang != "ar" else "ملفك الشخصي")
        
        # Education level
        education_level = st.selectbox(t['education_level'], EDUCATION_LEVELS)
        
        # Current major
        major = st.text_input(t['current_major'], placeholder="e.g., Computer Science, Business")
        
        # Experience
        experience = st.slider(t['experience'], 0, 20, 0)
        
        # Skills input
        skills_text = st.text_area(
            t['skills'],
            placeholder="Python\nData Analysis\nProject Management",
            height=150
        )
        
        # Strengths
        strengths_text = st.text_input(
            t['strengths'],
            placeholder="Leadership, Communication, Problem-solving"
        )
        
        # Weaknesses
        weaknesses_text = st.text_input(
            t['weaknesses'],
            placeholder="Public Speaking, Time Management"
        )
        
        # Locations
        locations = st.multiselect(t['locations'], TUNISIA_CITIES, default=["Tunis"])
        
        st.markdown("---")
        
        # Quick example button
        if st.button("💡 Fill Example Data"):
            st.session_state['example_mode'] = True
            st.rerun()
        
        # Find match button
        find_match = st.button(t['find_match'], type="primary", use_container_width=True)
    
    # Handle example data
    if st.session_state.get('example_mode', False):
        skills_text = "Python\nData Analysis\nMachine Learning\nSQL\nVisualization"
        strengths_text = "Problem-solving, Quick learner, Team player"
        major = "Computer Science"
        experience = 2
        st.session_state['example_mode'] = False
    
    # Main area
    render_header(lang)
    
    if find_match and major and skills_text:
        with st.spinner(t['loading']):
            # Process user input
            user_skills_list = [s.strip() for s in skills_text.split('\n') if s.strip()]
            strengths_list = [s.strip() for s in strengths_text.split(',') if s.strip()]
            
            # Map skills to canonical
            mapped_skills = map_skills_to_canonical(user_skills_list, skill_dict_df, model)
            st.session_state['mapped_skills'] = mapped_skills
            canonical_skills = [skill[1] for skill in mapped_skills]
            
            # Prepare user data
            user_data = {
                'education_level': education_level,
                'major': major,
                'experience': experience,
                'skills': user_skills_list,
                'strengths': strengths_list,
                'locations': locations
            }
            
            # Compute embeddings
            job_texts = (jobs_df['title'] + ' ' + jobs_df['description']).tolist()
            job_embeddings = load_or_compute_embeddings(model, job_texts, "jobs_embeddings.npy")
            
            # Compute user embedding
            user_embedding = compute_user_embedding(model, user_data, mapped_skills)
            
            # Get job recommendations
            job_recommendations = recommend_jobs(
                user_embedding, 
                jobs_df, 
                job_embeddings, 
                canonical_skills,
                top_k=5
            )
            
            # Display results
            st.markdown(f"## {t['top_jobs']}")
            
            all_recommendations = []
            
            for idx, job in enumerate(job_recommendations):
                # Get program recommendations for this job
                programs = recommend_programs_for_job(
                    job['required_skills'],
                    programs_df,
                    program_skills_df,
                    top_k=3
                )
                
                # Generate action plan
                action_plan = generate_action_plan(set(job['missing_skills']), programs)
                
                # Store for export
                all_recommendations.append({
                    'job': job,
                    'programs': programs,
                    'action_plan': action_plan
                })
                
                # Render job card
                render_job_card(job, idx, lang, programs, action_plan)
            
            # Visualizations section
            st.markdown("---")
            st.markdown("## 📊 Market Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Top demanded skills chart
                fig_skills = create_top_skills_chart(jobs_df)
                st.plotly_chart(fig_skills, use_container_width=True)
            
            with col2:
                # Program heatmap
                if all_recommendations:
                    all_job_skills = []
                    all_programs = []
                    for rec in all_recommendations:
                        all_job_skills.extend(rec['job']['required_skills'])
                        all_programs.extend(rec['programs'])
                    
                    # Get unique top skills
                    from collections import Counter
                    top_skills = [skill for skill, _ in Counter(all_job_skills).most_common(10)]
                    
                    if all_programs:
                        fig_heatmap = create_program_heatmap(all_programs, top_skills)
                        if fig_heatmap:
                            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Download button
            st.markdown("---")
            export_json = export_plan(all_recommendations, lang)
            st.download_button(
                label=f"📥 {t['download']}",
                data=export_json,
                file_name=f"masar_plan_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
            
    elif find_match:
        st.warning("Please fill in at least your major/field and some skills to get recommendations.")
    else:
        # Welcome message
        st.info("👈 Fill in your profile in the sidebar and click 'Find My Perfect Match' to get personalized career recommendations!")
        
        # Show market insights by default
        st.markdown("## 📊 Current Job Market Trends")
        fig_skills = create_top_skills_chart(jobs_df)
        st.plotly_chart(fig_skills, use_container_width=True)

if __name__ == "__main__":
    main()
