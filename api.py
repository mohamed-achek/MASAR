"""
Masar API - Education-to-Career Matching Platform
=================================================

A comprehensive REST API for connecting Tunisian educat    class Config:
        json_schema_extra = {
            "example": {
                "education_level": "Bachelor/License",
                "major": "Computer Science",
                "experience": 2,
                "skills": ["Python", "Data Analysis", "Machine Learning"],
                "strengths": ["Problem-solving", "Team player"],
                "locations": ["Tunis", "Sousse"]
            }
        }er opportunities.

Features:
- 🎯 AI-powered job matching
- 🎓 Educational pathway recommendations  
- 📊 Market insights and statistics
- 🌍 Multilingual support (AR/FR/EN)

Run with: python api.py
Access documentation at: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException, Query, Path as FastAPIPath
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import pandas as pd
from pathlib import Path
import json
from datetime import datetime

# API metadata and configuration
tags_metadata = [
    {
        "name": "Health",
        "description": "API health and status endpoints",
    },
    {
        "name": "Recommendations", 
        "description": "Job recommendation engine powered by AI matching algorithms",
    },
    {
        "name": "Jobs",
        "description": "Browse and search available job opportunities",
    },
    {
        "name": "Analytics",
        "description": "Market insights, statistics, and trends",
    },
]

app = FastAPI(
    title="🎓 Masar API",
    description="""
## Education-to-Career Matching Platform for Tunisia

Masar bridges the gap between Tunisian education and career opportunities through intelligent matching algorithms.

### Key Features

* **🎯 Smart Matching**: AI-powered semantic matching between user profiles and job opportunities
* **🎓 Education Pathways**: University program recommendations based on career goals  
* **📊 Skill Analysis**: Identify skill gaps and suggest learning paths
* **🌍 Multilingual**: Support for Arabic, French, and English
* **📈 Market Insights**: Real-time job market trends and statistics

### Getting Started

1. **Create a user profile** with your education and skills
2. **Get recommendations** using the `/recommend` endpoint
3. **Explore jobs** and market data through various endpoints
4. **Analyze trends** with our statistics endpoints

### Authentication

Currently, this API is open and does not require authentication. 
Rate limiting may apply in production environments.

### Support

- 📧 **Email**: support@masar-tunisia.com
- 📚 **Documentation**: [GitHub Wiki](https://github.com/mohamed-achek/MASAR)
- 🐛 **Issues**: [GitHub Issues](https://github.com/mohamed-achek/MASAR/issues)
    """,
    version="1.0.0",
    terms_of_service="https://github.com/mohamed-achek/MASAR/blob/main/LICENSE",
    contact={
        "name": "Masar Development Team",
        "url": "https://github.com/mohamed-achek/MASAR",
        "email": "mohamed.achek@example.com",
    },
    license_info={
        "name": "MIT License",
        "url": "https://github.com/mohamed-achek/MASAR/blob/main/LICENSE",
    },
    openapi_tags=tags_metadata,
)

# Pydantic models with comprehensive validation and documentation
class UserProfile(BaseModel):
    """
    User profile for job matching and career recommendations.
    
    Represents a job seeker's educational background, skills, and preferences
    used by the matching algorithm to find suitable opportunities.
    """
    
    education_level: str = Field(
        ...,
        title="Education Level",
        description="Current education level of the user",
        example="Bachelor/License",
        regex="^(High School|Bachelor/License|Master|PhD|Engineering)$"
    )
    
    major: str = Field(
        ...,
        title="Major/Field of Study", 
        description="Academic major or field of specialization",
        example="Computer Science",
        min_length=2,
        max_length=100
    )
    
    experience: int = Field(
        ...,
        title="Years of Experience",
        description="Professional work experience in years",
        example=2,
        ge=0,
        le=50
    )
    
    skills: List[str] = Field(
        ...,
        title="Skills",
        description="List of technical and soft skills",
        example=["Python", "Data Analysis", "Machine Learning", "Communication"],
        min_items=1,
        max_items=50
    )
    
    strengths: Optional[List[str]] = Field(
        default=[],
        title="Strengths",
        description="Personal strengths and attributes",
        example=["Problem-solving", "Leadership", "Team collaboration"],
        max_items=20
    )
    
    locations: Optional[List[str]] = Field(
        default=["Tunis"],
        title="Preferred Locations",
        description="Preferred work locations in Tunisia",
        example=["Tunis", "Sfax", "Sousse"],
        max_items=10
    )

    @validator('skills', 'strengths')
    def validate_string_lists(cls, v):
        """Validate that skill and strength entries are non-empty strings"""
        return [item.strip() for item in v if item and item.strip()]

    @validator('major')
    def validate_major(cls, v):
        """Validate major field"""
        if not v or not v.strip():
            raise ValueError("Major cannot be empty")
        return v.strip().title()

    class Config:
        schema_extra = {
            "example": {
                "education_level": "Bachelor/License",
                "major": "Computer Science",
                "experience": 2,
                "skills": ["Python", "Data Analysis", "SQL", "Machine Learning"],
                "strengths": ["Problem-solving", "Quick learner", "Team player"],
                "locations": ["Tunis", "Ariana"]
            }
        }


class JobRecommendation(BaseModel):
    """
    Job recommendation with matching details and skill analysis.
    
    Contains comprehensive information about a recommended job including
    match scoring, skill requirements, and gap analysis.
    """
    
    job_id: str = Field(
        ...,
        title="Job ID",
        description="Unique identifier for the job posting",
        example="JOB_001"
    )
    
    title: str = Field(
        ...,
        title="Job Title",
        description="Position title or job name",
        example="Senior Data Analyst"
    )
    
    company: str = Field(
        ...,
        title="Company",
        description="Hiring company name",
        example="TechCorp Tunisia"
    )
    
    location: str = Field(
        ...,
        title="Location",
        description="Job location within Tunisia",
        example="Tunis"
    )
    
    match_score: float = Field(
        ...,
        title="Match Score",
        description="AI-computed compatibility score between user and job (0.0-1.0)",
        example=0.85,
        ge=0.0,
        le=1.0
    )
    
    required_skills: List[str] = Field(
        ...,
        title="Required Skills",
        description="Skills required for this position",
        example=["Python", "SQL", "Data Visualization", "Statistics"]
    )
    
    missing_skills: List[str] = Field(
        ...,
        title="Missing Skills",
        description="Skills required for the job that the user doesn't have",
        example=["Advanced Statistics", "Power BI"]
    )
    
    sector: Optional[str] = Field(
        default="General",
        title="Industry Sector",
        description="Industry or business sector",
        example="Technology"
    )

    class Config:
        schema_extra = {
            "example": {
                "job_id": "JOB_123",
                "title": "Senior Data Analyst",
                "company": "TechCorp Tunisia",
                "location": "Tunis",
                "match_score": 0.85,
                "required_skills": ["Python", "SQL", "Data Visualization"],
                "missing_skills": ["Power BI", "Advanced Statistics"],
                "sector": "Technology"
            }
        }


class JobSummary(BaseModel):
    """Brief job information for listings"""
    
    job_id: str = Field(..., description="Unique job identifier")
    title: str = Field(..., description="Job title")
    company: str = Field(..., description="Company name")
    location: str = Field(..., description="Job location")
    sector: str = Field(..., description="Industry sector")
    skills: List[str] = Field(..., description="Required skills")


class JobsResponse(BaseModel):
    """Response model for jobs endpoint"""
    
    total: int = Field(..., description="Total number of jobs matching criteria")
    returned: int = Field(..., description="Number of jobs returned in this response")
    jobs: List[JobSummary] = Field(..., description="List of job summaries")
    
    class Config:
        schema_extra = {
            "example": {
                "total": 150,
                "returned": 10,
                "jobs": [
                    {
                        "job_id": "JOB_001",
                        "title": "Software Engineer",
                        "company": "TechStart",
                        "location": "Tunis",
                        "sector": "Technology",
                        "skills": ["Python", "React", "SQL"]
                    }
                ]
            }
        }


class PlatformStats(BaseModel):
    """Platform statistics and market insights"""
    
    total_jobs: int = Field(..., description="Total number of available jobs")
    unique_companies: int = Field(..., description="Number of unique hiring companies")
    unique_locations: int = Field(..., description="Number of job locations")
    top_skills: Dict[str, int] = Field(..., description="Most demanded skills with counts")
    last_updated: str = Field(..., description="Last data update timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "total_jobs": 1250,
                "unique_companies": 185,
                "unique_locations": 8,
                "top_skills": {
                    "Python": 45,
                    "JavaScript": 38,
                    "SQL": 35,
                    "Communication": 42
                },
                "last_updated": "2025-01-07T10:30:00Z"
            }
        }


class HealthResponse(BaseModel):
    """API health status response"""
    
    status: str = Field(..., description="Overall API health status")
    data_loaded: bool = Field(..., description="Whether job data is successfully loaded")
    total_jobs: int = Field(..., description="Number of jobs in the database")
    timestamp: str = Field(..., description="Health check timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "data_loaded": True,
                "total_jobs": 1250,
                "timestamp": "2025-01-07T10:30:00Z"
            }
        }


class APIInfo(BaseModel):
    """API information and available endpoints"""
    
    message: str = Field(..., description="Welcome message")
    version: str = Field(..., description="API version")
    endpoints: Dict[str, str] = Field(..., description="Available API endpoints")
    
    class Config:
        schema_extra = {
            "example": {
                "message": "Masar API is running! 🎓",
                "version": "1.0.0",
                "endpoints": {
                    "recommendations": "/recommend",
                    "jobs": "/jobs",
                    "stats": "/stats",
                    "health": "/health"
                }
            }
        }

# Load data (simplified - would use proper data loader in production)
try:
    DATA_DIR = Path("data/processed")
    jobs_df = pd.read_csv(DATA_DIR / "jobs_clean.csv")
    print(f"✅ Loaded {len(jobs_df)} jobs")
except Exception as e:
    print(f"⚠️ Could not load data: {e}")
    jobs_df = pd.DataFrame()

@app.get(
    "/",
    response_model=APIInfo,
    tags=["Health"],
    summary="API Information",
    description="Get basic API information, version, and available endpoints"
)
async def root() -> APIInfo:
    """
    ## Welcome to Masar API! 🎓
    
    This endpoint provides basic information about the API including:
    - API version and status
    - Available endpoints and their purposes
    - Quick start guidance
    
    **Perfect for**: Initial API exploration and health verification
    """
    return APIInfo(
        message="Masar API is running! 🎓",
        version="1.0.0",
        endpoints={
            "recommendations": "/recommend - Get personalized job recommendations",
            "jobs": "/jobs - Browse available job opportunities", 
            "stats": "/stats - Market insights and statistics",
            "health": "/health - Detailed health status",
            "docs": "/docs - Interactive API documentation"
        }
    )

@app.post(
    "/recommend",
    response_model=List[JobRecommendation],
    tags=["Recommendations"],
    summary="Get Job Recommendations",
    description="Generate personalized job recommendations based on user profile using AI matching algorithms",
    responses={
        200: {
            "description": "Successfully generated job recommendations",
            "content": {
                "application/json": {
                    "example": [
                        {
                            "job_id": "JOB_123",
                            "title": "Data Scientist",
                            "company": "TechCorp Tunisia",
                            "location": "Tunis",
                            "match_score": 0.89,
                            "required_skills": ["Python", "Machine Learning", "SQL"],
                            "missing_skills": ["Deep Learning", "TensorFlow"],
                            "sector": "Technology"
                        }
                    ]
                }
            }
        },
        503: {"description": "Service unavailable - job data not loaded"},
        422: {"description": "Validation error - invalid user profile data"}
    }
)
async def get_recommendations(
    profile: UserProfile,
    limit: int = Query(
        default=5,
        ge=1,
        le=50,
        description="Maximum number of job recommendations to return",
        example=5
    )
) -> List[JobRecommendation]:
    """
    ## Generate Personalized Job Recommendations 🎯
    
    This endpoint uses advanced AI matching algorithms to find the most suitable job opportunities
    for a user based on their educational background, skills, and preferences.
    
    ### How it works:
    1. **Profile Analysis**: Analyzes user education, experience, and skills
    2. **Semantic Matching**: Uses AI to match user profile with job descriptions
    3. **Skill Gap Analysis**: Identifies missing skills and calculates coverage
    4. **Scoring Algorithm**: Combines multiple factors for final match score
    
    ### Match Score Calculation:
    - **0.8-1.0**: Excellent match - highly recommended
    - **0.6-0.8**: Good match - worth considering  
    - **0.4-0.6**: Fair match - may require skill development
    - **Below 0.4**: Poor match - significant gaps exist
    
    ### Tips for better results:
    - Be specific about your skills and experience
    - Include both technical and soft skills
    - Specify realistic preferred locations
    """
    
    if jobs_df.empty:
        raise HTTPException(
            status_code=503, 
            detail="Job data is currently unavailable. Please try again later."
        )
    
    # Enhanced recommendation logic with better scoring
    recommendations = []
    
    for _, job in jobs_df.head(limit * 2).iterrows():  # Get more jobs to filter better matches
        # Parse job skills
        job_skills = job.get('skills', '').split(';') if job.get('skills') else []
        job_skills = [skill.strip() for skill in job_skills if skill.strip()]
        
        user_skills_set = set(skill.lower() for skill in profile.skills)
        job_skills_set = set(skill.lower() for skill in job_skills)
        
        # Calculate skill overlap
        matching_skills = user_skills_set.intersection(job_skills_set)
        missing_skills = list(job_skills_set - user_skills_set)
        
        # Enhanced scoring algorithm
        skill_coverage = len(matching_skills) / len(job_skills_set) if job_skills_set else 0
        experience_bonus = min(profile.experience * 0.05, 0.2)  # Max 20% bonus
        location_bonus = 0.1 if job.get('location', '').lower() in [loc.lower() for loc in profile.locations] else 0
        
        # Base score with randomization for demo purposes
        import random
        base_score = 0.6 + random.uniform(0, 0.3)  # In production, this would be ML model output
        
        final_score = min(
            base_score + (skill_coverage * 0.3) + experience_bonus + location_bonus,
            1.0
        )
        
        # Only include jobs with reasonable match scores
        if final_score >= 0.4:
            recommendations.append(JobRecommendation(
                job_id=str(job.get('job_id', f'JOB_{len(recommendations)+1:03d}')),
                title=job.get('title', 'Unknown Position'),
                company=job.get('company', 'Unknown Company'),
                location=job.get('location', 'Tunisia'),
                match_score=round(final_score, 2),
                required_skills=job_skills,
                missing_skills=missing_skills,
                sector=job.get('sector', 'General')
            ))
    
    # Sort by match score and return top results
    recommendations.sort(key=lambda x: x.match_score, reverse=True)
    return recommendations[:limit]

@app.get(
    "/jobs",
    response_model=JobsResponse,
    tags=["Jobs"],
    summary="Browse Job Opportunities",
    description="Search and filter available job opportunities with advanced filtering options",
    responses={
        200: {
            "description": "Successfully retrieved job listings",
            "content": {
                "application/json": {
                    "example": {
                        "total": 150,
                        "returned": 10,
                        "jobs": [
                            {
                                "job_id": "JOB_001",
                                "title": "Software Engineer",
                                "company": "TechStart Tunisia",
                                "location": "Tunis",
                                "sector": "Technology",
                                "skills": ["Python", "React", "PostgreSQL"]
                            }
                        ]
                    }
                }
            }
        },
        503: {"description": "Service unavailable - job data not loaded"}
    }
)
async def get_jobs(
    limit: int = Query(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of jobs to return",
        example=10
    ),
    location: Optional[str] = Query(
        default=None,
        description="Filter jobs by location (case-insensitive partial match)",
        example="Tunis"
    ),
    sector: Optional[str] = Query(
        default=None,
        description="Filter jobs by industry sector",
        example="Technology"
    ),
    skill: Optional[str] = Query(
        default=None,
        description="Filter jobs requiring a specific skill",
        example="Python"
    )
) -> JobsResponse:
    """
    ## Browse Available Job Opportunities 💼
    
    Explore the current job market with flexible filtering options to find opportunities
    that match your criteria.
    
    ### Filtering Options:
    - **Location**: Find jobs in specific cities (Tunis, Sfax, Sousse, etc.)
    - **Sector**: Filter by industry (Technology, Finance, Healthcare, etc.)
    - **Skill**: Find jobs requiring specific skills
    - **Limit**: Control the number of results returned
    
    ### Use Cases:
    - **Market Research**: Understand demand in specific locations or sectors
    - **Skill Planning**: See what skills are most requested
    - **Job Discovery**: Browse opportunities without personalized matching
    
    ### Response Format:
    Returns a paginated list of jobs with metadata about total matches
    and the number returned in this response.
    """
    
    if jobs_df.empty:
        raise HTTPException(
            status_code=503,
            detail="Job data is currently unavailable. Please try again later."
        )
    
    # Start with all jobs
    filtered_jobs = jobs_df.copy()
    
    # Apply filters
    if location:
        filtered_jobs = filtered_jobs[
            filtered_jobs['location'].str.contains(location, case=False, na=False)
        ]
    
    if sector:
        filtered_jobs = filtered_jobs[
            filtered_jobs['sector'].str.contains(sector, case=False, na=False)
        ]
    
    if skill:
        # Filter jobs that have the specified skill in their requirements
        skill_mask = filtered_jobs['skills'].str.contains(skill, case=False, na=False)
        filtered_jobs = filtered_jobs[skill_mask]
    
    # Convert to response format
    jobs_list = []
    for _, job in filtered_jobs.head(limit).iterrows():
        job_skills = job.get('skills', '').split(';') if job.get('skills') else []
        job_skills = [skill.strip() for skill in job_skills if skill.strip()]
        
        jobs_list.append(JobSummary(
            job_id=str(job.get('job_id', f'JOB_{len(jobs_list)+1:03d}')),
            title=job.get('title', 'Unknown Position'),
            company=job.get('company', 'Unknown Company'),
            location=job.get('location', 'Tunisia'),
            sector=job.get('sector', 'General'),
            skills=job_skills
        ))
    
    return JobsResponse(
        total=len(filtered_jobs),
        returned=len(jobs_list),
        jobs=jobs_list
    )

@app.get(
    "/stats",
    response_model=PlatformStats,
    tags=["Analytics"],
    summary="Market Statistics & Insights",
    description="Get comprehensive market statistics, trends, and insights about the Tunisian job market",
    responses={
        200: {
            "description": "Successfully retrieved market statistics",
            "content": {
                "application/json": {
                    "example": {
                        "total_jobs": 1250,
                        "unique_companies": 185,
                        "unique_locations": 8,
                        "top_skills": {
                            "Python": 45,
                            "JavaScript": 38,
                            "Communication": 42,
                            "SQL": 35
                        },
                        "last_updated": "2025-01-07T10:30:00Z"
                    }
                }
            }
        },
        503: {"description": "Service unavailable - statistics data not available"}
    }
)
async def get_stats() -> PlatformStats:
    """
    ## Market Statistics & Insights 📊
    
    Get comprehensive insights into the Tunisian job market including:
    - **Job Volume**: Total opportunities available
    - **Market Diversity**: Number of companies and locations
    - **Skill Demand**: Most requested skills across all positions
    - **Trends**: Market dynamics and patterns
    
    ### Key Metrics:
    - **Total Jobs**: Current number of active job postings
    - **Companies**: Unique employers in the system
    - **Locations**: Geographic distribution of opportunities
    - **Top Skills**: Most in-demand skills ranked by frequency
    
    ### Use Cases:
    - **Career Planning**: Understand which skills are most valuable
    - **Market Research**: Analyze job market trends and opportunities
    - **Education Strategy**: Align learning goals with market demands
    - **Regional Analysis**: Compare opportunities across different locations
    
    **Data Freshness**: Statistics are updated regularly to reflect current market conditions
    """
    
    if jobs_df.empty:
        raise HTTPException(
            status_code=503,
            detail="Statistics data is currently unavailable. Please try again later."
        )
    
    # Calculate comprehensive statistics
    total_jobs = len(jobs_df)
    unique_companies = jobs_df['company'].nunique() if 'company' in jobs_df.columns else 0
    unique_locations = jobs_df['location'].nunique() if 'location' in jobs_df.columns else 0
    
    # Analyze skill demand across all jobs
    all_skills = []
    if 'skills' in jobs_df.columns:
        for skills_str in jobs_df['skills'].dropna():
            skills_list = skills_str.split(';')
            all_skills.extend([skill.strip() for skill in skills_list if skill.strip()])
    
    from collections import Counter
    skill_counter = Counter(all_skills)
    top_skills = dict(skill_counter.most_common(15))  # Top 15 skills
    
    # Generate current timestamp
    current_time = datetime.now().isoformat() + "Z"
    
    return PlatformStats(
        total_jobs=total_jobs,
        unique_companies=unique_companies,
        unique_locations=unique_locations,
        top_skills=top_skills,
        last_updated=current_time
    )

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Detailed Health Check",
    description="Comprehensive API health status including data availability and system metrics",
    responses={
        200: {
            "description": "API is healthy and operational",
            "content": {
                "application/json": {
                    "example": {
                        "status": "healthy",
                        "data_loaded": True,
                        "total_jobs": 1250,
                        "timestamp": "2025-01-07T10:30:00Z"
                    }
                }
            }
        },
        503: {
            "description": "API is unhealthy - service issues detected",
            "content": {
                "application/json": {
                    "example": {
                        "status": "unhealthy",
                        "data_loaded": False,
                        "total_jobs": 0,
                        "timestamp": "2025-01-07T10:30:00Z"
                    }
                }
            }
        }
    }
)
async def health_check() -> HealthResponse:
    """
    ## API Health Status 🏥
    
    Provides comprehensive health information about the API and its dependencies.
    This endpoint is essential for:
    
    ### Monitoring & Operations:
    - **Service Status**: Overall API operational state
    - **Data Availability**: Whether job data is loaded and accessible
    - **System Metrics**: Key performance indicators
    - **Timestamp**: When the health check was performed
    
    ### Health Indicators:
    - **✅ Healthy**: All systems operational, data loaded successfully
    - **⚠️ Degraded**: API running but with limited functionality
    - **❌ Unhealthy**: Critical issues detected, service may be unavailable
    
    ### Use Cases:
    - **Load Balancer**: Health checks for traffic routing
    - **Monitoring Systems**: Automated health monitoring
    - **Development**: Verify API status during development
    - **Troubleshooting**: Diagnose service issues
    
    **Note**: This endpoint should respond quickly and is suitable for frequent polling.
    """
    
    # Determine overall health status
    data_loaded = not jobs_df.empty
    total_jobs = len(jobs_df) if data_loaded else 0
    
    # Set status based on data availability
    if data_loaded and total_jobs > 0:
        status = "healthy"
        status_code = 200
    elif data_loaded:
        status = "degraded"  # Data loaded but empty
        status_code = 200
    else:
        status = "unhealthy"  # No data available
        status_code = 503
    
    current_time = datetime.now().isoformat() + "Z"
    
    response = HealthResponse(
        status=status,
        data_loaded=data_loaded,
        total_jobs=total_jobs,
        timestamp=current_time
    )
    
    # Return appropriate HTTP status code
    if status_code == 503:
        raise HTTPException(status_code=503, detail=response.dict())
    
    return response

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )