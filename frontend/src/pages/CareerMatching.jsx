import React, { useState } from 'react';
import { Search, Briefcase, MapPin, TrendingUp } from 'lucide-react';
import { PieChart, Pie, Cell, ResponsiveContainer, Legend, BarChart, Bar, XAxis, YAxis, Tooltip } from 'recharts';

const CareerMatching = () => {
  const [profile, setProfile] = useState({
    educationLevel: '',
    major: '',
    experience: 0,
    skills: '',
    strengths: [],
    weaknesses: [],
    locations: [],
  });
  
  const [showResults, setShowResults] = useState(false);
  const [loading, setLoading] = useState(false);
  
  // Sample data
  const sampleJobs = [
    { title: 'Software Developer', company: 'TechCorp Tunisia', location: 'Tunis', match: 95, salary: '2000-3000 TND' },
    { title: 'Data Analyst', company: 'DataFlow', location: 'Sfax', match: 88, salary: '1800-2500 TND' },
    { title: 'Web Developer', company: 'WebSolutions', location: 'Sousse', match: 82, salary: '1500-2200 TND' },
    { title: 'Business Analyst', company: 'ConsultPro', location: 'Tunis', match: 75, salary: '1700-2400 TND' },
  ];
  
  const samplePrograms = [
    { name: 'Master in Computer Science', university: 'ENSI', match: 92 },
    { name: 'Data Science Bootcamp', university: 'GoMyCode', match: 85 },
    { name: 'Digital Marketing', university: 'ISIMS', match: 78 },
  ];
  
  const sectorData = [
    { name: 'Technology', value: 35 },
    { name: 'Healthcare', value: 20 },
    { name: 'Finance', value: 15 },
    { name: 'Education', value: 18 },
    { name: 'Tourism', value: 12 },
  ];
  
  const COLORS = ['#3b82f6', '#2563eb', '#1d4ed8', '#60a5fa', '#93c5fd'];
  
  const strengthOptions = ['Leadership', 'Problem Solving', 'Communication', 'Teamwork', 'Creativity', 'Analytical Thinking', 'Time Management'];
  const weaknessOptions = ['Public Speaking', 'Technical Writing', 'Networking', 'Foreign Languages', 'Advanced Math', 'Presentation Skills'];
  const locationOptions = ['Tunis', 'Sfax', 'Sousse', 'Monastir', 'Bizerte', 'Gabes', 'Kairouan', 'Ariana', 'Remote Work'];
  
  const handleSubmit = (e) => {
    e.preventDefault();
    setLoading(true);
    setTimeout(() => {
      setLoading(false);
      setShowResults(true);
    }, 2000);
  };
  
  const handleChange = (e) => {
    setProfile({
      ...profile,
      [e.target.name]: e.target.value,
    });
  };
  
  const toggleArrayItem = (key, value) => {
    const current = profile[key];
    if (current.includes(value)) {
      setProfile({
        ...profile,
        [key]: current.filter(item => item !== value),
      });
    } else {
      setProfile({
        ...profile,
        [key]: [...current, value],
      });
    }
  };
  
  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="card bg-gradient-to-r from-primary-50 to-primary-100 dark:from-primary-900/20 dark:to-primary-800/20">
        <h1 className="text-3xl font-bold text-gray-800 dark:text-gray-100 mb-2">Career Matching</h1>
        <p className="text-gray-600 dark:text-gray-400">Find the perfect job match based on your skills and preferences</p>
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Profile Form */}
        <div className="lg:col-span-1">
          <div className="card sticky top-4">
            <h2 className="text-xl font-bold text-gray-800 dark:text-gray-100 mb-6">Your Profile</h2>
            
            <form onSubmit={handleSubmit} className="space-y-4">
              {/* Education Level */}
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Education Level
                </label>
                <select
                  name="educationLevel"
                  value={profile.educationLevel}
                  onChange={handleChange}
                  className="input-field"
                  required
                >
                  <option value="">Select level</option>
                  <option value="high-school">High School</option>
                  <option value="bachelors">Bachelor's</option>
                  <option value="masters">Master's</option>
                  <option value="phd">PhD</option>
                  <option value="vocational">Vocational Training</option>
                </select>
              </div>
              
              {/* Major */}
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Current Major/Field
                </label>
                <input
                  type="text"
                  name="major"
                  value={profile.major}
                  onChange={handleChange}
                  className="input-field"
                  placeholder="e.g., Computer Science"
                  required
                />
              </div>
              
              {/* Experience */}
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Years of Experience: {profile.experience}
                </label>
                <input
                  type="range"
                  name="experience"
                  min="0"
                  max="20"
                  value={profile.experience}
                  onChange={handleChange}
                  className="w-full"
                />
              </div>
              
              {/* Skills */}
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Your Skills
                </label>
                <textarea
                  name="skills"
                  value={profile.skills}
                  onChange={handleChange}
                  className="input-field"
                  rows="4"
                  placeholder="Python&#10;JavaScript&#10;Data Analysis&#10;Project Management"
                />
              </div>
              
              {/* Strengths */}
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Your Strengths
                </label>
                <div className="flex flex-wrap gap-2">
                  {strengthOptions.map(strength => (
                    <button
                      key={strength}
                      type="button"
                      onClick={() => toggleArrayItem('strengths', strength)}
                      className={`px-3 py-1 rounded-full text-sm transition-colors ${
                        profile.strengths.includes(strength)
                          ? 'bg-primary-600 text-white'
                          : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-300 dark:hover:bg-gray-600'
                      }`}
                    >
                      {strength}
                    </button>
                  ))}
                </div>
              </div>
              
              {/* Weaknesses */}
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Areas to Improve
                </label>
                <div className="flex flex-wrap gap-2">
                  {weaknessOptions.map(weakness => (
                    <button
                      key={weakness}
                      type="button"
                      onClick={() => toggleArrayItem('weaknesses', weakness)}
                      className={`px-3 py-1 rounded-full text-sm transition-colors ${
                        profile.weaknesses.includes(weakness)
                          ? 'bg-blue-600 text-white'
                          : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-300 dark:hover:bg-gray-600'
                      }`}
                    >
                      {weakness}
                    </button>
                  ))}
                </div>
              </div>
              
              {/* Locations */}
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Preferred Locations
                </label>
                <div className="flex flex-wrap gap-2">
                  {locationOptions.map(location => (
                    <button
                      key={location}
                      type="button"
                      onClick={() => toggleArrayItem('locations', location)}
                      className={`px-3 py-1 rounded-full text-sm transition-colors ${
                        profile.locations.includes(location)
                          ? 'bg-green-600 text-white'
                          : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-300 dark:hover:bg-gray-600'
                      }`}
                    >
                      {location}
                    </button>
                  ))}
                </div>
              </div>
              
              {/* Submit Button */}
              <button
                type="submit"
                disabled={loading}
                className="w-full btn-primary flex items-center justify-center gap-2"
              >
                {loading ? (
                  <>
                    <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Search className="w-5 h-5" />
                    Find My Perfect Match
                  </>
                )}
              </button>
            </form>
          </div>
        </div>
        
        {/* Results Section */}
        <div className="lg:col-span-2 space-y-6">
          {!showResults ? (
            <div className="card text-center py-12">
              <div className="text-gray-400 dark:text-gray-500 mb-4">
                <Search className="w-16 h-16 mx-auto" />
              </div>
              <h3 className="text-xl font-semibold text-gray-700 dark:text-gray-300 mb-2">Ready to Find Your Match?</h3>
              <p className="text-gray-600 dark:text-gray-400">Fill out your profile and click "Find My Perfect Match" to get personalized recommendations</p>
              
              {/* Statistics */}
              <div className="mt-8">
                <h4 className="text-lg font-semibold text-gray-700 dark:text-gray-300 mb-4">Popular Career Paths</h4>
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={sectorData}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                      outerRadius={80}
                      fill="#8884d8"
                      dataKey="value"
                    >
                      {sectorData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </div>
          ) : (
            <>
              {/* Top Job Matches */}
              <div className="card">
                <h2 className="text-2xl font-bold text-gray-800 dark:text-gray-100 mb-6 flex items-center gap-2">
                  <Briefcase className="w-6 h-6 text-primary-600 dark:text-primary-400" />
                  Top Job Matches for You
                </h2>
                
                <div className="space-y-4">
                  {sampleJobs.map((job, index) => (
                    <div key={index} className="border border-gray-200 dark:border-gray-700 rounded-lg p-4 hover:shadow-md transition-shadow">
                      <div className="flex justify-between items-start mb-3">
                        <div>
                          <h3 className="text-lg font-bold text-gray-800 dark:text-gray-100">{job.title}</h3>
                          <p className="text-gray-600 dark:text-gray-400">{job.company}</p>
                          <div className="flex items-center gap-4 mt-2 text-sm text-gray-500 dark:text-gray-400">
                            <span className="flex items-center gap-1">
                              <MapPin className="w-4 h-4" />
                              {job.location}
                            </span>
                            <span className="flex items-center gap-1">
                              <TrendingUp className="w-4 h-4" />
                              {job.salary}
                            </span>
                          </div>
                        </div>
                        <div className="text-center">
                          <div className="text-3xl font-bold text-primary-600 dark:text-primary-400">{job.match}%</div>
                          <div className="text-xs text-gray-600 dark:text-gray-400">Match</div>
                        </div>
                      </div>
                      
                      <div className="flex gap-2">
                        <button className="btn-primary flex-1">View Details</button>
                        <button className="btn-secondary flex-1">Apply Now</button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
              
              {/* Recommended Programs */}
              <div className="card">
                <h2 className="text-2xl font-bold text-gray-800 dark:text-gray-100 mb-6">🎓 Recommended Programs</h2>
                
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {samplePrograms.map((program, index) => (
                    <div key={index} className="border border-gray-200 dark:border-gray-700 rounded-lg p-4 text-center hover:shadow-md transition-shadow">
                      <div className="text-2xl font-bold text-primary-600 dark:text-primary-400 mb-1">{program.match}%</div>
                      <div className="text-sm text-gray-600 dark:text-gray-400 mb-2">Match</div>
                      <h3 className="font-semibold text-gray-800 dark:text-gray-100 mb-1">{program.name}</h3>
                      <p className="text-sm text-gray-600 dark:text-gray-400">{program.university}</p>
                    </div>
                  ))}
                </div>
              </div>
              
              {/* Action Plan */}
              <div className="card">
                <h2 className="text-2xl font-bold text-gray-800 dark:text-gray-100 mb-6">📋 Your Action Plan</h2>
                
                <div className="space-y-6">
                  <div>
                    <h3 className="font-semibold text-gray-800 dark:text-gray-100 mb-3">📚 Short Term (Next 3 months)</h3>
                    <ul className="space-y-2 text-gray-700 dark:text-gray-300">
                      <li className="flex items-start">
                        <span className="text-primary-600 dark:text-primary-400 mr-2">•</span>
                        <span>Complete Python programming course</span>
                      </li>
                      <li className="flex items-start">
                        <span className="text-primary-600 dark:text-primary-400 mr-2">•</span>
                        <span>Build 2-3 portfolio projects</span>
                      </li>
                      <li className="flex items-start">
                        <span className="text-primary-600 dark:text-primary-400 mr-2">•</span>
                        <span>Practice coding interview questions</span>
                      </li>
                    </ul>
                  </div>
                  
                  <div>
                    <h3 className="font-semibold text-gray-800 dark:text-gray-100 mb-3">🎯 Medium Term (6-12 months)</h3>
                    <ul className="space-y-2 text-gray-700 dark:text-gray-300">
                      <li className="flex items-start">
                        <span className="text-primary-600 dark:text-primary-400 mr-2">•</span>
                        <span>Apply for internships in target companies</span>
                      </li>
                      <li className="flex items-start">
                        <span className="text-primary-600 dark:text-primary-400 mr-2">•</span>
                        <span>Attend tech meetups and networking events</span>
                      </li>
                      <li className="flex items-start">
                        <span className="text-primary-600 dark:text-primary-400 mr-2">•</span>
                        <span>Contribute to open source projects</span>
                      </li>
                    </ul>
                  </div>
                  
                  <div>
                    <h3 className="font-semibold text-gray-800 dark:text-gray-100 mb-3">🚀 Long Term (1-2 years)</h3>
                    <ul className="space-y-2 text-gray-700 dark:text-gray-300">
                      <li className="flex items-start">
                        <span className="text-primary-600 dark:text-primary-400 mr-2">•</span>
                        <span>Pursue Master's degree if needed</span>
                      </li>
                      <li className="flex items-start">
                        <span className="text-primary-600 dark:text-primary-400 mr-2">•</span>
                        <span>Apply for full-time positions</span>
                      </li>
                      <li className="flex items-start">
                        <span className="text-primary-600 dark:text-primary-400 mr-2">•</span>
                        <span>Build professional network</span>
                      </li>
                    </ul>
                  </div>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default CareerMatching;
