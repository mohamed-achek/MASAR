import React from 'react';
import { Link } from 'react-router-dom';
import { Target, BookOpen, TrendingUp, Users, Briefcase, GraduationCap } from 'lucide-react';

const Dashboard = () => {
  const stats = [
    { label: 'Universities', value: '', change: '', icon: GraduationCap, color: 'blue' },
    { label: 'Skills Mapped', value: '', change: '', icon: TrendingUp, color: 'purple' },
  ];
  
  const features = [
    {
      title: 'Career Matching',
      description: 'Get personalized job recommendations based on your skills, education, and career goals.',
      icon: Target,
      link: '/career-matching',
      color: 'orange',
    },
    {
      title: 'Handbook Assistant',
      description: 'Ask questions about Tunisian universities policies, courses, and procedures. Get instant answers powered by AI.',
      icon: BookOpen,
      link: '/handbook',
      color: 'blue',
    },
  ];
  
  const getColorClasses = (color) => {
    const colors = {
      orange: 'bg-primary-100 dark:bg-primary-900/20 text-primary-600 dark:text-primary-400',
      blue: 'bg-blue-100 dark:bg-blue-900/20 text-blue-600 dark:text-blue-400',
      green: 'bg-green-100 dark:bg-green-900/20 text-green-600 dark:text-green-400',
      purple: 'bg-purple-100 dark:bg-purple-900/20 text-purple-600 dark:text-purple-400',
    };
    return colors[color] || colors.orange;
  };
  
  return (
    <div className="space-y-8">
      {/* Welcome Section */}
      <div className="bg-gradient-to-r from-primary-600 to-primary-700 dark:from-primary-700 dark:to-primary-800 rounded-2xl shadow-xl p-8 text-white">
        <h1 className="text-4xl font-bold mb-2">Welcome to Masar</h1>
        <p className="text-xl opacity-90">Your journey to the perfect career starts here</p>
      </div>
      
      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {stats.map((stat, index) => {
          const Icon = stat.icon;
          return (
            <div key={index} className="card">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">{stat.label}</p>
                  <p className="text-3xl font-bold text-gray-800 dark:text-gray-100">{stat.value}</p>
                  <p className="text-sm text-green-600 dark:text-green-400 mt-1">{stat.change} this month</p>
                </div>
                <div className={`w-12 h-12 rounded-lg ${getColorClasses(stat.color)} flex items-center justify-center`}>
                  <Icon className="w-6 h-6" />
                </div>
              </div>
            </div>
          );
        })}
      </div>
      
      {/* Features */}
      <div>
        <h2 className="text-2xl font-bold text-gray-800 dark:text-gray-100 mb-6">Explore Features</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {features.map((feature, index) => {
            const Icon = feature.icon;
            return (
              <Link
                key={index}
                to={feature.link}
                className="card hover:shadow-lg transition-shadow cursor-pointer group"
              >
                <div className={`w-16 h-16 rounded-xl ${getColorClasses(feature.color)} flex items-center justify-center mb-4 group-hover:scale-110 transition-transform`}>
                  <Icon className="w-8 h-8" />
                </div>
                <h3 className="text-xl font-bold text-gray-800 dark:text-gray-100 mb-2 group-hover:text-primary-600 dark:group-hover:text-primary-400 transition-colors">
                  {feature.title}
                </h3>
                <p className="text-gray-600 dark:text-gray-400">{feature.description}</p>
                <div className="mt-4 text-primary-600 dark:text-primary-400 font-semibold flex items-center group-hover:gap-2 transition-all">
                  Get Started
                  <span className="ml-2 group-hover:ml-0 transition-all">→</span>
                </div>
              </Link>
            );
          })}
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
