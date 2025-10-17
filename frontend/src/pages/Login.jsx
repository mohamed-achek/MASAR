import React, { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { LogIn, Mail, Lock, Moon, Sun } from 'lucide-react';
import useAuthStore from '../store/authStore';
import useThemeStore from '../store/themeStore';

const Login = () => {
  const navigate = useNavigate();
  const { login, loading, error, isAuthenticated, clearError } = useAuthStore();
  const { theme, toggleTheme } = useThemeStore();
  
  const [formData, setFormData] = useState({
    email: '',
    password: '',
  });
  
  const [language, setLanguage] = useState('en');
  
  const translations = {
    en: {
      title: 'Welcome Back to Masar',
      subtitle: 'Connecting Education to Career Opportunities',
      email: 'Email Address',
      password: 'Password',
      login: 'Sign In',
      noAccount: "Don't have an account?",
      signup: 'Sign Up',
      loading: 'Signing in...',
    },
    fr: {
      title: 'Bienvenue sur Masar',
      subtitle: 'Connecter l\'Éducation aux Opportunités de Carrière',
      email: 'Adresse Email',
      password: 'Mot de Passe',
      login: 'Se Connecter',
      noAccount: "Vous n'avez pas de compte?",
      signup: 'S\'inscrire',
      loading: 'Connexion...',
    },
    ar: {
      title: 'مرحباً بك في مسار',
      subtitle: 'ربط التعليم بفرص العمل',
      email: 'البريد الإلكتروني',
      password: 'كلمة المرور',
      login: 'تسجيل الدخول',
      noAccount: 'ليس لديك حساب؟',
      signup: 'إنشاء حساب',
      loading: 'جاري تسجيل الدخول...',
    },
  };
  
  const t = translations[language];
  
  useEffect(() => {
    if (isAuthenticated) {
      navigate('/');
    }
  }, [isAuthenticated, navigate]);
  
  useEffect(() => {
    clearError();
  }, [clearError]);
  
  const handleSubmit = async (e) => {
    e.preventDefault();
    const success = await login(formData.email, formData.password);
    if (success) {
      navigate('/');
    }
  };
  
  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };
  
  return (
    <div className={`min-h-screen bg-gradient-to-br from-primary-50 to-primary-100 dark:from-gray-900 dark:to-gray-800 flex items-center justify-center p-4 ${language === 'ar' ? 'rtl' : 'ltr'}`}>
      <div className="max-w-md w-full">
        {/* Language Selector and Theme Toggle */}
        <div className="flex justify-between items-center mb-4">
          <button
            onClick={toggleTheme}
            className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-2 hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
            aria-label="Toggle theme"
          >
            {theme === 'light' ? (
              <Moon className="w-5 h-5 text-gray-600 dark:text-gray-300" />
            ) : (
              <Sun className="w-5 h-5 text-gray-600 dark:text-gray-300" />
            )}
          </button>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-2 flex gap-2">
            <button
              onClick={() => setLanguage('en')}
              className={`px-3 py-1 rounded ${language === 'en' ? 'bg-primary-600 text-white' : 'text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700'}`}
            >
              EN
            </button>
            <button
              onClick={() => setLanguage('fr')}
              className={`px-3 py-1 rounded ${language === 'fr' ? 'bg-primary-600 text-white' : 'text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700'}`}
            >
              FR
            </button>
            <button
              onClick={() => setLanguage('ar')}
              className={`px-3 py-1 rounded ${language === 'ar' ? 'bg-primary-600 text-white' : 'text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700'}`}
            >
              AR
            </button>
          </div>
        </div>
        
        {/* Login Card */}
        <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-2xl p-8">
          {/* Header */}
          <div className="text-center mb-8">
            <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-r from-primary-600 to-primary-700 dark:from-primary-700 dark:to-primary-800 rounded-full mb-4">
              <LogIn className="w-8 h-8 text-white" />
            </div>
            <h1 className="text-3xl font-bold text-gray-800 dark:text-gray-100 mb-2">{t.title}</h1>
            <p className="text-gray-600 dark:text-gray-400">{t.subtitle}</p>
          </div>
          
          {/* Error Message */}
          {error && (
            <div className="mb-6 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg text-red-700 dark:text-red-400">
              {error}
            </div>
          )}
          
          {/* Form */}
          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Email */}
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                {t.email}
              </label>
              <div className="relative">
                <Mail className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400 dark:text-gray-500" />
                <input
                  type="email"
                  name="email"
                  value={formData.email}
                  onChange={handleChange}
                  className="input-field pl-10"
                  placeholder="you@example.com"
                  required
                  disabled={loading}
                />
              </div>
            </div>
            
            {/* Password */}
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                {t.password}
              </label>
              <div className="relative">
                <Lock className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400 dark:text-gray-500" />
                <input
                  type="password"
                  name="password"
                  value={formData.password}
                  onChange={handleChange}
                  className="input-field pl-10"
                  placeholder="••••••••"
                  required
                  disabled={loading}
                />
              </div>
            </div>
            
            {/* Submit Button */}
            <button
              type="submit"
              disabled={loading}
              className="w-full btn-primary flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? (
                <>
                  <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                  {t.loading}
                </>
              ) : (
                <>
                  <LogIn className="w-5 h-5" />
                  {t.login}
                </>
              )}
            </button>
          </form>
          
          {/* Signup Link */}
          <div className="mt-6 text-center text-sm text-gray-600 dark:text-gray-400">
            {t.noAccount}{' '}
            <Link to="/signup" className="text-primary-600 dark:text-primary-400 hover:text-primary-700 dark:hover:text-primary-300 font-semibold">
              {t.signup}
            </Link>
          </div>
        </div>
        
        {/* Footer */}
        <div className="text-center mt-8 text-sm text-gray-600 dark:text-gray-400">
          <p>© 2025 Masar — EduMatch Tunisia</p>
        </div>
      </div>
    </div>
  );
};

export default Login;
