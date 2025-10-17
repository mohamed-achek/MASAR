import React, { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { UserPlus, Mail, Lock, User, Moon, Sun } from 'lucide-react';
import useAuthStore from '../store/authStore';
import useThemeStore from '../store/themeStore';

const Signup = () => {
  const navigate = useNavigate();
  const { signup, loading, error, isAuthenticated, clearError } = useAuthStore();
  const { theme, toggleTheme } = useThemeStore();
  
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    password: '',
    confirmPassword: '',
  });
  
  const [language, setLanguage] = useState('en');
  const [validationError, setValidationError] = useState('');
  
  const translations = {
    en: {
      title: 'Join Masar',
      subtitle: 'Start your career journey today',
      name: 'Full Name',
      email: 'Email Address',
      password: 'Password',
      confirmPassword: 'Confirm Password',
      signup: 'Create Account',
      hasAccount: 'Already have an account?',
      login: 'Sign In',
      loading: 'Creating account...',
      passwordMismatch: 'Passwords do not match',
      passwordShort: 'Password must be at least 8 characters',
    },
    fr: {
      title: 'Rejoindre Masar',
      subtitle: 'Commencez votre parcours professionnel aujourd\'hui',
      name: 'Nom Complet',
      email: 'Adresse Email',
      password: 'Mot de Passe',
      confirmPassword: 'Confirmer le Mot de Passe',
      signup: 'Créer un Compte',
      hasAccount: 'Vous avez déjà un compte?',
      login: 'Se Connecter',
      loading: 'Création du compte...',
      passwordMismatch: 'Les mots de passe ne correspondent pas',
      passwordShort: 'Le mot de passe doit contenir au moins 8 caractères',
    },
    ar: {
      title: 'انضم إلى مسار',
      subtitle: 'ابدأ رحلتك المهنية اليوم',
      name: 'الاسم الكامل',
      email: 'البريد الإلكتروني',
      password: 'كلمة المرور',
      confirmPassword: 'تأكيد كلمة المرور',
      signup: 'إنشاء حساب',
      hasAccount: 'هل لديك حساب بالفعل؟',
      login: 'تسجيل الدخول',
      loading: 'جاري إنشاء الحساب...',
      passwordMismatch: 'كلمات المرور غير متطابقة',
      passwordShort: 'يجب أن تحتوي كلمة المرور على 8 أحرف على الأقل',
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
  
  const validate = () => {
    if (formData.password.length < 8) {
      setValidationError(t.passwordShort);
      return false;
    }
    
    if (formData.password !== formData.confirmPassword) {
      setValidationError(t.passwordMismatch);
      return false;
    }
    
    setValidationError('');
    return true;
  };
  
  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!validate()) return;
    
    const success = await signup(formData.email, formData.password, formData.name);
    if (success) {
      navigate('/');
    }
  };
  
  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
    setValidationError('');
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
        
        {/* Signup Card */}
        <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-2xl p-8">
          {/* Header */}
          <div className="text-center mb-8">
            <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-r from-primary-600 to-primary-700 dark:from-primary-700 dark:to-primary-800 rounded-full mb-4">
              <UserPlus className="w-8 h-8 text-white" />
            </div>
            <h1 className="text-3xl font-bold text-gray-800 dark:text-gray-100 mb-2">{t.title}</h1>
            <p className="text-gray-600 dark:text-gray-400">{t.subtitle}</p>
          </div>
          
          {/* Error Messages */}
          {(error || validationError) && (
            <div className="mb-6 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg text-red-700 dark:text-red-400">
              {error || validationError}
            </div>
          )}
          
          {/* Form */}
          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Name */}
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                {t.name}
              </label>
              <div className="relative">
                <User className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400 dark:text-gray-500" />
                <input
                  type="text"
                  name="name"
                  value={formData.name}
                  onChange={handleChange}
                  className="input-field pl-10"
                  placeholder="John Doe"
                  required
                  disabled={loading}
                />
              </div>
            </div>
            
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
                  minLength={8}
                />
              </div>
            </div>
            
            {/* Confirm Password */}
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                {t.confirmPassword}
              </label>
              <div className="relative">
                <Lock className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400 dark:text-gray-500" />
                <input
                  type="password"
                  name="confirmPassword"
                  value={formData.confirmPassword}
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
                  <UserPlus className="w-5 h-5" />
                  {t.signup}
                </>
              )}
            </button>
          </form>
          
          {/* Login Link */}
          <div className="mt-6 text-center text-sm text-gray-600 dark:text-gray-400">
            {t.hasAccount}{' '}
            <Link to="/login" className="text-primary-600 dark:text-primary-400 hover:text-primary-700 dark:hover:text-primary-300 font-semibold">
              {t.login}
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

export default Signup;
