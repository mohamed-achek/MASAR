import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';

const Landing = () => {
  const navigate = useNavigate();
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    setIsVisible(true);
  }, []);

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.15,
        delayChildren: 0.3,
      },
    },
  };

  const titleVariants = {
    hidden: { 
      opacity: 0, 
      y: 50,
      scale: 0.8,
    },
    visible: {
      opacity: 1,
      y: 0,
      scale: 1,
      transition: {
        type: 'spring',
        stiffness: 100,
        damping: 15,
        duration: 0.8,
      },
    },
  };

  const letterVariants = {
    hidden: { 
      opacity: 0,
      y: 50,
    },
    visible: (i) => ({
      opacity: 1,
      y: 0,
      transition: {
        delay: i * 0.1,
        type: 'spring',
        stiffness: 100,
        damping: 12,
      },
    }),
  };

  const buttonVariants = {
    hidden: { opacity: 0, y: 30 },
    visible: {
      opacity: 1,
      y: 0,
      transition: {
        type: 'spring',
        stiffness: 120,
        damping: 12,
      },
    },
    hover: {
      scale: 1.08,
      y: -5,
      transition: {
        type: 'spring',
        stiffness: 400,
        damping: 10,
      },
    },
    tap: { scale: 0.95 },
  };

  const glowVariants = {
    animate: {
      boxShadow: [
        '0 0 20px rgba(30, 64, 175, 0.3)',
        '0 0 40px rgba(30, 64, 175, 0.5)',
        '0 0 20px rgba(30, 64, 175, 0.3)',
      ],
      transition: {
        duration: 2,
        repeat: Infinity,
        ease: 'easeInOut',
      },
    },
  };

  const floatingVariants = {
    animate: {
      y: [0, -15, 0],
      rotate: [0, 2, -2, 0],
      transition: {
        duration: 4,
        repeat: Infinity,
        ease: 'easeInOut',
      },
    },
  };

  const particleVariants = {
    animate: (i) => ({
      y: [0, -100, -200],
      x: [(i % 2 === 0 ? 1 : -1) * Math.random() * 50, (i % 2 === 0 ? 1 : -1) * Math.random() * 100],
      opacity: [0, 1, 0],
      scale: [0, 1.5, 0],
      transition: {
        duration: 3 + Math.random() * 2,
        repeat: Infinity,
        delay: i * 0.5,
        ease: 'easeOut',
      },
    }),
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-950 via-blue-900 to-slate-900 flex items-center justify-center overflow-hidden relative">
      {/* Animated background elements */}
      <div className="absolute inset-0 overflow-hidden">
        <motion.div
          className="absolute top-20 left-20 w-72 h-72 bg-blue-800 rounded-full mix-blend-multiply filter blur-xl opacity-20"
          animate={{
            x: [0, 100, 0],
            y: [0, 50, 0],
          }}
          transition={{
            duration: 8,
            repeat: Infinity,
            ease: 'easeInOut',
          }}
        />
        <motion.div
          className="absolute top-40 right-20 w-72 h-72 bg-blue-700 rounded-full mix-blend-multiply filter blur-xl opacity-20"
          animate={{
            x: [0, -100, 0],
            y: [0, 100, 0],
          }}
          transition={{
            duration: 10,
            repeat: Infinity,
            ease: 'easeInOut',
          }}
        />
        <motion.div
          className="absolute bottom-20 left-1/2 w-72 h-72 bg-slate-800 rounded-full mix-blend-multiply filter blur-xl opacity-20"
          animate={{
            x: [0, -50, 0],
            y: [0, -50, 0],
          }}
          transition={{
            duration: 12,
            repeat: Infinity,
            ease: 'easeInOut',
          }}
        />
      </div>

      {/* Main content */}
      <div className="relative z-10 text-center px-4">
        {/* Logo/Title section */}
        <motion.div
          variants={floatingVariants}
          animate="animate"
          className="mb-12"
        >
          {/* Combined English and Arabic Title */}
          <motion.div
            initial="hidden"
            animate={isVisible ? "visible" : "hidden"}
            variants={containerVariants}
            className="mb-8"
          >
            {/* English Title */}
            {/* <motion.div
              variants={titleVariants}
              className="mb-3"
            >
              <h1 className="text-7xl md:text-9xl font-black text-white tracking-tight">
                {['M', 'A', 'S', 'A', 'R'].map((letter, i) => (
                  <motion.span
                    key={i}
                    custom={i}
                    variants={letterVariants}
                    initial="hidden"
                    animate={isVisible ? "visible" : "hidden"}
                    className="inline-block"
                    whileHover={{ 
                      scale: 1.2, 
                      color: '#a78bfa',
                      transition: { type: 'spring', stiffness: 300 }
                    }}
                  >
                    {letter}
                  </motion.span>
                ))}
              </h1>
            </motion.div> */}

            {/* Arabic Title - closer to English */}
            <motion.div
              variants={titleVariants}
              className="mb-6"
            >
              <h2 className="text-8xl md:text-[12rem] font-black text-white tracking-tight">
                {['مسار'].map((letter, i) => (
                  <motion.span
                    key={i}          
                    custom={i + 5}
                    variants={letterVariants}
                    initial="hidden"
                    animate={isVisible ? "visible" : "hidden"}
                    className="inline-block"
                    whileHover={{ 
                      scale: 1.2,
                      transition: { type: 'spring', stiffness: 300 }
                    }}
                  >
                    {letter}
                  </motion.span>
                ))}
              </h2>
            </motion.div>

            {/* Decorative line */}
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: isVisible ? '200px' : 0 }}
              transition={{ delay: 1, duration: 0.8 }}
              className="h-1 bg-gradient-to-r from-transparent via-blue-500 to-transparent mx-auto mb-6"
            />

            {/* Tagline */}
            {/* <motion.p
              initial={{ opacity: 0, y: 20 }}
              animate={{ 
                opacity: isVisible ? 1 : 0,
                y: isVisible ? 0 : 20,
              }}
              transition={{ delay: 1.2, duration: 0.8 }}
              className="text-xl md:text-2xl text-purple-200 font-light tracking-wide"
            >
              Your Path to Academic Excellence
            </motion.p> */}
            
            {/* Arabic Tagline */}
            <motion.p
              initial={{ opacity: 0, y: 20 }}
              animate={{ 
                opacity: isVisible ? 1 : 0,
                y: isVisible ? 0 : 20,
              }}
              transition={{ delay: 1.4, duration: 0.8 }}
              className="text-2xl md:text-4xl text-white font-bold mt-2"
              dir="rtl"
            >
              طريقك نحو التميز الأكاديمي
            </motion.p>
          </motion.div>
        </motion.div>

        {/* Buttons */}
        <motion.div
          variants={containerVariants}
          initial="hidden"
          animate={isVisible ? "visible" : "hidden"}
          className="flex flex-col sm:flex-row gap-6 justify-center items-center mt-16"
        >
          <motion.button
            variants={buttonVariants}
            whileHover="hover"
            whileTap="tap"
            onClick={() => navigate('/login')}
            className="group relative px-14 py-5 bg-gradient-to-r from-blue-700 to-blue-800 text-white rounded-full text-lg font-bold shadow-2xl overflow-hidden border-2 border-blue-600/50"
          >
            <motion.div
              className="absolute inset-0 bg-gradient-to-r from-blue-600 to-blue-700"
              initial={{ opacity: 0 }}
              whileHover={{ opacity: 1 }}
              transition={{ duration: 0.3 }}
            />
            <span className="relative z-10 flex items-center gap-3">
              <svg
                className="w-5 h-5"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M11 16l-4-4m0 0l4-4m-4 4h14m-5 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h7a3 3 0 013 3v1"
                />
              </svg>
              Sign In
              <svg
                className="w-5 h-5 transition-transform group-hover:translate-x-2"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M13 7l5 5m0 0l-5 5m5-5H6"
                />
              </svg>
            </span>
          </motion.button>

          <motion.button
            variants={buttonVariants}
            whileHover="hover"
            whileTap="tap"
            onClick={() => navigate('/signup')}
            className="group relative px-14 py-5 bg-transparent text-white border-2 border-white/80 rounded-full text-lg font-bold shadow-2xl overflow-hidden backdrop-blur-sm"
          >
            <motion.div
              className="absolute inset-0 bg-white"
              initial={{ opacity: 0 }}
              whileHover={{ opacity: 0.15 }}
              transition={{ duration: 0.3 }}
            />
            <span className="relative z-10 flex items-center gap-3">
              <svg
                className="w-5 h-5"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M18 9v3m0 0v3m0-3h3m-3 0h-3m-2-5a4 4 0 11-8 0 4 4 0 018 0zM3 20a6 6 0 0112 0v1H3v-1z"
                />
              </svg>
              Sign Up
              <svg
                className="w-5 h-5 transition-transform group-hover:translate-x-2"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M13 7l5 5m0 0l-5 5m5-5H6"
                />
              </svg>
            </span>
          </motion.button>
        </motion.div>

        {/* Decorative elements */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: isVisible ? 1 : 0 }}
          transition={{ delay: 1.8, duration: 1 }}
          className="mt-16 flex justify-center gap-3"
        >
          {[0, 1, 2].map((i) => (
            <motion.div
              key={i}
              animate={{
                y: [0, -15, 0],
                scale: [1, 1.2, 1],
              }}
              transition={{
                duration: 2,
                repeat: Infinity,
                delay: i * 0.3,
                ease: 'easeInOut',
              }}
              className="w-3 h-3 bg-gradient-to-r from-blue-500 to-blue-600 rounded-full shadow-lg shadow-blue-500/50"
            />
          ))}
        </motion.div>

        {/* Floating particles */}
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          {[...Array(6)].map((_, i) => (
            <motion.div
              key={i}
              custom={i}
              variants={particleVariants}
              animate="animate"
              className="absolute bottom-0 left-1/2"
              style={{
                left: `${20 + i * 12}%`,
              }}
            >
              <div className="w-2 h-2 bg-blue-500/60 rounded-full blur-sm" />
            </motion.div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default Landing;
