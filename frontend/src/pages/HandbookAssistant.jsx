import React, { useState, useRef, useEffect } from 'react';
import { Send, BookOpen, Trash2, Loader, AlertCircle } from 'lucide-react';
import axios from 'axios';

const HandbookAssistant = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [ragAvailable, setRagAvailable] = useState(true);
  const messagesEndRef = useRef(null);
  
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };
  
  useEffect(() => {
    scrollToBottom();
  }, [messages]);
  
  useEffect(() => {
    // Check if RAG is available
    axios.get('/api/rag/status')
      .then(response => {
        setRagAvailable(response.data.available);
      })
      .catch(() => {
        setRagAvailable(false);
      });
  }, []);
  
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || loading) return;
    
    const userMessage = input.trim();
    setInput('');
    
    // Add user message
    const newMessages = [...messages, { role: 'user', content: userMessage }];
    setMessages(newMessages);
    setLoading(true);
    
    try {
      const response = await axios.post('/api/rag/query', {
        question: userMessage,
        language: 'auto',
        top_k: 3,
      });
      
      // Add assistant response
      setMessages([
        ...newMessages,
        {
          role: 'assistant',
          content: response.data.answer,
          sources: response.data.sources || [],
        },
      ]);
    } catch (error) {
      setMessages([
        ...newMessages,
        {
          role: 'assistant',
          content: 'Sorry, I encountered an error. Please try again.',
          error: true,
        },
      ]);
    } finally {
      setLoading(false);
    }
  };
  
  const clearChat = () => {
    setMessages([]);
  };
  
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="card bg-gradient-to-r from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-800/20">
        <div className="flex items-center gap-4">
          <div className="w-12 h-12 bg-blue-600 dark:bg-blue-700 rounded-lg flex items-center justify-center">
            <BookOpen className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-3xl font-bold text-gray-800 dark:text-gray-100">Handbook Assistant</h1>
            <p className="text-gray-600 dark:text-gray-400">Ask me anything about Tunisian universities policies, courses, or procedures</p>
          </div>
        </div>
      </div>
      
      {/* RAG Status */}
      {!ragAvailable && (
        <div className="card bg-yellow-50 dark:bg-yellow-900/20 border-yellow-200 dark:border-yellow-800">
          <div className="flex items-start gap-3">
            <AlertCircle className="w-6 h-6 text-yellow-600 dark:text-yellow-400 flex-shrink-0 mt-1" />
            <div>
              <h3 className="font-semibold text-yellow-800 dark:text-yellow-200 mb-1">RAG System Not Available</h3>
              <p className="text-sm text-yellow-700 dark:text-yellow-300">
                The handbook assistant is currently unavailable. Please ensure the backend is running and the FAISS index is loaded.
              </p>
            </div>
          </div>
        </div>
      )}
      
      {/* Chat Interface */}
      <div className="card flex flex-col h-[600px]">
        {/* Messages */}
        <div className="flex-1 overflow-y-auto mb-4 space-y-4">
          {messages.length === 0 ? (
            <div className="text-center py-12 text-gray-500 dark:text-gray-400">
              <BookOpen className="w-16 h-16 mx-auto mb-4 text-gray-400 dark:text-gray-500" />
              <p className="text-lg font-medium mb-2">No messages yet</p>
              <p className="text-sm">Ask a question about the TBS Handbook to get started</p>
              
              {/* Example Questions */}
              <div className="mt-8 text-left max-w-md mx-auto">
                <p className="font-medium text-gray-700 dark:text-gray-300 mb-3">Example questions:</p>
                <div className="space-y-2">
                  {[
                    "What are the admission requirements?",
                    "Quels sont les règlements d'examen?",
                    "ما هي إجراءات التسجيل؟",
                  ].map((example, index) => (
                    <button
                      key={index}
                      onClick={() => setInput(example)}
                      className="w-full text-left px-4 py-2 bg-gray-50 dark:bg-gray-700 hover:bg-gray-100 dark:hover:bg-gray-600 rounded-lg text-sm text-gray-700 dark:text-gray-300 transition-colors"
                    >
                      {example}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <>
              {messages.map((message, index) => (
                <div
                  key={index}
                  className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`max-w-[80%] rounded-lg p-4 ${
                      message.role === 'user'
                        ? 'bg-primary-600 text-white'
                        : message.error
                        ? 'bg-red-50 dark:bg-red-900/20 text-red-800 dark:text-red-400 border border-red-200 dark:border-red-800'
                        : 'bg-gray-100 dark:bg-gray-700 text-gray-800 dark:text-gray-100'
                    }`}
                  >
                    <div className="whitespace-pre-wrap">{message.content}</div>
                    
                    {/* Sources */}
                    {message.sources && message.sources.length > 0 && (
                      <details className="mt-4 text-sm">
                        <summary className="cursor-pointer font-semibold text-gray-700 dark:text-gray-300 hover:text-gray-900 dark:hover:text-gray-100">
                          📚 View {message.sources.length} sources
                        </summary>
                        <div className="mt-3 space-y-3">
                          {message.sources.map((source, idx) => (
                            <div key={idx} className="bg-white dark:bg-gray-600 rounded p-3 border border-gray-200 dark:border-gray-500">
                              <div className="flex items-center justify-between mb-2">
                                <span className="font-semibold text-gray-700 dark:text-gray-200">Source {source.rank}</span>
                                <span className="text-xs text-gray-500 dark:text-gray-400">Score: {(source.score * 100).toFixed(1)}%</span>
                              </div>
                              <p className="text-xs text-gray-600 dark:text-gray-300 line-clamp-3">{source.text}</p>
                            </div>
                          ))}
                        </div>
                      </details>
                    )}
                  </div>
                </div>
              ))}
              
              {loading && (
                <div className="flex justify-start">
                  <div className="bg-gray-100 dark:bg-gray-700 rounded-lg p-4">
                    <Loader className="w-6 h-6 text-gray-600 dark:text-gray-400 animate-spin" />
                  </div>
                </div>
              )}
              
              <div ref={messagesEndRef} />
            </>
          )}
        </div>
        
        {/* Input Form */}
        <div className="border-t border-gray-200 dark:border-gray-700 pt-4">
          <form onSubmit={handleSubmit} className="flex gap-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask a question about TBS Handbook..."
              className="input-field flex-1"
              disabled={loading || !ragAvailable}
            />
            <button
              type="submit"
              disabled={loading || !input.trim() || !ragAvailable}
              className="btn-primary flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Send className="w-5 h-5" />
              <span className="hidden sm:inline">Send</span>
            </button>
            {messages.length > 0 && (
              <button
                type="button"
                onClick={clearChat}
                className="btn-secondary flex items-center gap-2"
              >
                <Trash2 className="w-5 h-5" />
                <span className="hidden sm:inline">Clear</span>
              </button>
            )}
          </form>
        </div>
      </div>
      
      
    </div>
  );
};

export default HandbookAssistant;
