import React, { useState, useRef, useEffect } from 'react';
import { Send, BookOpen, Trash2, Loader, AlertCircle } from 'lucide-react';
import axios from 'axios';

const HandbookAssistant = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [ragAvailable, setRagAvailable] = useState(true);
  const [streamingMessage, setStreamingMessage] = useState('');
  const [loadingStage, setLoadingStage] = useState('');
  const messagesEndRef = useRef(null);
  const abortControllerRef = useRef(null);
  
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };
  
  useEffect(() => {
    scrollToBottom();
  }, [messages, streamingMessage]);
  
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
    setStreamingMessage('');
    
    // Add placeholder for assistant message with loading stage
    const placeholderMessage = { 
      role: 'assistant', 
      content: '', 
      sources: [], 
      streaming: true,
      loadingStage: 'thinking'
    };
    setMessages([...newMessages, placeholderMessage]);
    setLoadingStage('thinking');
    
    try {
      // Simulate stage transitions
      setTimeout(() => setLoadingStage('searching'), 500);
      setTimeout(() => setLoadingStage('analyzing'), 1500);
      setTimeout(() => setLoadingStage('generating'), 2500);
      
      // Create abort controller for cancellation
      abortControllerRef.current = new AbortController();
      
      // Prepare conversation history (last 5 messages for context)
      const conversationHistory = messages
        .slice(-10) // Get last 10 messages (5 exchanges)
        .map(msg => ({
          role: msg.role,
          content: msg.content
        }));
      
      const response = await fetch('/api/rag/query-stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': axios.defaults.headers.common['Authorization'],
        },
        body: JSON.stringify({
          question: userMessage,
          language: 'auto',
          top_k: 3,
          conversation_history: conversationHistory,
        }),
        signal: abortControllerRef.current.signal,
      });
      
      if (!response.ok) {
        throw new Error('Failed to fetch streaming response');
      }
      
      setLoadingStage('');
      
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let accumulatedText = '';
      let sources = [];
      
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');
        
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              
              if (data.type === 'token') {
                accumulatedText += data.content;
                setStreamingMessage(accumulatedText);
                
                // Update the last message in real-time
                setMessages(prev => {
                  const updated = [...prev];
                  updated[updated.length - 1] = {
                    role: 'assistant',
                    content: accumulatedText,
                    sources: [],
                    streaming: true,
                  };
                  return updated;
                });
              } else if (data.type === 'sources') {
                sources = data.content;
              } else if (data.type === 'done') {
                // Finalize the message
                setMessages(prev => {
                  const updated = [...prev];
                  updated[updated.length - 1] = {
                    role: 'assistant',
                    content: accumulatedText,
                    sources: sources,
                    streaming: false,
                  };
                  return updated;
                });
                setStreamingMessage('');
              } else if (data.type === 'error') {
                throw new Error(data.content);
              }
            } catch (parseError) {
              console.error('Error parsing SSE data:', parseError);
            }
          }
        }
      }
    } catch (error) {
      if (error.name === 'AbortError') {
        console.log('Streaming aborted');
      } else {
        console.error('Streaming error:', error);
        setMessages(prev => {
          const updated = [...prev];
          updated[updated.length - 1] = {
            role: 'assistant',
            content: 'Sorry, I encountered an error. Please try again.',
            error: true,
            streaming: false,
          };
          return updated;
        });
      }
    } finally {
      setLoading(false);
      abortControllerRef.current = null;
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
                    {/* Loading Stage Indicator */}
                    {message.streaming && !message.content && (
                      <div className="flex items-center gap-3">
                        <div className="relative">
                          <Loader className="w-5 h-5 text-blue-600 dark:text-blue-400 animate-spin" />
                          <div className="absolute -inset-1 bg-blue-400/20 dark:bg-blue-400/10 rounded-full animate-ping"></div>
                        </div>
                        <div className="space-y-1">
                          {loadingStage === 'thinking' && (
                            <div className="flex items-center gap-2">
                              <span className="text-sm font-medium text-blue-600 dark:text-blue-400">
                                🤔 Understanding your question...
                              </span>
                              <div className="flex gap-1">
                                <span className="w-1.5 h-1.5 bg-blue-600 dark:bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></span>
                                <span className="w-1.5 h-1.5 bg-blue-600 dark:bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></span>
                                <span className="w-1.5 h-1.5 bg-blue-600 dark:bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></span>
                              </div>
                            </div>
                          )}
                          {loadingStage === 'searching' && (
                            <div className="flex items-center gap-2">
                              <span className="text-sm font-medium text-purple-600 dark:text-purple-400">
                                🔍 Searching through handbooks...
                              </span>
                              <div className="flex gap-1">
                                <span className="w-1.5 h-1.5 bg-purple-600 dark:bg-purple-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></span>
                                <span className="w-1.5 h-1.5 bg-purple-600 dark:bg-purple-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></span>
                                <span className="w-1.5 h-1.5 bg-purple-600 dark:bg-purple-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></span>
                              </div>
                            </div>
                          )}
                          {loadingStage === 'analyzing' && (
                            <div className="flex items-center gap-2">
                              <span className="text-sm font-medium text-amber-600 dark:text-amber-400">
                                📊 Analyzing relevant information...
                              </span>
                              <div className="flex gap-1">
                                <span className="w-1.5 h-1.5 bg-amber-600 dark:bg-amber-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></span>
                                <span className="w-1.5 h-1.5 bg-amber-600 dark:bg-amber-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></span>
                                <span className="w-1.5 h-1.5 bg-amber-600 dark:bg-amber-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></span>
                              </div>
                            </div>
                          )}
                          {loadingStage === 'generating' && (
                            <div className="flex items-center gap-2">
                              <span className="text-sm font-medium text-green-600 dark:text-green-400">
                                ✨ Generating optimal answer...
                              </span>
                              <div className="flex gap-1">
                                <span className="w-1.5 h-1.5 bg-green-600 dark:bg-green-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></span>
                                <span className="w-1.5 h-1.5 bg-green-600 dark:bg-green-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></span>
                                <span className="w-1.5 h-1.5 bg-green-600 dark:bg-green-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></span>
                              </div>
                            </div>
                          )}
                        </div>
                      </div>
                    )}
                    
                    {/* Streaming content */}
                    {message.content && (
                      <div className="whitespace-pre-wrap">
                        {message.content}
                        {message.streaming && (
                          <span className="inline-block w-2 h-5 ml-1 bg-gray-600 dark:bg-gray-300 animate-pulse"></span>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              ))}
              
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
