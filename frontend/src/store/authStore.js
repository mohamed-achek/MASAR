import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import axios from 'axios';
import { jwtDecode } from 'jwt-decode';

const API_URL = '/api';

const useAuthStore = create(
  persist(
    (set, get) => ({
      user: null,
      token: null,
      isAuthenticated: false,
      loading: false,
      error: null,

      // Login
      login: async (email, password) => {
        set({ loading: true, error: null });
        try {
          const response = await axios.post(`${API_URL}/auth/login`, {
            email,
            password,
          });
          
          const { access_token, user } = response.data;
          
          // Set axios default header
          axios.defaults.headers.common['Authorization'] = `Bearer ${access_token}`;
          
          set({
            user,
            token: access_token,
            isAuthenticated: true,
            loading: false,
            error: null,
          });
          
          return true;
        } catch (error) {
          set({
            loading: false,
            error: error.response?.data?.detail || 'Login failed',
          });
          return false;
        }
      },

      // Signup
      signup: async (email, password, name) => {
        set({ loading: true, error: null });
        try {
          const response = await axios.post(`${API_URL}/auth/signup`, {
            email,
            password,
            name,
          });
          
          const { access_token, user } = response.data;
          
          axios.defaults.headers.common['Authorization'] = `Bearer ${access_token}`;
          
          set({
            user,
            token: access_token,
            isAuthenticated: true,
            loading: false,
            error: null,
          });
          
          return true;
        } catch (error) {
          set({
            loading: false,
            error: error.response?.data?.detail || 'Signup failed',
          });
          return false;
        }
      },

      // Logout
      logout: () => {
        delete axios.defaults.headers.common['Authorization'];
        set({
          user: null,
          token: null,
          isAuthenticated: false,
          error: null,
        });
      },

      // Check if token is valid
      checkAuth: () => {
        const { token } = get();
        if (token) {
          try {
            const decoded = jwtDecode(token);
            const currentTime = Date.now() / 1000;
            
            if (decoded.exp < currentTime) {
              // Token expired
              get().logout();
              return false;
            }
            
            // Set axios header
            axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
            return true;
          } catch (error) {
            get().logout();
            return false;
          }
        }
        return false;
      },

      // Clear error
      clearError: () => set({ error: null }),
    }),
    {
      name: 'auth-storage',
      partialize: (state) => ({
        user: state.user,
        token: state.token,
        isAuthenticated: state.isAuthenticated,
      }),
    }
  )
);

export default useAuthStore;
