import axios from "axios";

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

// Create axios instance with default config
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

// Add auth token to requests if available
api.interceptors.request.use((config) => {
  const token = localStorage.getItem("token");
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Auth API
export const authAPI = {
  signup: async (email: string, password: string, fullName: string) => {
    const response = await api.post("/auth/signup", {
      email,
      password,
      full_name: fullName,
    });
    return response.data;
  },

  login: async (email: string, password: string) => {
    const response = await api.post("/auth/login", { email, password });
    return response.data;
  },

  getCurrentUser: async () => {
    const response = await api.get("/auth/me");
    return response.data;
  },
};

// Chat API
export const chatAPI = {
  sendMessage: async (message: string, userId: string) => {
    const response = await api.post("/chat", {
      message,
      user_id: userId,
    });
    return response.data;
  },

  getSessions: async () => {
    const response = await api.get("/sessions");
    return response.data;
  },

  getSessionMessages: async (
    sessionId: string,
    limit: number = 200,
    offset: number = 0,
  ) => {
    const response = await api.get(`/sessions/${sessionId}/messages`, {
      params: { limit, offset },
    });
    return response.data;
  },

  clearSessions: async () => {
    const response = await api.delete("/sessions/clear");
    return response.data;
  },
};

// Memory/Mindmap API
export const memoryAPI = {
  getMindmap: async () => {
    const response = await api.get("/memory/mindmap");
    return response.data;
  },

  clearGraph: async () => {
    const response = await api.delete("/memory/clear");
    return response.data;
  },

  getVectorEntries: async (limit: number = 100) => {
    const response = await api.get("/memory/vectors", {
      params: { limit },
    });
    return response.data;
  },

  clearGraph: async () => {
    const response = await api.delete('/memory/clear');
    return response.data;
  },
};

// Document API
export const documentAPI = {
  uploadDocument: async (file: File) => {
    const formData = new FormData();
    formData.append("file", file);
    const response = await api.post("/documents/upload", formData, {
      headers: { "Content-Type": "multipart/form-data" },
    });
    return response.data;
  },

  ingestDocument: async (
    documentText: string,
    documentName: string,
    documentFormat: string,
    userId: string,
    metadata?: Record<string, any>,
  ) => {
    const response = await api.post("/documents/ingest", {
      user_id: userId,
      document_text: documentText,
      document_name: documentName,
      document_format: documentFormat,
      metadata: metadata || {},
    });
    return response.data;
  },

  uploadAndIngest: async (file: File) => {
    const formData = new FormData();
    formData.append("file", file);
    const response = await api.post("/documents/upload-and-ingest", formData, {
      headers: { "Content-Type": "multipart/form-data" },
    });
    return response.data;
  },
};

export default api;
