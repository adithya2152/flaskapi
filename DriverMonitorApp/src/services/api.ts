const BASE_URL = 'http://192.168.1.100:5000'; // Replace with your Flask server IP

export interface ApiResponse {
  success: boolean;
  message?: string;
  data?: any;
}

export class ApiService {
  static async registerUser(name: string, imageBase64: string): Promise<ApiResponse> {
    try {
      const response = await fetch(`${BASE_URL}/api/register`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          name,
          image: imageBase64,
        }),
      });

      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Registration error:', error);
      return { success: false, message: 'Network error during registration' };
    }
  }

  static async loginUser(imageBase64: string): Promise<ApiResponse> {
    try {
      const response = await fetch(`${BASE_URL}/api/login`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image: imageBase64,
        }),
      });

      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Login error:', error);
      return { success: false, message: 'Network error during login' };
    }
  }

  static async sendFrame(imageBase64: string, username: string): Promise<ApiResponse> {
    try {
      const response = await fetch(`${BASE_URL}/api/monitor`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image: imageBase64,
          username,
        }),
      });

      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Frame monitoring error:', error);
      return { success: false, message: 'Network error during monitoring' };
    }
  }

  static async getAlerts(): Promise<string[]> {
    try {
      const response = await fetch(`${BASE_URL}/api/alerts`);
      const data = await response.json();
      return data.alerts || [];
    } catch (error) {
      console.error('Error fetching alerts:', error);
      return [];
    }
  }
}</parameter>