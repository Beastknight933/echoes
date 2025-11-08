// API client for Echoes backend

const API_BASE_URL = typeof import.meta !== 'undefined' && import.meta.env?.VITE_API_URL 
  ? import.meta.env.VITE_API_URL 
  : 'http://localhost:8000';

export interface EmbeddingResponse {
  embedding: number[];
}

export interface TimelineItem {
  text: string;
  similarity: number;
  metadata?: Record<string, any>;
}

export interface EraData {
  era: string;
  items: TimelineItem[];
  centroid?: number[];
}

export interface TimelineResponse {
  concept: string;
  eras: EraData[];
  semantic_shift?: number;
  primary_association?: {
    from: string;
    to: string;
  };
}

export interface EraQueryResponse {
  concept: string;
  era: string;
  items: TimelineItem[];
}

export interface SymbolPair {
  ancient: {
    path: string;
    title: string;
    description: string;
    era: string;
  };
  modern: {
    path: string;
    title: string;
    description: string;
    era: string;
  };
}

export interface SymbolPairsResponse {
  symbol: string;
  pairs: Record<string, SymbolPair>;
}

class EchoesAPI {
  private baseURL: string;
  private timeout: number = 5000; // 5 second timeout

  constructor(baseURL: string = API_BASE_URL) {
    this.baseURL = baseURL;
  }

  /**
   * Fetch with timeout
   */
  private async fetchWithTimeout(url: string, options: RequestInit = {}): Promise<Response> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(url, {
        ...options,
        signal: controller.signal,
      });
      clearTimeout(timeoutId);
      return response;
    } catch (error) {
      clearTimeout(timeoutId);
      throw error;
    }
  }

  /**
   * Health check endpoint
   */
  async health(): Promise<{ status: string }> {
    const response = await this.fetchWithTimeout(`${this.baseURL}/health`);
    if (!response.ok) {
      throw new Error(`Health check failed: ${response.statusText}`);
    }
    return response.json();
  }

  /**
   * Get embedding for a text string
   */
  async embed(text: string): Promise<EmbeddingResponse> {
    const response = await this.fetchWithTimeout(`${this.baseURL}/embed`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text }),
    });

    if (!response.ok) {
      throw new Error(`Embed failed: ${response.statusText}`);
    }
    return response.json();
  }

  /**
   * Get timeline data for a concept across eras
   */
  async getTimeline(concept: string, topN: number = 5): Promise<TimelineResponse> {
    const params = new URLSearchParams({
      concept: concept.toLowerCase(),
      top_n: topN.toString(),
    });

    const response = await this.fetchWithTimeout(`${this.baseURL}/timeline?${params}`);
    
    if (!response.ok) {
      throw new Error(`Timeline fetch failed: ${response.statusText}`);
    }
    return response.json();
  }

  /**
   * Get data for a specific era
   */
  async getEraData(concept: string, era: string, topN: number = 10): Promise<EraQueryResponse> {
    const params = new URLSearchParams({
      concept: concept.toLowerCase(),
      era,
      top_n: topN.toString(),
    });

    const response = await this.fetchWithTimeout(`${this.baseURL}/era?${params}`);
    
    if (!response.ok) {
      throw new Error(`Era data fetch failed: ${response.statusText}`);
    }
    return response.json();
  }

  /**
   * Get symbol pairs for pattern visualization
   */
  async getSymbolPairs(symbol: string): Promise<SymbolPairsResponse> {
    const params = new URLSearchParams({
      symbol: symbol.toLowerCase(),
    });

    const response = await this.fetchWithTimeout(`${this.baseURL}/symbol-pairs?${params}`);
    
    if (!response.ok) {
      throw new Error(`Symbol pairs fetch failed: ${response.statusText}`);
    }
    return response.json();
  }
}

// Export singleton instance
export const api = new EchoesAPI();

// Export class for custom instances
export default EchoesAPI;