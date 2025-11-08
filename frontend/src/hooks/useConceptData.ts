import { useState, useEffect } from 'react';
import { api } from '../lib/api';
import { ConceptData, getConceptData as getMockData } from '../lib/mockData';
import { transformTimelineToConceptData } from '../lib/dataTransform';

interface UseConceptDataResult {
  data: ConceptData | null;
  loading: boolean;
  error: Error | null;
  refetch: () => void;
}

/**
 * Hook to fetch concept data from backend API with fallback to mock data
 */
export function useConceptData(concept: string): UseConceptDataResult {
  const [data, setData] = useState<ConceptData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const fetchData = async () => {
    if (!concept) {
      setLoading(false);
      return;
    }

    setLoading(true);
    setError(null);

    try {
      // Try to fetch from backend
      const timelineData = await api.getTimeline(concept, 10);
      
      // Try to fetch symbol pairs
      let symbolData;
      try {
        symbolData = await api.getSymbolPairs(concept);
      } catch (symbolError) {
        // Symbol pairs are optional, silently continue
      }

      // Transform backend data to frontend format
      const conceptData = transformTimelineToConceptData(
        concept,
        timelineData,
        symbolData
      );

      setData(conceptData);
    } catch (err) {
      // Backend not available - use mock data
      console.log(`📡 Backend unavailable - using offline mode for "${concept}"`);
      
      // Fallback to mock data
      try {
        const mockData = getMockData(concept);
        setData(mockData);
        setError(null); // Don't show error if fallback succeeds
      } catch (mockErr) {
        setError(err instanceof Error ? err : new Error('Failed to fetch data'));
      }
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, [concept]);

  return {
    data,
    loading,
    error,
    refetch: fetchData
  };
}

/**
 * Hook to check backend health
 */
export function useBackendHealth() {
  const [isHealthy, setIsHealthy] = useState<boolean | null>(null);
  const [checking, setChecking] = useState(true);

  useEffect(() => {
    const checkHealth = async () => {
      try {
        await api.health();
        setIsHealthy(true);
      } catch (err) {
        console.warn('Backend not available:', err);
        setIsHealthy(false);
      } finally {
        setChecking(false);
      }
    };

    checkHealth();
  }, []);

  return { isHealthy, checking };
}