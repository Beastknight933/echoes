// Transform backend API responses to frontend data structures

import { TimelineResponse, EraQueryResponse, SymbolPairsResponse, TimelineItem } from './api';
import { ConceptData, Node, Link } from './mockData';

/**
 * Convert backend timeline data to frontend ConceptData format
 */
export function transformTimelineToConceptData(
  concept: string,
  timelineData: TimelineResponse,
  symbolData?: SymbolPairsResponse
): ConceptData {
  const eras = timelineData.eras || [];
  
  // Extract year keys from eras (convert "1900s" to 1940, "2020s" to 2020, etc.)
  const eraToYear = (era: string): number => {
    const match = era.match(/(\d{4})s?/);
    if (match) {
      const baseYear = parseInt(match[1]);
      // Convert decade to mid-point year for display
      return baseYear + (era.includes('s') ? 40 : 0);
    }
    return 2000; // fallback
  };

  // Calculate time range
  const years = eras.map(e => eraToYear(e.era)).sort((a, b) => a - b);
  const timeRange = years.length > 0 
    ? `${Math.min(...years)} - ${Math.max(...years)} CE`
    : '1900 - 2025 CE';

  // Build narrative from backend data
  const narrative = {
    summary: timelineData.semantic_shift 
      ? `The concept of "${concept}" evolved by a ${Math.round(timelineData.semantic_shift)}% semantic shift, moving its primary association from "${timelineData.primary_association?.from || 'traditional contexts'}" to "${timelineData.primary_association?.to || 'contemporary usage'}".`
      : `The concept of "${concept}" has undergone significant transformation throughout history, adapting to cultural, technological, and social changes.`,
    semanticShift: timelineData.semantic_shift ? Math.round(timelineData.semantic_shift) : 42,
    primaryAssociation: timelineData.primary_association || {
      from: 'traditional meaning',
      to: 'contemporary interpretation'
    }
  };

  // Build evolution graph data for each era
  const evolution: { [year: number]: { nodes: Node[]; links: Link[] } } = {};
  
  eras.forEach((eraData, eraIndex) => {
    const year = eraToYear(eraData.era);
    const items = eraData.items || [];
    
    // Create main concept node
    const nodes: Node[] = [
      {
        id: 'main',
        label: concept,
        size: 30,
        color: '#06b6d4'
      }
    ];

    // Create nodes from top items
    const colors = ['#3b82f6', '#8b5cf6', '#06b6d4', '#3b82f6', '#8b5cf6'];
    items.slice(0, 6).forEach((item, idx) => {
      // Extract key terms from text (simple word extraction)
      const words = item.text.split(/\s+/).filter(w => w.length > 3);
      const label = words[0] || `Term ${idx + 1}`;
      
      nodes.push({
        id: `node-${idx}`,
        label: label.charAt(0).toUpperCase() + label.slice(1),
        size: 20 + (item.similarity || 0) * 10,
        color: colors[idx % colors.length]
      });
    });

    // Create links based on similarity
    const links: Link[] = [];
    items.slice(0, 6).forEach((item, idx) => {
      links.push({
        source: 'main',
        target: `node-${idx}`,
        strength: item.similarity || 0.5
      });
    });

    evolution[year] = { nodes, links };
  });

  // Build pattern data from symbol pairs
  const patterns: { [year: number]: any } = {};
  
  if (symbolData?.pairs) {
    Object.keys(symbolData.pairs).forEach(eraKey => {
      const year = eraToYear(eraKey);
      const pair = symbolData.pairs[eraKey];
      
      patterns[year] = {
        ancient: {
          title: pair.ancient.title,
          description: pair.ancient.description,
          era: pair.ancient.era,
          imagePath: pair.ancient.path
        },
        modern: {
          title: pair.modern.title,
          description: pair.modern.description,
          era: pair.modern.era,
          imagePath: pair.modern.path
        }
      };
    });
  } else {
    // Fallback pattern data based on eras
    eras.forEach(eraData => {
      const year = eraToYear(eraData.era);
      patterns[year] = {
        ancient: {
          title: `Historical ${concept}`,
          description: `Representation from earlier period`,
          era: eraData.era
        },
        modern: {
          title: `Modern ${concept}`,
          description: `Contemporary interpretation`,
          era: eraData.era
        }
      };
    });
  }

  return {
    concept,
    timeRange,
    narrative,
    evolution,
    patterns
  };
}

/**
 * Calculate semantic shift between two eras based on item similarities
 */
export function calculateSemanticShift(
  era1Items: TimelineItem[],
  era2Items: TimelineItem[]
): number {
  // Simple heuristic: compare average similarities
  const avg1 = era1Items.reduce((sum, item) => sum + (item.similarity || 0), 0) / (era1Items.length || 1);
  const avg2 = era2Items.reduce((sum, item) => sum + (item.similarity || 0), 0) / (era2Items.length || 1);
  
  return Math.abs(avg2 - avg1) * 100;
}

/**
 * Extract primary associations from top items
 */
export function extractAssociations(
  oldEraItems: TimelineItem[],
  newEraItems: TimelineItem[]
): { from: string; to: string } {
  // Get most relevant term from each era
  const extractTopTerm = (items: TimelineItem[]) => {
    if (items.length === 0) return 'unknown';
    const topItem = items[0];
    const words = topItem.text.split(/\s+/).filter(w => w.length > 3);
    return words[0] || 'concept';
  };

  return {
    from: extractTopTerm(oldEraItems),
    to: extractTopTerm(newEraItems)
  };
}
