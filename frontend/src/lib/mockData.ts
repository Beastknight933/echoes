export interface Node {
  id: string;
  label: string;
  size: number;
  color: string;
}

export interface Link {
  source: string;
  target: string;
  strength: number;
}

export interface ConceptData {
  concept: string;
  timeRange: string;
  narrative: {
    summary: string;
    semanticShift: number;
    primaryAssociation: {
      from: string;
      to: string;
    };
  };
  evolution: {
    [year: number]: {
      nodes: Node[];
      links: Link[];
    };
  };
  patterns: {
    [year: number]: {
      ancient: {
        title: string;
        description: string;
        era: string;
      };
      modern: {
        title: string;
        description: string;
        era: string;
      };
    };
  };
}

const mockDatasets: { [key: string]: ConceptData } = {
  Freedom: {
    concept: 'Freedom',
    timeRange: '3000 BCE - 2025 CE',
    narrative: {
      summary: 'The concept of "Freedom" evolved by a 35% semantic shift over the last 80 years, moving its primary association from "national independence" to "digital autonomy."',
      semanticShift: 35,
      primaryAssociation: {
        from: 'national independence',
        to: 'digital autonomy'
      }
    },
    evolution: {
      1940: {
        nodes: [
          { id: 'freedom', label: 'Freedom', size: 30, color: '#06b6d4' },
          { id: 'liberty', label: 'Liberty', size: 20, color: '#3b82f6' },
          { id: 'independence', label: 'Independence', size: 25, color: '#8b5cf6' },
          { id: 'sovereignty', label: 'Sovereignty', size: 18, color: '#06b6d4' },
          { id: 'democracy', label: 'Democracy', size: 22, color: '#3b82f6' },
        ],
        links: [
          { source: 'freedom', target: 'liberty', strength: 0.9 },
          { source: 'freedom', target: 'independence', strength: 0.95 },
          { source: 'freedom', target: 'sovereignty', strength: 0.7 },
          { source: 'independence', target: 'democracy', strength: 0.8 },
        ]
      },
      1980: {
        nodes: [
          { id: 'freedom', label: 'Freedom', size: 30, color: '#06b6d4' },
          { id: 'liberty', label: 'Liberty', size: 18, color: '#3b82f6' },
          { id: 'rights', label: 'Rights', size: 24, color: '#8b5cf6' },
          { id: 'expression', label: 'Expression', size: 20, color: '#06b6d4' },
          { id: 'choice', label: 'Choice', size: 22, color: '#3b82f6' },
          { id: 'democracy', label: 'Democracy', size: 19, color: '#8b5cf6' },
        ],
        links: [
          { source: 'freedom', target: 'rights', strength: 0.9 },
          { source: 'freedom', target: 'expression', strength: 0.85 },
          { source: 'freedom', target: 'choice', strength: 0.8 },
          { source: 'rights', target: 'democracy', strength: 0.7 },
        ]
      },
      2020: {
        nodes: [
          { id: 'freedom', label: 'Freedom', size: 30, color: '#06b6d4' },
          { id: 'privacy', label: 'Privacy', size: 26, color: '#3b82f6' },
          { id: 'autonomy', label: 'Autonomy', size: 24, color: '#8b5cf6' },
          { id: 'digital-rights', label: 'Digital Rights', size: 22, color: '#06b6d4' },
          { id: 'data', label: 'Data Control', size: 20, color: '#3b82f6' },
          { id: 'expression', label: 'Expression', size: 18, color: '#8b5cf6' },
          { id: 'encryption', label: 'Encryption', size: 16, color: '#06b6d4' },
        ],
        links: [
          { source: 'freedom', target: 'privacy', strength: 0.95 },
          { source: 'freedom', target: 'autonomy', strength: 0.9 },
          { source: 'freedom', target: 'digital-rights', strength: 0.88 },
          { source: 'privacy', target: 'data', strength: 0.85 },
          { source: 'digital-rights', target: 'encryption', strength: 0.8 },
        ]
      }
    },
    patterns: {
      1940: {
        ancient: {
          title: 'Liberty Bell',
          description: 'Symbol of American independence and national freedom',
          era: '1753 CE'
        },
        modern: {
          title: 'Victory Sign',
          description: 'WWII-era symbol representing freedom and peace',
          era: '1940s'
        }
      },
      1980: {
        ancient: {
          title: 'Statue of Liberty',
          description: 'Global icon of freedom and democracy',
          era: '1886 CE'
        },
        modern: {
          title: 'Berlin Wall Fall',
          description: 'Symbol of liberation and reunification',
          era: '1989'
        }
      },
      2020: {
        ancient: {
          title: 'Open Padlock',
          description: 'Traditional symbol of liberation and access',
          era: 'Ancient Symbol'
        },
        modern: {
          title: 'Encryption Key',
          description: 'Digital freedom through data privacy and security',
          era: '2020s'
        }
      }
    }
  },
  Circle: {
    concept: 'Circle',
    timeRange: '5000 BCE - 2025 CE',
    narrative: {
      summary: 'The "Circle" symbol has maintained remarkable consistency, representing wholeness, unity, and infinity across 7000 years. Modern usage shows a 58% increase in technological contexts.',
      semanticShift: 58,
      primaryAssociation: {
        from: 'cosmic perfection',
        to: 'user interface design'
      }
    },
    evolution: {
      1940: {
        nodes: [
          { id: 'circle', label: 'Circle', size: 30, color: '#06b6d4' },
          { id: 'unity', label: 'Unity', size: 22, color: '#3b82f6' },
          { id: 'infinity', label: 'Infinity', size: 20, color: '#8b5cf6' },
          { id: 'wholeness', label: 'Wholeness', size: 18, color: '#06b6d4' },
          { id: 'cycle', label: 'Cycle', size: 19, color: '#3b82f6' },
        ],
        links: [
          { source: 'circle', target: 'unity', strength: 0.85 },
          { source: 'circle', target: 'infinity', strength: 0.9 },
          { source: 'circle', target: 'wholeness', strength: 0.8 },
          { source: 'infinity', target: 'cycle', strength: 0.75 },
        ]
      },
      1980: {
        nodes: [
          { id: 'circle', label: 'Circle', size: 30, color: '#06b6d4' },
          { id: 'logo', label: 'Logo Design', size: 24, color: '#3b82f6' },
          { id: 'symbol', label: 'Symbol', size: 22, color: '#8b5cf6' },
          { id: 'unity', label: 'Unity', size: 20, color: '#06b6d4' },
          { id: 'brand', label: 'Branding', size: 21, color: '#3b82f6' },
        ],
        links: [
          { source: 'circle', target: 'logo', strength: 0.9 },
          { source: 'circle', target: 'symbol', strength: 0.85 },
          { source: 'logo', target: 'brand', strength: 0.8 },
        ]
      },
      2020: {
        nodes: [
          { id: 'circle', label: 'Circle', size: 30, color: '#06b6d4' },
          { id: 'ui', label: 'UI Element', size: 26, color: '#3b82f6' },
          { id: 'button', label: 'Button', size: 24, color: '#8b5cf6' },
          { id: 'icon', label: 'Icon', size: 22, color: '#06b6d4' },
          { id: 'avatar', label: 'Avatar', size: 20, color: '#3b82f6' },
          { id: 'loading', label: 'Loading Indicator', size: 19, color: '#8b5cf6' },
        ],
        links: [
          { source: 'circle', target: 'ui', strength: 0.95 },
          { source: 'circle', target: 'button', strength: 0.9 },
          { source: 'ui', target: 'icon', strength: 0.85 },
          { source: 'ui', target: 'avatar', strength: 0.8 },
          { source: 'ui', target: 'loading', strength: 0.75 },
        ]
      }
    },
    patterns: {
      1940: {
        ancient: {
          title: 'Sun Disk',
          description: 'Ancient Egyptian Ra symbol representing eternal life',
          era: '3000 BCE'
        },
        modern: {
          title: 'Olympic Rings',
          description: 'Symbol of global unity and athletic excellence',
          era: '1936'
        }
      },
      1980: {
        ancient: {
          title: 'Ouroboros',
          description: 'Ancient symbol of cyclical renewal and infinity',
          era: '1600 BCE'
        },
        modern: {
          title: 'Corporate Logos',
          description: 'Circle-based brand identities (BMW, Pepsi, Target)',
          era: '1980s'
        }
      },
      2020: {
        ancient: {
          title: 'Medicine Wheel',
          description: 'Indigenous symbol of balance and interconnection',
          era: 'Ancient Symbol'
        },
        modern: {
          title: 'App Icons',
          description: 'Circular UI elements in mobile and web design',
          era: '2020s'
        }
      }
    }
  },
  AI: {
    concept: 'AI',
    timeRange: '1950 - 2025 CE',
    narrative: {
      summary: 'The concept of "AI" transformed from theoretical mathematics to ubiquitous technology, showing a 72% semantic shift from "automation" to "intelligence augmentation" and creative partnership.',
      semanticShift: 72,
      primaryAssociation: {
        from: 'automation',
        to: 'intelligence augmentation'
      }
    },
    evolution: {
      1940: {
        nodes: [
          { id: 'ai', label: 'AI', size: 30, color: '#06b6d4' },
          { id: 'computing', label: 'Computing', size: 24, color: '#3b82f6' },
          { id: 'logic', label: 'Logic', size: 22, color: '#8b5cf6' },
          { id: 'algorithm', label: 'Algorithm', size: 20, color: '#06b6d4' },
          { id: 'machine', label: 'Machine', size: 18, color: '#3b82f6' },
        ],
        links: [
          { source: 'ai', target: 'computing', strength: 0.9 },
          { source: 'ai', target: 'logic', strength: 0.85 },
          { source: 'computing', target: 'algorithm', strength: 0.8 },
          { source: 'computing', target: 'machine', strength: 0.75 },
        ]
      },
      1980: {
        nodes: [
          { id: 'ai', label: 'AI', size: 30, color: '#06b6d4' },
          { id: 'expert-systems', label: 'Expert Systems', size: 24, color: '#3b82f6' },
          { id: 'neural-nets', label: 'Neural Networks', size: 22, color: '#8b5cf6' },
          { id: 'automation', label: 'Automation', size: 21, color: '#06b6d4' },
          { id: 'robotics', label: 'Robotics', size: 20, color: '#3b82f6' },
        ],
        links: [
          { source: 'ai', target: 'expert-systems', strength: 0.9 },
          { source: 'ai', target: 'neural-nets', strength: 0.85 },
          { source: 'ai', target: 'automation', strength: 0.8 },
          { source: 'automation', target: 'robotics', strength: 0.75 },
        ]
      },
      2020: {
        nodes: [
          { id: 'ai', label: 'AI', size: 30, color: '#06b6d4' },
          { id: 'ml', label: 'Machine Learning', size: 26, color: '#3b82f6' },
          { id: 'gpt', label: 'LLM/GPT', size: 25, color: '#8b5cf6' },
          { id: 'creativity', label: 'Creativity', size: 23, color: '#06b6d4' },
          { id: 'assistant', label: 'AI Assistant', size: 22, color: '#3b82f6' },
          { id: 'ethics', label: 'AI Ethics', size: 20, color: '#8b5cf6' },
          { id: 'art', label: 'Generative Art', size: 19, color: '#06b6d4' },
        ],
        links: [
          { source: 'ai', target: 'ml', strength: 0.95 },
          { source: 'ai', target: 'gpt', strength: 0.92 },
          { source: 'ai', target: 'creativity', strength: 0.88 },
          { source: 'gpt', target: 'assistant', strength: 0.85 },
          { source: 'ai', target: 'ethics', strength: 0.8 },
          { source: 'creativity', target: 'art', strength: 0.82 },
        ]
      }
    },
    patterns: {
      1940: {
        ancient: {
          title: 'Turing Machine',
          description: 'Theoretical foundation of computation and AI',
          era: '1936'
        },
        modern: {
          title: 'ENIAC Computer',
          description: 'First electronic general-purpose computer',
          era: '1945'
        }
      },
      1980: {
        ancient: {
          title: 'Chess Computer',
          description: 'Early AI demonstrating game-playing intelligence',
          era: '1970s'
        },
        modern: {
          title: 'Expert Systems',
          description: 'Rule-based AI for specialized domains',
          era: '1980s'
        }
      },
      2020: {
        ancient: {
          title: 'Neural Network Diagram',
          description: 'Biological inspiration for machine learning',
          era: '1950s Concept'
        },
        modern: {
          title: 'ChatGPT Interface',
          description: 'Conversational AI and creative partnership',
          era: '2020s'
        }
      }
    }
  }
};

export function getConceptData(concept: string): ConceptData {
  const normalizedConcept = concept.toLowerCase();
  
  // Find matching concept (case-insensitive)
  const matchedKey = Object.keys(mockDatasets).find(
    key => key.toLowerCase() === normalizedConcept
  );

  if (matchedKey) {
    return mockDatasets[matchedKey];
  }

  // Default fallback for any other concept
  return {
    concept: concept,
    timeRange: '2000 BCE - 2025 CE',
    narrative: {
      summary: `The concept of "${concept}" has undergone significant transformation throughout history, adapting to cultural, technological, and social changes across millennia.`,
      semanticShift: 42,
      primaryAssociation: {
        from: 'traditional meaning',
        to: 'contemporary interpretation'
      }
    },
    evolution: {
      1940: {
        nodes: [
          { id: 'main', label: concept, size: 30, color: '#06b6d4' },
          { id: 'tradition', label: 'Tradition', size: 22, color: '#3b82f6' },
          { id: 'culture', label: 'Culture', size: 20, color: '#8b5cf6' },
          { id: 'heritage', label: 'Heritage', size: 18, color: '#06b6d4' },
        ],
        links: [
          { source: 'main', target: 'tradition', strength: 0.9 },
          { source: 'main', target: 'culture', strength: 0.85 },
          { source: 'tradition', target: 'heritage', strength: 0.8 },
        ]
      },
      1980: {
        nodes: [
          { id: 'main', label: concept, size: 30, color: '#06b6d4' },
          { id: 'modern', label: 'Modernization', size: 24, color: '#3b82f6' },
          { id: 'global', label: 'Globalization', size: 22, color: '#8b5cf6' },
          { id: 'change', label: 'Social Change', size: 20, color: '#06b6d4' },
        ],
        links: [
          { source: 'main', target: 'modern', strength: 0.9 },
          { source: 'main', target: 'global', strength: 0.85 },
          { source: 'modern', target: 'change', strength: 0.8 },
        ]
      },
      2020: {
        nodes: [
          { id: 'main', label: concept, size: 30, color: '#06b6d4' },
          { id: 'digital', label: 'Digital', size: 26, color: '#3b82f6' },
          { id: 'innovation', label: 'Innovation', size: 24, color: '#8b5cf6' },
          { id: 'tech', label: 'Technology', size: 22, color: '#06b6d4' },
          { id: 'future', label: 'Future', size: 20, color: '#3b82f6' },
        ],
        links: [
          { source: 'main', target: 'digital', strength: 0.95 },
          { source: 'main', target: 'innovation', strength: 0.9 },
          { source: 'digital', target: 'tech', strength: 0.85 },
          { source: 'innovation', target: 'future', strength: 0.8 },
        ]
      }
    },
    patterns: {
      1940: {
        ancient: {
          title: `Ancient ${concept}`,
          description: `Historical representation in ancient civilizations`,
          era: 'Ancient Era'
        },
        modern: {
          title: `Mid-Century ${concept}`,
          description: `Industrial age interpretation and symbolism`,
          era: '1940s'
        }
      },
      1980: {
        ancient: {
          title: `Classical ${concept}`,
          description: `Traditional artistic and cultural representations`,
          era: 'Classical Period'
        },
        modern: {
          title: `Modern ${concept}`,
          description: `Contemporary cultural and media representation`,
          era: '1980s'
        }
      },
      2020: {
        ancient: {
          title: `Timeless ${concept}`,
          description: `Enduring symbols across cultures and time`,
          era: 'Timeless'
        },
        modern: {
          title: `Digital ${concept}`,
          description: `Contemporary digital and technological manifestation`,
          era: '2020s'
        }
      }
    }
  };
}