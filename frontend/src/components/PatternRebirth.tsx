import { motion } from 'motion/react';
import { ImageIcon, Calendar, ArrowRight } from 'lucide-react';
import { Slider } from './ui/slider';
import { ConceptData } from '../lib/mockData';
import { ImageWithFallback } from './figma/ImageWithFallback';

interface PatternRebirthProps {
  data: ConceptData;
  year: number;
  onYearChange: (year: number) => void;
}

export function PatternRebirth({ data, year, onYearChange }: PatternRebirthProps) {
  const availableYears = Object.keys(data.patterns).map(Number).sort((a, b) => a - b);
  const currentYear = availableYears.reduce((prev, curr) => 
    Math.abs(curr - year) < Math.abs(prev - year) ? curr : prev
  );

  const patternData = data.patterns[currentYear];

  // Get image source - use backend path if available, otherwise Unsplash
  const getImageSource = (patternInfo: any, isAncient: boolean) => {
    // If backend provides an image path, use it
    if (patternInfo.imagePath) {
      // Construct full URL if it's a relative path
      const apiUrl = typeof import.meta !== 'undefined' && import.meta.env?.VITE_API_URL 
        ? import.meta.env.VITE_API_URL 
        : 'http://localhost:8000';
      if (patternInfo.imagePath.startsWith('http')) {
        return patternInfo.imagePath;
      }
      return `${apiUrl}${patternInfo.imagePath}`;
    }
    
    // Otherwise use Unsplash search query
    return getImageQuery(data.concept, patternInfo.era, isAncient);
  };

  // Map concept to search queries (fallback for Unsplash)
  const getImageQuery = (concept: string, era: string, isAncient: boolean) => {
    const conceptLower = concept.toLowerCase();
    
    if (conceptLower === 'freedom') {
      if (isAncient) {
        if (era.includes('1753')) return 'liberty bell historical';
        if (era.includes('1886')) return 'statue of liberty monument';
        return 'ancient freedom symbol';
      } else {
        if (era.includes('1940')) return 'victory peace sign WWII';
        if (era.includes('1989')) return 'berlin wall fall';
        return 'digital privacy encryption';
      }
    }
    
    if (conceptLower === 'circle') {
      if (isAncient) {
        if (era.includes('3000 BCE')) return 'egyptian sun disk ancient';
        if (era.includes('1600 BCE')) return 'ouroboros snake symbol';
        return 'medicine wheel indigenous';
      } else {
        if (era.includes('1936')) return 'olympic rings flag';
        if (era.includes('1980')) return 'corporate logo circle design';
        return 'mobile app icons interface';
      }
    }
    
    if (conceptLower === 'ai') {
      if (isAncient) {
        if (era.includes('1936')) return 'turing machine vintage computer';
        if (era.includes('1970')) return 'chess computer vintage';
        return 'brain neural network diagram';
      } else {
        if (era.includes('1945')) return 'ENIAC vintage computer technology';
        if (era.includes('1980')) return 'retro computer expert system';
        return 'artificial intelligence robot modern';
      }
    }
    
    // Default queries
    if (isAncient) {
      return `ancient ${conceptLower} symbol historical`;
    } else {
      return `modern ${conceptLower} digital technology`;
    }
  };

  return (
    <div className="bg-slate-800/30 backdrop-blur-sm border border-slate-700/50 rounded-xl p-6 space-y-4 h-full">
      <div className="flex items-center gap-3 mb-4">
        <ImageIcon className="w-5 h-5 text-purple-400" />
        <div>
          <h3 className="text-purple-400">Pattern Rebirth</h3>
          <p className="text-xs text-slate-500 font-mono mt-1">
            Visual Evolution Across Eras
          </p>
        </div>
      </div>

      <motion.div
        key={currentYear}
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5 }}
        className="space-y-4"
      >
        {/* Comparison View */}
        <div className="grid grid-cols-2 gap-4">
          {/* Ancient Pattern */}
          <div className="space-y-3">
            <div className="relative bg-slate-900/50 rounded-lg overflow-hidden aspect-square">
              <ImageWithFallback
                src={getImageSource(patternData.ancient, true)}
                alt={patternData.ancient.title}
                className="w-full h-full object-cover"
              />
              <div className="absolute top-2 left-2 px-2 py-1 bg-blue-500/80 backdrop-blur-sm rounded text-xs">
                Ancient
              </div>
            </div>
            <div className="space-y-1">
              <h4 className="text-sm text-slate-200">{patternData.ancient.title}</h4>
              <p className="text-xs text-slate-500">{patternData.ancient.era}</p>
              <p className="text-xs text-slate-400 leading-relaxed">
                {patternData.ancient.description}
              </p>
            </div>
          </div>

          {/* Modern Pattern */}
          <div className="space-y-3">
            <div className="relative bg-slate-900/50 rounded-lg overflow-hidden aspect-square">
              <ImageWithFallback
                src={getImageSource(patternData.modern, false)}
                alt={patternData.modern.title}
                className="w-full h-full object-cover"
              />
              <div className="absolute top-2 left-2 px-2 py-1 bg-purple-500/80 backdrop-blur-sm rounded text-xs">
                Modern
              </div>
            </div>
            <div className="space-y-1">
              <h4 className="text-sm text-slate-200">{patternData.modern.title}</h4>
              <p className="text-xs text-slate-500">{patternData.modern.era}</p>
              <p className="text-xs text-slate-400 leading-relaxed">
                {patternData.modern.description}
              </p>
            </div>
          </div>
        </div>

        {/* Evolution Arrow */}
        <div className="flex items-center justify-center py-2">
          <div className="flex items-center gap-2 px-4 py-2 bg-slate-900/50 rounded-full border border-slate-700/50">
            <span className="text-xs text-slate-400 font-mono">Evolution Path</span>
            <ArrowRight className="w-4 h-4 text-cyan-400" />
          </div>
        </div>
      </motion.div>

      {/* Timeline Slider */}
      <div className="space-y-3 pt-2">
        <div className="flex items-center justify-between text-sm">
          <div className="flex items-center gap-2">
            <Calendar className="w-4 h-4 text-slate-500" />
            <span className="text-slate-400 font-mono">Era</span>
          </div>
          <span className="text-purple-400 font-mono">{currentYear}</span>
        </div>
        
        <Slider
          value={[year]}
          onValueChange={([value]) => onYearChange(value)}
          min={Math.min(...availableYears)}
          max={Math.max(...availableYears)}
          step={1}
          className="w-full"
        />

        <div className="flex justify-between text-xs text-slate-500 font-mono">
          {availableYears.map(y => (
            <button
              key={y}
              onClick={() => onYearChange(y)}
              className={`transition-colors ${
                y === currentYear ? 'text-purple-400' : 'hover:text-slate-300'
              }`}
            >
              {y}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}