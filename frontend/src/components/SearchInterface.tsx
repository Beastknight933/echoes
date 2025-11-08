import { useState } from 'react';
import { Brain, Search, Wifi, WifiOff } from 'lucide-react';
import { motion } from 'motion/react';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { useBackendHealth } from '../hooks/useConceptData';

interface SearchInterfaceProps {
  onSearch: (query: string) => void;
}

export function SearchInterface({ onSearch }: SearchInterfaceProps) {
  const [input, setInput] = useState('');
  const [showAbout, setShowAbout] = useState(false);
  const { isHealthy, checking } = useBackendHealth();

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim()) {
      onSearch(input.trim());
    }
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center px-4">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
        className="max-w-4xl w-full text-center space-y-8"
      >
        {/* Backend Status Indicator */}
        {!checking && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="absolute top-4 right-4 flex items-center gap-2 px-3 py-2 bg-slate-800/50 backdrop-blur-sm rounded-full border border-slate-700/50"
          >
            {isHealthy ? (
              <>
                <Wifi className="w-4 h-4 text-green-400" />
                <span className="text-xs text-green-400 font-mono">Backend Connected</span>
              </>
            ) : (
              <>
                <WifiOff className="w-4 h-4 text-amber-400" />
                <span className="text-xs text-amber-400 font-mono">Offline Mode</span>
              </>
            )}
          </motion.div>
        )}

        {/* Offline Mode Info Banner */}
        {!checking && !isHealthy && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="absolute top-20 left-1/2 -translate-x-1/2 max-w-md mx-auto"
          >
            <div className="bg-amber-500/10 border border-amber-500/30 rounded-lg px-4 py-2 text-center backdrop-blur-sm">
              <p className="text-xs text-amber-300">
                Backend not connected. Using demo data for <span className="font-mono">Freedom</span>, <span className="font-mono">Circle</span>, and <span className="font-mono">AI</span>
              </p>
            </div>
          </motion.div>
        )}

        {/* Header */}
        <div className="space-y-4">
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ delay: 0.2, type: "spring", stiffness: 200 }}
            className="flex items-center justify-center gap-3 mb-6"
          >
            <Brain 
              className="w-12 h-12 text-cyan-400 cursor-pointer hover:text-cyan-300 transition-colors"
              onClick={() => setShowAbout(!showAbout)}
            />
            <h1 className="text-5xl md:text-7xl tracking-tight bg-gradient-to-r from-cyan-400 via-blue-500 to-purple-600 bg-clip-text text-transparent">
              Echoes
            </h1>
          </motion.div>

          <motion.h2
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.4 }}
            className="text-2xl md:text-3xl text-slate-300"
          >
            Decoding the Evolution of Human Thought
          </motion.h2>

          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.6 }}
            className="text-slate-400 text-lg max-w-2xl mx-auto"
          >
            From ancient carvings to viral tweets — decoding how human ideas evolve through time.
          </motion.p>
        </div>

        {/* About Modal */}
        {showAbout && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="bg-slate-800/50 backdrop-blur-lg border border-cyan-500/30 rounded-lg p-6 text-left"
          >
            <div className="flex items-start gap-3">
              <Brain className="w-6 h-6 text-cyan-400 flex-shrink-0 mt-1" />
              <div>
                <h3 className="text-cyan-400 mb-2">The Memetic Lens</h3>
                <p className="text-slate-300 text-sm leading-relaxed">
                  Our AI system analyzes the evolution of concepts across millennia, 
                  tracking semantic shifts, visual patterns, and cultural transformations. 
                  Enter any word or symbol to decode its journey through human history.
                </p>
              </div>
            </div>
          </motion.div>
        )}

        {/* Search Form */}
        <motion.form
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.8 }}
          onSubmit={handleSubmit}
          className="space-y-4"
        >
          <div className="relative max-w-2xl mx-auto">
            <Input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Type a Concept (e.g., Freedom, Circle, AI)..."
              className="h-16 px-6 text-lg bg-slate-800/50 border-2 border-slate-700 focus:border-cyan-500 transition-all duration-300 rounded-xl backdrop-blur-sm"
            />
            <Search className="absolute right-6 top-1/2 -translate-y-1/2 w-6 h-6 text-slate-500" />
          </div>

          <Button
            type="submit"
            className="h-14 px-10 bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white shadow-lg shadow-cyan-500/50 transition-all duration-300 hover:shadow-xl hover:shadow-cyan-500/70 hover:scale-105"
          >
            Decode
          </Button>
        </motion.form>

        {/* Suggestion Pills */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1 }}
          className="flex flex-wrap gap-3 justify-center"
        >
          {['Freedom', 'Circle', 'AI', 'Love', 'War'].map((suggestion) => (
            <button
              key={suggestion}
              onClick={() => setInput(suggestion)}
              className="px-4 py-2 bg-slate-800/30 border border-slate-700/50 rounded-full text-sm text-slate-400 hover:text-cyan-400 hover:border-cyan-500/50 transition-all duration-300"
            >
              {suggestion}
            </button>
          ))}
        </motion.div>
      </motion.div>

      {/* Background Effects */}
      <div className="fixed inset-0 -z-10 overflow-hidden pointer-events-none">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-cyan-500/10 rounded-full blur-3xl animate-pulse" />
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-blue-500/10 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '1s' }} />
      </div>
    </div>
  );
}