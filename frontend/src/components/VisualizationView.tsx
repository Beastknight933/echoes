import { useState } from 'react';
import { ArrowLeft, Brain, RefreshCw, AlertCircle } from 'lucide-react';
import { motion } from 'motion/react';
import { Button } from './ui/button';
import { MemeticEvolution } from './MemeticEvolution';
import { PatternRebirth } from './PatternRebirth';
import { NarrativeSummary } from './NarrativeSummary';
import { useConceptData } from '../hooks/useConceptData';
import { Alert, AlertDescription } from './ui/alert';

interface VisualizationViewProps {
  query: string;
  onBack: () => void;
}

export function VisualizationView({ query, onBack }: VisualizationViewProps) {
  const [timelineYear, setTimelineYear] = useState(2020);
  const { data: conceptData, loading, error, refetch } = useConceptData(query);

  // Loading state
  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center p-4">
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="text-center space-y-4"
        >
          <div className="w-16 h-16 border-4 border-cyan-500/30 border-t-cyan-500 rounded-full animate-spin mx-auto" />
          <div className="space-y-2">
            <h2 className="text-xl text-cyan-400">Decoding {query}...</h2>
            <p className="text-slate-400 text-sm">
              Analyzing semantic patterns across time
            </p>
          </div>
        </motion.div>
      </div>
    );
  }

  // Error state
  if (error || !conceptData) {
    return (
      <div className="min-h-screen flex items-center justify-center p-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="max-w-md w-full space-y-4"
        >
          <Alert className="border-red-500/50 bg-red-500/10">
            <AlertCircle className="w-5 h-5 text-red-400" />
            <AlertDescription className="text-slate-300">
              {error?.message || 'Failed to load concept data. Please try again.'}
            </AlertDescription>
          </Alert>
          <div className="flex gap-3 justify-center">
            <Button onClick={onBack} variant="outline">
              <ArrowLeft className="w-4 h-4 mr-2" />
              Back to Search
            </Button>
            <Button onClick={refetch} className="bg-cyan-500 hover:bg-cyan-600">
              <RefreshCw className="w-4 h-4 mr-2" />
              Retry
            </Button>
          </div>
        </motion.div>
      </div>
    );
  }

  return (
    <div className="min-h-screen p-4 md:p-8">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="max-w-7xl mx-auto mb-8"
      >
        <div className="flex items-center justify-between mb-6">
          <Button
            onClick={onBack}
            variant="ghost"
            className="text-slate-400 hover:text-cyan-400 transition-colors"
          >
            <ArrowLeft className="w-5 h-5 mr-2" />
            New Search
          </Button>

          <div className="flex items-center gap-2">
            <Brain className="w-6 h-6 text-cyan-400" />
            <span className="text-slate-400 text-sm font-mono">MEMETIC LENS v2.0</span>
          </div>
        </div>

        <div className="space-y-2">
          <h1 className="text-4xl md:text-5xl bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
            Analyzing: {query}
          </h1>
          <p className="text-slate-400">
            Decoding patterns across {conceptData.timeRange}
          </p>
        </div>
      </motion.div>

      {/* Main Content Grid */}
      <div className="max-w-7xl mx-auto space-y-8">
        {/* Narrative Summary */}
        <NarrativeSummary data={conceptData} />

        {/* Visualizations Grid */}
        <div className="grid lg:grid-cols-2 gap-8">
          {/* Memetic Evolution */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
          >
            <MemeticEvolution 
              data={conceptData} 
              year={timelineYear}
              onYearChange={setTimelineYear}
            />
          </motion.div>

          {/* Pattern Rebirth */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.4 }}
          >
            <PatternRebirth 
              data={conceptData}
              year={timelineYear}
              onYearChange={setTimelineYear}
            />
          </motion.div>
        </div>
      </div>

      {/* Background Effects */}
      <div className="fixed inset-0 -z-10 overflow-hidden pointer-events-none">
        <div className="absolute top-0 left-1/3 w-96 h-96 bg-cyan-500/5 rounded-full blur-3xl" />
        <div className="absolute bottom-0 right-1/3 w-96 h-96 bg-blue-500/5 rounded-full blur-3xl" />
      </div>
    </div>
  );
}