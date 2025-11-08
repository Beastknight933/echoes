import { motion } from 'motion/react';
import { Brain, TrendingUp } from 'lucide-react';
import { ConceptData } from '../lib/mockData';

interface NarrativeSummaryProps {
  data: ConceptData;
}

export function NarrativeSummary({ data }: NarrativeSummaryProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-gradient-to-br from-slate-800/50 to-slate-900/50 backdrop-blur-lg border border-cyan-500/30 rounded-xl p-6 md:p-8 shadow-xl shadow-cyan-500/10"
    >
      <div className="flex items-start gap-4">
        <div className="p-3 bg-cyan-500/10 rounded-lg">
          <Brain className="w-8 h-8 text-cyan-400" />
        </div>
        
        <div className="flex-1 space-y-4">
          <div>
            <h2 className="text-cyan-400 mb-2 flex items-center gap-2">
              <span className="font-mono text-sm">ECHOES AI DECODES:</span>
              <TrendingUp className="w-4 h-4" />
            </h2>
            <blockquote className="text-slate-100 text-lg leading-relaxed border-l-4 border-cyan-500/50 pl-4">
              {data.narrative.summary}
            </blockquote>
          </div>

          <div className="flex flex-wrap gap-6 pt-2">
            <div className="space-y-1">
              <p className="text-xs text-slate-500 uppercase tracking-wider font-mono">
                Semantic Shift
              </p>
              <p className="text-2xl text-cyan-400">
                {data.narrative.semanticShift}%
              </p>
            </div>

            <div className="space-y-1">
              <p className="text-xs text-slate-500 uppercase tracking-wider font-mono">
                Primary Association
              </p>
              <p className="text-sm text-slate-300">
                <span className="text-slate-500">From:</span>{' '}
                <span className="text-blue-400">{data.narrative.primaryAssociation.from}</span>
                {' → '}
                <span className="text-purple-400">{data.narrative.primaryAssociation.to}</span>
              </p>
            </div>

            <div className="space-y-1">
              <p className="text-xs text-slate-500 uppercase tracking-wider font-mono">
                Time Range
              </p>
              <p className="text-sm text-slate-300">
                {data.timeRange}
              </p>
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  );
}
