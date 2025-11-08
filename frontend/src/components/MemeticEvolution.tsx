import { useEffect, useRef, useState } from 'react';
import { motion } from 'motion/react';
import { Network, Calendar } from 'lucide-react';
import { Slider } from './ui/slider';
import { ConceptData, Node, Link } from '../lib/mockData';

interface MemeticEvolutionProps {
  data: ConceptData;
  year: number;
  onYearChange: (year: number) => void;
}

export function MemeticEvolution({ data, year, onYearChange }: MemeticEvolutionProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);
  const animationRef = useRef<number>();
  const nodesPositionRef = useRef<Map<string, { x: number; y: number; vx: number; vy: number }>>(new Map());

  const availableYears = Object.keys(data.evolution).map(Number).sort((a, b) => a - b);
  const currentYear = availableYears.reduce((prev, curr) => 
    Math.abs(curr - year) < Math.abs(prev - year) ? curr : prev
  );

  const graphData = data.evolution[currentYear];

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const width = rect.width;
    const height = rect.height;

    // Initialize or update node positions
    const nodePositions = nodesPositionRef.current;
    graphData.nodes.forEach(node => {
      if (!nodePositions.has(node.id)) {
        nodePositions.set(node.id, {
          x: Math.random() * width,
          y: Math.random() * height,
          vx: 0,
          vy: 0
        });
      }
    });

    // Remove nodes that no longer exist
    const currentNodeIds = new Set(graphData.nodes.map(n => n.id));
    Array.from(nodePositions.keys()).forEach(id => {
      if (!currentNodeIds.has(id)) {
        nodePositions.delete(id);
      }
    });

    // Force-directed layout simulation
    const simulate = () => {
      const centerX = width / 2;
      const centerY = height / 2;
      const iterations = 3;

      for (let i = 0; i < iterations; i++) {
        // Apply forces
        graphData.nodes.forEach(node => {
          const pos = nodePositions.get(node.id)!;
          
          // Center attraction
          const dx = centerX - pos.x;
          const dy = centerY - pos.y;
          pos.vx += dx * 0.001;
          pos.vy += dy * 0.001;

          // Repulsion between nodes
          graphData.nodes.forEach(otherNode => {
            if (node.id === otherNode.id) return;
            const otherPos = nodePositions.get(otherNode.id)!;
            const dx = pos.x - otherPos.x;
            const dy = pos.y - otherPos.y;
            const dist = Math.sqrt(dx * dx + dy * dy) + 1;
            const force = (node.size + otherNode.size) * 100 / (dist * dist);
            pos.vx += (dx / dist) * force;
            pos.vy += (dy / dist) * force;
          });
        });

        // Link forces
        graphData.links.forEach(link => {
          const sourcePos = nodePositions.get(link.source)!;
          const targetPos = nodePositions.get(link.target)!;
          if (!sourcePos || !targetPos) return;

          const dx = targetPos.x - sourcePos.x;
          const dy = targetPos.y - sourcePos.y;
          const dist = Math.sqrt(dx * dx + dy * dy) + 1;
          const force = (dist - 100) * link.strength * 0.01;

          sourcePos.vx += (dx / dist) * force;
          sourcePos.vy += (dy / dist) * force;
          targetPos.vx -= (dx / dist) * force;
          targetPos.vy -= (dy / dist) * force;
        });

        // Update positions with damping
        graphData.nodes.forEach(node => {
          const pos = nodePositions.get(node.id)!;
          pos.x += pos.vx;
          pos.y += pos.vy;
          pos.vx *= 0.85;
          pos.vy *= 0.85;

          // Boundary conditions
          const margin = node.size;
          pos.x = Math.max(margin, Math.min(width - margin, pos.x));
          pos.y = Math.max(margin, Math.min(height - margin, pos.y));
        });
      }
    };

    // Render function
    const render = () => {
      ctx.clearRect(0, 0, width, height);

      // Draw links
      graphData.links.forEach(link => {
        const sourcePos = nodePositions.get(link.source);
        const targetPos = nodePositions.get(link.target);
        if (!sourcePos || !targetPos) return;

        ctx.beginPath();
        ctx.moveTo(sourcePos.x, sourcePos.y);
        ctx.lineTo(targetPos.x, targetPos.y);
        ctx.strokeStyle = `rgba(6, 182, 212, ${link.strength * 0.3})`;
        ctx.lineWidth = link.strength * 2;
        ctx.stroke();
      });

      // Draw nodes
      graphData.nodes.forEach(node => {
        const pos = nodePositions.get(node.id);
        if (!pos) return;

        const isHovered = hoveredNode === node.id;
        const radius = node.size * (isHovered ? 1.2 : 1);

        // Glow effect
        const gradient = ctx.createRadialGradient(pos.x, pos.y, 0, pos.x, pos.y, radius * 2);
        gradient.addColorStop(0, `${node.color}40`);
        gradient.addColorStop(1, 'transparent');
        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, radius * 2, 0, Math.PI * 2);
        ctx.fill();

        // Node circle
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);
        ctx.fillStyle = node.color;
        ctx.fill();
        ctx.strokeStyle = isHovered ? '#ffffff' : `${node.color}80`;
        ctx.lineWidth = isHovered ? 3 : 2;
        ctx.stroke();

        // Label
        ctx.fillStyle = '#e2e8f0';
        ctx.font = `${isHovered ? '14px' : '12px'} "Inter", sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(node.label, pos.x, pos.y + radius + 15);
      });
    };

    // Animation loop
    const animate = () => {
      simulate();
      render();
      animationRef.current = requestAnimationFrame(animate);
    };

    animate();

    // Mouse interaction
    const handleMouseMove = (e: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;

      let foundNode: string | null = null;
      graphData.nodes.forEach(node => {
        const pos = nodePositions.get(node.id);
        if (!pos) return;
        const dist = Math.sqrt((x - pos.x) ** 2 + (y - pos.y) ** 2);
        if (dist < node.size) {
          foundNode = node.id;
        }
      });
      setHoveredNode(foundNode);
    };

    canvas.addEventListener('mousemove', handleMouseMove);

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
      canvas.removeEventListener('mousemove', handleMouseMove);
    };
  }, [graphData, hoveredNode, currentYear]);

  return (
    <div className="bg-slate-800/30 backdrop-blur-sm border border-slate-700/50 rounded-xl p-6 space-y-4 h-full">
      <div className="flex items-center gap-3 mb-4">
        <Network className="w-5 h-5 text-cyan-400" />
        <div>
          <h3 className="text-cyan-400">Memetic Evolution</h3>
          <p className="text-xs text-slate-500 font-mono mt-1">
            Conceptual Network Over Time
          </p>
        </div>
      </div>

      <motion.div
        key={currentYear}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.5 }}
        className="relative bg-slate-900/50 rounded-lg overflow-hidden"
        style={{ height: '400px' }}
      >
        <canvas
          ref={canvasRef}
          className="w-full h-full"
          style={{ cursor: hoveredNode ? 'pointer' : 'default' }}
        />
      </motion.div>

      <div className="space-y-3 pt-2">
        <div className="flex items-center justify-between text-sm">
          <div className="flex items-center gap-2">
            <Calendar className="w-4 h-4 text-slate-500" />
            <span className="text-slate-400 font-mono">Timeline</span>
          </div>
          <span className="text-cyan-400 font-mono">{currentYear}</span>
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
                y === currentYear ? 'text-cyan-400' : 'hover:text-slate-300'
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
