import { useState } from 'react';
import { SearchInterface } from './components/SearchInterface';
import { VisualizationView } from './components/VisualizationView';

export default function App() {
  const [currentView, setCurrentView] = useState<'search' | 'visualization'>('search');
  const [searchQuery, setSearchQuery] = useState('');

  const handleSearch = (query: string) => {
    setSearchQuery(query);
    setCurrentView('visualization');
  };

  const handleBack = () => {
    setCurrentView('search');
    setSearchQuery('');
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 text-slate-100">
      {currentView === 'search' ? (
        <SearchInterface onSearch={handleSearch} />
      ) : (
        <VisualizationView query={searchQuery} onBack={handleBack} />
      )}
    </div>
  );
}
