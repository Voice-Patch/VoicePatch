import { useState } from 'react';
import { Play, Info, SlidersHorizontal } from 'lucide-react';
import WaveformVisualizer from './WaveformVisualizer';

const Slider = ({ label, value, onChange, min, max, step, info }) => (
    <div className="relative">
        <label className="block text-sm font-medium text-slate-300 mb-1">{label}</label>
        {info && (
            <div className="group absolute top-0 right-0">
                <Info size={16} className="text-slate-500 cursor-pointer" />
                <div className="absolute bottom-full right-0 mb-2 w-48 p-2 text-xs text-white bg-slate-800 border border-slate-700 rounded-md opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none">
                    {info}
                </div>
            </div>
        )}
        <input
            type="range"
            min={min}
            max={max}
            step={step}
            value={value}
            onChange={onChange}
            className="w-full h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-teal-400 focus:outline-none"
        />
        <div className="text-right text-xs text-slate-400 mt-1">{value}</div>
    </div>
);

const AnalysisSetup = ({ audioSource, onStartProcessing, vadParams, setVadParams, isProcessing, vadGaps, hoveredGapIndex }) => {
    const [paramsVisible, setParamsVisible] = useState(false);

    return (
        <div className="animate-fade-in-up">
            <div className="glass-card p-8 flex flex-col items-center">
                <h3 className="text-2xl font-bold mb-2 gradient-text">Analyze Audio & Detect Gaps</h3>
                <p className="text-slate-400 mb-6 max-w-md truncate">{audioSource.name}</p>
                <div className="w-full max-w-3xl mb-6">
                   <WaveformVisualizer 
                        audioUrl={audioSource.url} 
                        vadGaps={vadGaps} 
                        hoveredGapIndex={hoveredGapIndex}
                        isPlayer={true} 
                    />
                </div>
                
                <div className={`w-full max-w-2xl transition-all duration-500 ease-in-out ${isProcessing ? 'max-h-0 opacity-0' : 'max-h-[500px] opacity-100'}`}>
                    <div className="overflow-hidden">
                        <div className="flex items-center justify-center gap-4">
                            <button 
                                onClick={() => setParamsVisible(!paramsVisible)}
                                className="flex-shrink-0 w-12 h-12 flex items-center justify-center bg-black/20 rounded-lg hover:bg-black/30 transition-colors"
                                title="Tune VAD Parameters"
                            >
                                <SlidersHorizontal size={20} />
                            </button>

                            {/* "Reconstruct" button */}
                            <button 
                                onClick={onStartProcessing} 
                                className="group relative flex-shrink-0 inline-flex items-center justify-center bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-6 rounded-lg text-base transition-all duration-300 transform hover:scale-105"
                            >
                                 <span className="relative flex items-center gap-2">
                                    <Play size={18} /> Reconstruct Audio
                                 </span>
                            </button>
                        </div>

                        {/* Collapsible Parameters Section - now appears below the button row */}
                        <div className={`transition-all duration-500 ease-in-out grid md:grid-cols-3 gap-6 bg-black/20 p-4 rounded-lg ${paramsVisible ? 'max-h-screen mt-4 pt-4' : 'max-h-0 !p-0'}`}>
                            <Slider 
                                label="Silence Sensitivity"
                                info="Adjust how sensitive the AI is to detecting silence. Higher values detect shorter pauses."
                                value={vadParams.threshold}
                                onChange={(e) => setVadParams(p => ({...p, threshold: e.target.value}))}
                                min="0.1" max="0.9" step="0.05"
                            />
                            <Slider 
                                label="Min Speech (ms)"
                                info="The shortest duration of sound to be considered speech."
                                value={vadParams.minSpeechMs}
                                onChange={(e) => setVadParams(p => ({...p, minSpeechMs: e.target.value}))}
                                min="50" max="1000" step="10"
                            />
                            <Slider 
                                label="Min Silence (ms)"
                                info="The shortest duration of silence to be considered a gap."
                                value={vadParams.minSilenceMs}
                                onChange={(e) => setVadParams(p => ({...p, minSilenceMs: e.target.value}))}
                                min="50" max="1000" step="10"
                            />
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default AnalysisSetup;
