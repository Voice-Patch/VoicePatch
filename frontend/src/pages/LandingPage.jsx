import { Mic, RotateCw, CassetteTape, ChevronsRight, AlertTriangle } from 'lucide-react';

const FeatureCard = ({ icon, title, description, delay }) => (
    <div 
        className="glass-card p-6 text-left animate-fade-in-up"
        style={{ animationDelay: `${delay}s` }}
    >
        <div className="flex items-center mb-4">
            <div className="p-3 bg-white/5 rounded-lg mr-4 border border-white/10">
                {icon}
            </div>
            <h3 className="text-lg font-bold text-slate-100">{title}</h3>
        </div>
        <p className="text-slate-400 text-sm">{description}</p>
    </div>
);


const LandingPage = ({ onStartDemo, isEngineReady, engineStatusMessage }) => {
    const hasError = engineStatusMessage.startsWith('Error:');

    return (
        <div className="flex flex-col items-center justify-center min-h-[90vh] text-center">
            <header className="animate-fade-in-up max-w-4xl">
                <div className={`inline-flex items-center gap-2 border rounded-full px-4 py-1.5 mb-6 text-sm transition-colors ${
                    hasError ? 'border-red-500/50 bg-red-500/10 text-red-300' : 'bg-white/5 border-white/10 text-slate-300'
                }`}>
                    {hasError ? (
                        <AlertTriangle size={14} />
                    ) : (
                        <span className="relative flex h-2 w-2">
                            <span className={`animate-ping absolute inline-flex h-full w-full rounded-full ${isEngineReady ? 'bg-teal-400' : 'bg-amber-400'} opacity-75`}></span>
                            <span className={`relative inline-flex rounded-full h-2 w-2 ${isEngineReady ? 'bg-teal-500' : 'bg-amber-500'}`}></span>
                        </span>
                    )}
                    {engineStatusMessage}
                </div>
                <h1 className="text-7xl md:text-9xl font-black text-slate-50 mb-4">
                    <span className="gradient-text">VoicePatch</span>
                </h1>
                <p className="text-lg md:text-xl text-slate-400 max-w-2xl mx-auto mb-10">
                    Our AI pipeline intelligently transcribes, detects gaps, and reconstructs speech to generate flawless, intellegible audio.
                </p>
                <button
                    onClick={onStartDemo}
                    disabled={!isEngineReady}
                    className="group relative inline-flex items-center justify-center bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white font-bold py-4 px-10 rounded-lg text-lg transition-all duration-300 ease-in-out transform hover:scale-105 overflow-hidden"
                >
                    <span className="absolute left-0 top-0 h-full w-0 bg-white/10 transition-all duration-300 ease-out group-hover:w-full"></span>
                    <span className="relative flex items-center gap-2">
                      Start Demo <ChevronsRight size={22} />
                    </span>
                </button>
            </header>

            <div className="grid md:grid-cols-3 gap-6 mt-20 w-full max-w-5xl">
                <FeatureCard 
                    icon={<Mic size={24} className="text-blue-400"/>} 
                    title="Transcription & VAD" 
                    description="Utilizes Fast-Whisper for accurate transcription and Silero-VAD to detect and mark speech gaps." 
                    delay={0.2}
                />
                <FeatureCard 
                    icon={<RotateCw size={24} className="text-blue-400"/>} 
                    title="Sentence Reconstruction" 
                    description="Employs a T5 model to intelligently fill in the detected gaps, creating complete and contextually relevant sentences." 
                    delay={0.3}
                />
                <FeatureCard 
                    icon={<CassetteTape size={24} className="text-blue-400"/>} 
                    title="Voice Cloning & Synthesis" 
                    description="Generates a new, high-quality audio file using E2 TTS, cloning the original voice to speak the reconstructed sentence." 
                    delay={0.4}
                />
            </div>
        </div>
    );
};

export default LandingPage;
