import { useState, useCallback, useEffect } from 'react';
import { RotateCw, ArrowLeft } from 'lucide-react';
import AudioInput from '../components/AudioInput';
import AnalysisSetup from '../components/AnalysisSetup';
import ProcessingFlow from '../components/ProcessingFlow';

const fileToBase64 = (file) => new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = () => resolve(reader.result);
    reader.onerror = error => reject(error);
});

const DemoPage = ({ onGoBack, websocket }) => {
    const [audioSource, setAudioSource] = useState(null);
    const [processingState, setProcessingState] = useState('idle');
    const [results, setResults] = useState(null);
    const [vadParams, setVadParams] = useState({
        threshold: 0.4,
        minSpeechMs: 250,
        minSilenceMs: 100,
    });
    const [hoveredGapIndex, setHoveredGapIndex] = useState(null);

    const resetState = useCallback(() => {
        setAudioSource(null);
        setProcessingState('idle');
        setResults(null);
        setHoveredGapIndex(null);
    }, []);

    const handleAudioReady = useCallback((file, fileName) => {
        const source = { file, name: fileName, url: URL.createObjectURL(file) };
        setAudioSource(source);
        setProcessingState('ready');
    }, []);

    useEffect(() => {
        if (!websocket) return;

        const handleMessage = (event) => {
            const message = JSON.parse(event.data);
            
            if(message.error || message.step === 'engine_ready') return;

            console.log("Received message:", message);

            // Handle each step from the backend individually
            if (message.step === 'reconstruction') {
                setResults(prev => ({ 
                    ...prev, 
                    fullReconstructedText: message.data.full_reconstructed_text 
                }));
            } else if (message.step === 'synthesis') {
                 const backendUrl = "http://localhost:8000";
                 const fullAudioUrl = `${backendUrl}${message.data.audio_url}`;
                 setResults(prev => ({ ...prev, synthesisUrl: fullAudioUrl, synthesisFilename: message.data.synthesisFilename }));
                 setProcessingState('success');
            } else { // Handles 'transcription' and any other steps
                 setResults(prev => ({ ...prev, ...message.data }));
            }
        };

        websocket.onmessage = handleMessage;

        return () => {
            if (websocket) {
                websocket.onmessage = null;
            }
        };
    }, [websocket]);


    const handleStartProcessing = async () => {
        if (!audioSource || !websocket) return;

        setProcessingState('processing');
        setResults({}); // Start with an empty results object

        const audioData = await fileToBase64(audioSource.file);

        websocket.send(JSON.stringify({
            audio_data: audioData,
            file_name: audioSource.name,
            vad_params: vadParams
        }));
    };

    return (
        <div className="min-h-[90vh] flex flex-col">
            <header className="flex items-center justify-between mb-8 animate-fade-in">
                <button onClick={onGoBack} className="flex items-center gap-2 text-slate-300 hover:text-white transition-colors">
                    <ArrowLeft size={20} /> Back
                </button>
                <h1 className="text-3xl font-bold gradient-text">Audio Processing Pipeline</h1>
                <button onClick={resetState} disabled={processingState === 'idle'} className="flex items-center gap-2 bg-white/5 hover:bg-white/10 disabled:opacity-50 disabled:cursor-not-allowed text-slate-300 font-semibold py-2 px-4 rounded-lg transition">
                    <RotateCw size={16} /> New Audio
                </button>
            </header>

            <div className="flex-grow">
                {!audioSource ? (
                    <AudioInput onAudioReady={handleAudioReady} />
                ) : (
                    <div className="space-y-12">
                        <AnalysisSetup 
                            audioSource={audioSource}
                            onStartProcessing={handleStartProcessing}
                            vadParams={vadParams}
                            setVadParams={setVadParams}
                            isProcessing={processingState === 'processing' || processingState === 'success'}
                            vadGaps={results?.vadGaps}
                            hoveredGapIndex={hoveredGapIndex}
                        />
                        
                        {(processingState === 'processing' || processingState === 'success') && (
                             <ProcessingFlow
                                key={audioSource?.name}
                                results={results}
                                processingState={processingState}
                                setHoveredGapIndex={setHoveredGapIndex}
                            />
                        )}
                    </div>
                )}
            </div>
        </div>
    );
};

export default DemoPage;
