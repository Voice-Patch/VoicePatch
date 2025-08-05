import React, { useState, useEffect, useCallback, useRef } from 'react';
import { CheckCircle } from 'lucide-react';
import WaveformVisualizer from './WaveformVisualizer';

// --- Helper Components for Animations ---

const LiveTranscription = ({ text, startAnimation, onAnimationComplete, setHoveredGapIndex }) => {
    const [visibleWordCount, setVisibleWordCount] = useState(0);
    let maskCounter = -1;

    useEffect(() => {
        if (startAnimation) {
            const words = text.split(' ');
            const BASE_WORD_DELAY = 200;
            const PER_CHAR_DELAY = 50;
            let cumulativeDelay = 0;

            const timeouts = words.map((word, index) => {
                const delay = cumulativeDelay;
                cumulativeDelay += BASE_WORD_DELAY + (word.length * PER_CHAR_DELAY);
                return setTimeout(() => {
                    setVisibleWordCount(index + 1);
                }, delay);
            });

            // Set a final timeout to signal completion
            const finalTimeout = setTimeout(() => {
                if (onAnimationComplete) onAnimationComplete();
            }, cumulativeDelay + 500); // Add a small buffer

            return () => {
                timeouts.forEach(clearTimeout);
                clearTimeout(finalTimeout);
            };
        }
    }, [startAnimation, text, onAnimationComplete]);

    const visibleWords = text.split(' ').slice(0, visibleWordCount);

    return (
        <p className="text-lg font-mono text-slate-300">
            {visibleWords.map((word, index) => {
                if (word.includes('[MASK]')) {
                    maskCounter++;
                    const currentMaskIndex = maskCounter;
                    return (
                        <span 
                            key={index} 
                            className="text-red-400 font-bold cursor-pointer"
                            onMouseEnter={() => setHoveredGapIndex(currentMaskIndex)}
                            onMouseLeave={() => setHoveredGapIndex(null)}
                        >
                            {word}{' '}
                        </span>
                    );
                }
                return <span key={index}>{word}{' '}</span>;
            })}
        </p>
    );
};


const ReconstructionText = ({ originalText, finalText, startAnimation, onAnimationComplete, setHoveredGapIndex }) => {
    const [displayText, setDisplayText] = useState(originalText.split(' '));
    const [wordStates, setWordStates] = useState({});
    const alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';

const scrambleAndReveal = useCallback(async () => {
    // Helper function to pre-calculate the correct multi-word replacements for each [MASK]
    const getTargets = (original, final) => {
        const originalWords = original.split(' ');
        const finalWords = final.split(' ');
        const maskData = [];

        let finalPtr = 0;
        for (let i = 0; i < originalWords.length; i++) {
            if (originalWords[i].includes('[MASK]')) {
                const nextAnchorWord = originalWords.slice(i + 1).find(w => !w.includes('[MASK]'));
                let replacementWords = [];
                
                if (nextAnchorWord) {
                    const anchorIndexInFinal = finalWords.indexOf(nextAnchorWord, finalPtr);
                    if (anchorIndexInFinal !== -1) {
                        replacementWords = finalWords.slice(finalPtr, anchorIndexInFinal);
                        finalPtr = anchorIndexInFinal;
                    }
                } else {
                    // This handles the case where [MASK] is at the end of the text
                    replacementWords = finalWords.slice(finalPtr);
                }

                maskData.push({
                    index: i,
                    text: replacementWords.join(' ')
                });
            } else {
                // Move the final pointer forward to maintain alignment
                const newFinalPtr = finalWords.indexOf(originalWords[i], finalPtr);
                finalPtr = (newFinalPtr !== -1 ? newFinalPtr : finalPtr) + 1;
            }
        }
        return maskData;
    };

    const targets = getTargets(originalText, finalText);
    const initialDisplayText = originalText.split(' ');

    for (const targetInfo of targets) {
        const { index, text: targetWord } = targetInfo;

        if (!targetWord) continue;

        setWordStates(prev => ({ ...prev, [index]: 'processing' }));
        await new Promise(resolve => setTimeout(resolve, 100));

        let currentWord = Array(targetWord.length).fill(null);
        for (let i = 0; i < targetWord.length; i++) {
            // Scramble effect
            for (let j = 0; j < 8; j++) {
                currentWord[i] = alphabet[Math.floor(Math.random() * alphabet.length)];
                setDisplayText(prev => {
                    const newText = [...(Array.isArray(prev) ? prev : initialDisplayText)];
                    newText[index] = currentWord.join('');
                    return newText;
                });
                await new Promise(resolve => setTimeout(resolve, 25));
            }
            // Reveal the correct character
            currentWord[i] = targetWord[i];
             setDisplayText(prev => {
                const newText = [...(Array.isArray(prev) ? prev : initialDisplayText)];
                newText[index] = currentWord.join('');
                return newText;
            });
        }
        setWordStates(prev => ({ ...prev, [index]: 'done' }));
    }

    if (onAnimationComplete) onAnimationComplete();
}, [finalText, alphabet, originalText, onAnimationComplete]);

    useEffect(() => {
        if (startAnimation) {
            const timer = setTimeout(() => scrambleAndReveal(), 1000);
            return () => clearTimeout(timer);
        } else {
             setDisplayText(originalText.split(' '));
             setWordStates({});
        }
    }, [startAnimation, scrambleAndReveal, originalText]);
    
    let maskCounterRec = -1;

    return (
        <p className="text-lg font-mono text-slate-300">
            {displayText.map((word, i) => {
                const originalWord = originalText.split(' ')[i];
                const isMasked = originalWord && originalWord.includes('[MASK]');
                if(isMasked) maskCounterRec++;
                const currentMaskIndexRec = maskCounterRec;

                const wordState = wordStates[i];
                const colorClass = isMasked
                    ? (wordState === 'processing' || wordState === 'done' ? 'text-teal-300 font-bold' : 'text-red-400 font-bold')
                    : '';

                return (
                    <span 
                        key={i} 
                        className={`whitespace-pre-wrap ${colorClass} ${isMasked ? 'cursor-pointer' : ''}`}
                        onMouseEnter={() => isMasked && setHoveredGapIndex(currentMaskIndexRec)}
                        onMouseLeave={() => isMasked && setHoveredGapIndex(null)}
                    >
                        {word.split('').map((char, charIndex) => <span key={charIndex} className="char-slot">{char}</span>)}
                        {' '}
                    </span>
                );
            })}
        </p>
    );
};


const Step = ({ title, description, children }) => (
    <div>
        <h3 className="flex items-center gap-3 text-2xl font-bold mb-2">
           <CheckCircle size={28} className="text-teal-400"/>
           {title}
        </h3>
        <p className="pl-11 mb-4 text-slate-400 -mt-2">{description}</p>
        <div className="pl-11">
            {children}
        </div>
    </div>
);


const ProcessingFlow = ({ results, processingState, setHoveredGapIndex }) => {
    // State to track visibility of each block
    const [showTranscription, setShowTranscription] = useState(false);
    const [showReconstruction, setShowReconstruction] = useState(false);
    const [showSynthesis, setShowSynthesis] = useState(false);

    // State to track animation completion
    const [transcriptionAnimationDone, setTranscriptionAnimationDone] = useState(false);
    const [reconstructionAnimationDone, setReconstructionAnimationDone] = useState(false);

    // Effect to show the first block as soon as its data is available
    useEffect(() => {
        if (results?.transcript) {
            setShowTranscription(true);
        }
    }, [results?.transcript]);

    // Effect to show the second block ONLY after the first animation is done AND its data is available
    useEffect(() => {
        if (transcriptionAnimationDone && results?.fullReconstructedText) {
            setShowReconstruction(true);
        }
    }, [transcriptionAnimationDone, results?.fullReconstructedText]);

    // Effect to show the third block ONLY after the second animation is done AND its data is available
    useEffect(() => {
        if (reconstructionAnimationDone && results?.synthesisUrl) {
            setShowSynthesis(true);
        }
    }, [reconstructionAnimationDone, results?.synthesisUrl]);

    // Callbacks to be passed to children to signal animation completion.
    // useCallback with empty deps ensures these functions are stable and don't cause re-renders.
    const handleTranscriptionComplete = useCallback(() => {
        setTranscriptionAnimationDone(true);
    }, []);

    const handleReconstructionComplete = useCallback(() => {
        setReconstructionAnimationDone(true);
    }, []);


    if (processingState === 'processing' && !results) {
        return <div className="text-center p-10 text-xl animate-fade-in">Processing audio...</div>;
    }

    if (!results) return null;

    return (
        <div className="space-y-12">
            <div className={`transition-all duration-700 ease-in-out overflow-hidden ${showTranscription ? 'max-h-[500px] opacity-100' : 'max-h-0 opacity-0'}`}>
                <Step title="Transcription" description="First, we convert your audio into text and use our AI to pinpoint silent gaps.">
                        <div className="glass-card p-6 text-lg font-mono text-slate-300">
                            {results.transcript && (
                                <LiveTranscription 
                                    text={results.transcript}
                                    startAnimation={showTranscription}
                                    onAnimationComplete={handleTranscriptionComplete}
                                    setHoveredGapIndex={setHoveredGapIndex}
                                />
                            )}
                        </div>
                </Step>
            </div>

            <div className={`transition-all duration-700 ease-in-out overflow-hidden ${showReconstruction ? 'max-h-[500px] opacity-100' : 'max-h-0 opacity-0'}`}>
                <Step title="Reconstruction" description="Next, our T5 model analyzes the context and intelligently fills in the missing words.">
                    <div className="glass-card p-6">
                        {results.fullReconstructedText && (
                             <ReconstructionText 
                                originalText={results.transcript}
                                finalText={results.fullReconstructedText}
                                startAnimation={showReconstruction}
                                onAnimationComplete={handleReconstructionComplete}
                                setHoveredGapIndex={setHoveredGapIndex}
                            />
                        )}
                    </div>
                </Step>
            </div>

            <div className={`transition-all duration-700 ease-in-out overflow-hidden ${showSynthesis ? 'max-h-[500px] opacity-100' : 'max-h-0 opacity-0'}`}>
                 <Step title="Synthesis" description="Finally, we generate a new, seamless audio file using a clone of the original voice.">
                    <div className="glass-card p-6">
                        {results.synthesisUrl && (
                            <WaveformVisualizer 
                                audioUrl={results.synthesisUrl}
                                waveColor="#475569"
                                progressColor="#2DD4BF"
                                isPlayer={true}
                            />
                        )}
                    </div>
                </Step>
            </div>
        </div>
    );
};

export default ProcessingFlow;
