import React, { useState, useCallback } from 'react';
import { UploadCloud, FileAudio } from 'lucide-react';

const AudioInput = ({ onAudioReady }) => {
    const fileInputRef = React.useRef(null);
    const [isDragging, setIsDragging] = useState(false);

    const handleFileChange = (e) => {
        const file = e.target.files[0];
        if (file) onAudioReady(file, file.name);
    };

    const handleExampleSelect = async () => {
        // This path should correctly point to your file in the `public` directory.
        const audioPath = '/output_muted_400.mp3';
        const fileName = 'output_muted_400.mp3';

        try {
            const res = await fetch(audioPath);
            if (!res.ok) {
                // This error will trigger if the file is not found (e.g., 404 error).
                throw new Error(`Could not fetch the audio file: ${res.statusText}`);
            }
            const blob = await res.blob();
            onAudioReady(blob, fileName);
        } catch (error) {
            console.error("Failed to load example audio:", error);
            alert(`Could not load the example audio. Please make sure the file "${fileName}" exists directly inside your project's "public" folder.`);
        }
    };

    const handleDragEnter = useCallback((e) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragging(true);
    }, []);

    const handleDragLeave = useCallback((e) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragging(false);
    }, []);

    const handleDragOver = useCallback((e) => {
        e.preventDefault();
        e.stopPropagation();
    }, []);

    const handleDrop = useCallback((e) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragging(false);
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            onAudioReady(e.dataTransfer.files[0], e.dataTransfer.files[0].name);
            e.dataTransfer.clearData();
        }
    }, [onAudioReady]);


    return (
        <div 
            className={`animate-fade-in-up w-full h-[60vh] max-w-3xl mx-auto flex flex-col items-center justify-center glass-card p-8 border-2 border-dashed transition-all duration-300 ${isDragging ? 'border-blue-400 bg-blue-500/10' : 'border-white/10'}`}
            onDragEnter={handleDragEnter}
            onDragLeave={handleDragLeave}
            onDragOver={handleDragOver}
            onDrop={handleDrop}
        >
            <UploadCloud size={64} className={`transition-transform duration-300 ${isDragging ? 'scale-110' : ''} text-slate-400 mb-4`} />
            <h2 className="text-2xl font-bold text-slate-100">Drag & drop your audio file</h2>
            <p className="text-slate-400 mb-6">or</p>
            <button
                onClick={() => fileInputRef.current.click()}
                className="bg-slate-700 hover:bg-slate-600 text-white font-semibold py-2 px-6 rounded-lg transition-colors"
            >
                Browse Files
            </button>
            <input type="file" ref={fileInputRef} onChange={handleFileChange} className="hidden" accept="audio/*"/>
            
            <div className="mt-12 text-center">
                <p className="text-slate-500 mb-2">Don't have an audio file?</p>
                       <button
                    onClick={handleExampleSelect}
                    className="flex items-center gap-2 text-blue-400 hover:text-blue-300 font-semibold transition-colors justify-center"
                    >
                    <FileAudio size={18} />
                        Use an Example
                    </button>
            </div>
        </div>
    );
};

export default AudioInput;
