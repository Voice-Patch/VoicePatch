import React, { useRef, useEffect, useState } from 'react';
import { Play, Pause, Volume2, Volume1, VolumeX, Rewind } from 'lucide-react';

// Helper function to format time from seconds to MM:SS
const formatTime = (seconds) => {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    return `${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`;
};

const WaveformVisualizer = ({ audioUrl, waveColor = '#64748B', progressColor = '#3B82F6', isPlayer = false, vadGaps = [], hoveredGapIndex = null }) => {
    const waveformRef = useRef(null);
    const wavesurfer = useRef(null);
    const [isPlaying, setIsPlaying] = useState(false);
    const [duration, setDuration] = useState(0);
    const [currentTime, setCurrentTime] = useState(0);
    const [volume, setVolume] = useState(0.75);

    useEffect(() => {
        if (!waveformRef.current || !window.WaveSurfer) return;

        wavesurfer.current = window.WaveSurfer.create({
            container: waveformRef.current,
            waveColor: waveColor,
            progressColor: progressColor,
            cursorColor: 'transparent',
            barWidth: 3,
            barRadius: 3,
            responsive: true,
            height: isPlayer ? 80 : 128,
            hideScrollbar: true,
        });

        if (audioUrl) wavesurfer.current.load(audioUrl);

        wavesurfer.current.on('ready', () => {
            setDuration(wavesurfer.current.getDuration());
            wavesurfer.current.setVolume(volume);
        });
        wavesurfer.current.on('audioprocess', (time) => setCurrentTime(time));
        wavesurfer.current.on('play', () => setIsPlaying(true));
        wavesurfer.current.on('pause', () => setIsPlaying(false));
        wavesurfer.current.on('finish', () => {
            setIsPlaying(false);
            wavesurfer.current.seekTo(0);
            setCurrentTime(0);
        });

        return () => wavesurfer.current.destroy();
    }, [audioUrl, waveColor, progressColor, isPlayer]);

    const handlePlayPause = () => wavesurfer.current.playPause();
    const handleRewind = () => wavesurfer.current.seekTo(0);
    const handleVolumeChange = (e) => {
        const newVolume = parseFloat(e.target.value);
        setVolume(newVolume);
        wavesurfer.current.setVolume(newVolume);
    };
    
    const VolumeIcon = volume === 0 ? VolumeX : volume < 0.5 ? Volume1 : Volume2;

    return (
        <div className="w-full">
            <div className="w-full relative">
                <div ref={waveformRef}></div>
                {duration > 0 && vadGaps && vadGaps.map((gap, i) => {
                    const left = (gap.start / duration) * 100;
                    const width = ((gap.end - gap.start) / duration) * 100;
                    const isHovered = i === hoveredGapIndex;
                    return (
                        <div 
                            key={i}
                            className={`absolute top-0 h-full rounded pointer-events-none transition-all duration-200 ${
                                isHovered ? 'bg-yellow-500/50 backdrop-blur-sm' : 'bg-red-500/30'
                            }`}
                            style={{ left: `${left}%`, width: `${width}%` }}
                        ></div>
                    );
                })}
            </div>

            {isPlayer && (
                 <div className="flex items-center justify-between mt-3">
                    <div className="flex items-center gap-4">
                        <button onClick={handleRewind} className="p-2 text-slate-300 hover:text-white transition-colors">
                            <Rewind size={20} />
                        </button>
                        <button onClick={handlePlayPause} className="p-3 bg-white/10 hover:bg-white/20 rounded-full text-white transition-colors">
                            {isPlaying ? <Pause size={20} /> : <Play size={20} />}
                        </button>
                    </div>

                    <div className="text-sm font-mono text-slate-400">
                        {formatTime(currentTime)} / {formatTime(duration)}
                    </div>

                    <div className="flex items-center gap-2 w-28">
                         <VolumeIcon size={20} className="text-slate-400" />
                         <input
                            type="range"
                            min="0"
                            max="1"
                            step="0.05"
                            value={volume}
                            onChange={handleVolumeChange}
                            className="w-full h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-teal-400 focus:outline-none"
                        />
                    </div>
                </div>
            )}
        </div>
    );
};

export default WaveformVisualizer;
