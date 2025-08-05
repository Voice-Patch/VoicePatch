import React, { useState, useEffect, useRef } from 'react';
import LandingPage from './pages/LandingPage';
import DemoPage from './pages/DemoPage';
import { CSSTransition, SwitchTransition } from 'react-transition-group';

const WEBSOCKET_URL = "ws://localhost:8000/ws/process";

function App() {
    const [page, setPage] = useState('landing');
    const [backendStatus, setBackendStatus] = useState('connecting'); // 'connecting', 'ready', 'error'
    const [wavesurferStatus, setWavesurferStatus] = useState('loading'); // 'loading', 'ready', 'error'
    
    const websocket = useRef(null);
    const nodeRef = useRef(null);
    // This ref prevents the useEffect from running its setup logic twice in StrictMode
    const effectRan = useRef(false);

    // Effect to manage WebSocket connection and script loading
    useEffect(() => {
        // In StrictMode, this will be false on the first run and true on the second.
        if (effectRan.current === true) {
            // Load WaveSurfer.js script with error handling
            const script = document.createElement('script');
            script.src = 'https://unpkg.com/wavesurfer.js';
            script.async = true;
            script.onload = () => {
                console.log("WaveSurfer.js script loaded successfully.");
                setWavesurferStatus('ready');
            };
            script.onerror = () => {
                console.error("Fatal Error: Could not load WaveSurfer.js script from its CDN.");
                setWavesurferStatus('error');
            };
            document.body.appendChild(script);

            // Establish WebSocket connection
            websocket.current = new WebSocket(WEBSOCKET_URL);

            websocket.current.onopen = () => {
                console.log("WebSocket connection established.");
            };

            websocket.current.onmessage = (event) => {
                const message = JSON.parse(event.data);
                if (message.step === 'engine_ready') {
                    console.log("Backend engine is ready.");
                    setBackendStatus('ready');
                }
            };
            
            websocket.current.onerror = (error) => {
                console.error("WebSocket error. Ensure your FastAPI backend server is running.", error);
                setBackendStatus('error');
            };
            
            websocket.current.onclose = () => {
                console.log("WebSocket connection closed.");
                if (backendStatus !== 'ready') {
                    setBackendStatus('error');
                }
            };
        }

        // The cleanup function will run on unmount
        return () => {
            effectRan.current = true; // Mark that the effect has run once
            if (websocket.current) {
                websocket.current.close();
            }
        };
    }, []); // Empty dependency array ensures this runs only once on mount/unmount cycle

    const isEngineReady = backendStatus === 'ready' && wavesurferStatus === 'ready';

    const getEngineStatusMessage = () => {
        if (wavesurferStatus === 'error') return 'Error: Failed to load audio library';
        if (backendStatus === 'error') return 'Error: Cannot connect to backend';
        if (isEngineReady) return 'Audio Engine Ready';
        return 'Loading Audio Engine...';
    };

    const navigateToDemo = () => {
        if (isEngineReady) {
            setPage('demo');
        }
    };

    const currentPageComponent = page === 'landing' 
        ? <LandingPage onStartDemo={navigateToDemo} isEngineReady={isEngineReady} engineStatusMessage={getEngineStatusMessage()} />
        : <DemoPage onGoBack={() => setPage('landing')} websocket={websocket.current} />;

    return (
        <div className="relative min-h-screen w-full overflow-hidden bg-[#0A0A0A]">
            <div className="absolute top-0 left-0 w-full h-full z-0 overflow-hidden pointer-events-none">
                 <div className="animated-blob-1 absolute top-[-30%] left-[-15%] w-[600px] h-[600px] bg-blue-900/50 rounded-full blur-3xl opacity-60"></div>
                 <div className="animated-blob-2 absolute bottom-[-30%] right-[-15%] w-[700px] h-[700px] bg-teal-900/50 rounded-full blur-3xl opacity-50"></div>
            </div>

            <main className="relative z-10 w-full max-w-7xl mx-auto p-4 sm:p-6 lg:p-8">
                <SwitchTransition mode="out-in">
                    <CSSTransition
                        key={page}
                        nodeRef={nodeRef}
                        timeout={500}
                        classNames="demo-transition"
                    >
                        <div ref={nodeRef}>
                            {currentPageComponent}
                        </div>
                    </CSSTransition>
                </SwitchTransition>
            </main>
        </div>
    );
}

export default App;
