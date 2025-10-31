document.addEventListener('DOMContentLoaded', () => {
    // Get audio and canvas elements
    const originalAudio = document.getElementById('original-audio');
    const enhancedAudio = document.getElementById('enhanced-audio');
    const originalCanvas = document.getElementById('original-visualizer');
    const enhancedCanvas = document.getElementById('enhanced-visualizer');

    // Create a single AudioContext for all audio elements on the page
    const audioContext = new AudioContext();
    
    // Create and initialize the visualizers
    initVisualizer(originalAudio, originalCanvas, audioContext);
    initVisualizer(enhancedAudio, enhancedCanvas, audioContext);
});

function initVisualizer(audioElement, canvasElement, audioContext) {
    if (!audioElement || !canvasElement) {
        console.error("Audio or canvas element not found.");
        return;
    }

    let analyser;
    let source;
    let animationId; // Use an ID to control the animation frame

    // This is the CRITICAL fix: Resume the AudioContext on any user interaction.
    // This addresses autoplay policies in modern browsers.
    document.addEventListener('click', () => {
        if (audioContext.state === 'suspended') {
            audioContext.resume().then(() => {
                console.log("AudioContext resumed successfully.");
            });
        }
    }, { once: true }); // The listener is removed after one click to prevent conflicts

    // Use the `canplay` event to ensure the audio source is ready
    audioElement.addEventListener('canplay', () => {
        // Prevent re-initialization
        if (source) return;

        analyser = audioContext.createAnalyser();
        source = audioContext.createMediaElementSource(audioElement);
        
        // Connect the source to the analyser and the analyser to the audio output
        source.connect(analyser);
        analyser.connect(audioContext.destination);

        analyser.fftSize = 256;
        const bufferLength = analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);

        const canvasCtx = canvasElement.getContext('2d');
        
        canvasElement.width = 300; 
        canvasElement.height = 100;
        
        const canvasWidth = canvasElement.width;
        const canvasHeight = canvasElement.height;

        // The draw loop for the visualizer
        function draw() {
            animationId = requestAnimationFrame(draw);
            analyser.getByteTimeDomainData(dataArray);

            canvasCtx.fillStyle = 'rgba(240, 244, 248, 1)';
            canvasCtx.clearRect(0, 0, canvasWidth, canvasHeight);

            canvasCtx.lineWidth = 2;
            canvasCtx.strokeStyle = 'rgb(93, 93, 255)';
            canvasCtx.beginPath();

            const sliceWidth = canvasWidth * 1.0 / bufferLength;
            let x = 0;

            for (let i = 0; i < bufferLength; i++) {
                const v = dataArray[i] / 128.0;
                const y = v * canvasHeight / 2;

                if (i === 0) {
                    canvasCtx.moveTo(x, y);
                } else {
                    canvasCtx.lineTo(x, y);
                }
                x += sliceWidth;
            }

            canvasCtx.lineTo(canvasWidth, canvasHeight / 2);
            canvasCtx.stroke();
        }

        // Start drawing when the audio is played
        audioElement.addEventListener('play', () => {
            draw();
        });

        audioElement.addEventListener('pause', () => {
            cancelAnimationFrame(animationId); 
        });
    });
}