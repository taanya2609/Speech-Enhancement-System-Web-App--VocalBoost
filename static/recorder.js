document.addEventListener('DOMContentLoaded', () => {
    const micButton = document.getElementById('mic-button');
    const audioInput = document.getElementById('audio-file-input');
    const statusMessage = document.getElementById('status-message');
    const loadingSpinner = document.getElementById('loading-spinner');
    const enhancementLevelToggle = document.getElementById('enhancement-level-toggle');
    const progressBar = document.getElementById('progress-container');
    const progressFill = document.getElementById('progress-fill');
    const progressText = document.getElementById('progress-text');

    let mediaRecorder;
    let audioChunks = [];
    let isRecording = false;

    // Handle recording button click
    micButton.addEventListener('click', async () => {
        if (!isRecording) {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' }); 
                audioChunks = [];

                mediaRecorder.ondataavailable = event => {
                    audioChunks.push(event.data);
                };

                mediaRecorder.onstop = () => {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                    uploadAudio(audioBlob);
                };

                mediaRecorder.start();
                isRecording = true;
                micButton.innerHTML = '<i class="fas fa-stop-circle"></i> Stop Recording';
                statusMessage.textContent = 'Recording...';
            } catch (err) {
                statusMessage.textContent = 'Error: ' + err.message;
            }
        } else {
            mediaRecorder.stop();
            isRecording = false;
            micButton.innerHTML = '<i class="fas fa-microphone"></i> Start Recording';
            statusMessage.textContent = 'Recording stopped. Processing...';
        }
    });

    // Handle file input change
    audioInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            statusMessage.textContent = 'File selected. Processing...';
            uploadAudio(file);
        }
    });

    // Function to upload audio blob or file to the server
    function uploadAudio(fileBlob) {
        const formData = new FormData();
        formData.append('audio', fileBlob, 'recording.webm');
        formData.append('enhancement_level', enhancementLevelToggle.value);

        // Show progress bar and hide spinner
        progressBar.style.display = 'flex';
        loadingSpinner.style.display = 'none';
        
        // Simulate progress for a better user experience
        let progress = 0;
        const progressInterval = setInterval(() => {
            progress += 5;
            if (progress <= 95) {
                progressFill.style.width = `${progress}%`;
                progressText.textContent = `Processing... ${progress}%`;
            } else {
                clearInterval(progressInterval);
            }
        }, 500);

        fetch('/upload_audio', {
            method: 'POST',
            body: formData,
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            clearInterval(progressInterval);
            progressBar.style.display = 'none';
            if (data.success && data.file_id) {
                const confidence = data.confidence !== undefined ? data.confidence : 0;
                window.location.href = `/result/${data.file_id}?confidence=${confidence}`;
            } else {
                statusMessage.textContent = `Error: ${data.error || 'Server returned an invalid response.'}`;
            }
        })
        .catch(error => {
            clearInterval(progressInterval);
            progressBar.style.display = 'none';
            console.error('Error:', error);
            statusMessage.textContent = 'An error occurred during upload. Please try again.';
        });
    }
});