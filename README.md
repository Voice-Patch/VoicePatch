
# VoicePatch

VoicePatch is an end-to-end speech inpainting pipeline that leverages voice activity detection, transformer-based language modeling, and voice-cloned TTS to restore semantically coherent segments in degraded audio.

## Components Used

- Pre-Processing :  [FFmpeg](https://ffmpeg.org/download.html)
- Voice Activity Detection (VAD) : [SileroVAD (high ROC-AUC for segmentation)](https://github.com/snakers4/silero-vad)
- Transcription : [OpenAI-Whisper-turbo](https://huggingface.co/openai/whisper-large-v3-turbo)
- Contextual Prediction: [T5-base](https://huggingface.co/google-t5/t5-base)
- Speech Synthesis (Maintain user voice) : [E2-TTS-base](https://huggingface.co/SWivid/E2-TTS)

**For more information regarding performance metrcis, research and evaluations refer to the [report](./voicepatchreport.pdf)**

## Usage Instructions
Download this repository
```
git clone https://github.com/Voice-Patch/VoicePatch.git
```
### Server Setup
```
cd backend
```
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
```

**Make sure ffmpeg is installed**

```
uvicorn main:app --reload --port 8000
```

### Frontend

```
cd frontend
```
```
npm install

npm run dev
```
**The frontend application should now be running at**
`http://localhost:5173`



## Workflow

![flowhcart](backend/readme_images/flowchart.png)

## Gallery
![home](backend/readme_images/home.png)
![file](backend/readme_images/file.png)
![trans](backend/readme_images/trans.png)
![recons](backend/readme_images/recons.png)

## Authors
- [Ashish Adhikari](https://github.com/Aashish1-1-1)  
- [Swastik Aryal](https://github.com/Swastik-Aryal)  
- [James Bhattarai](https://github.com/jamesii-b)  
- [Rikesh Panta](https://github.com/RiCEmare)
- [Kritan Rijal](https://github.com/justfoolingaround)

