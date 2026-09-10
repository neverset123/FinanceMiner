## Video Generation Pipeline

```bash
cd pipeline/vlog
export NODE_TLS_REJECT_UNAUTHORIZED=0
python run_pipeline.py --script transcript.txt --out output.mp4
```

open issues:
  Summary:
  - The pipeline completed and wrote output.mp4 and output.srt
  - But the video has no voice narration (TTS broken)
  - And 31 out of 33 scenes have no background images (rate-limited by Commons)

