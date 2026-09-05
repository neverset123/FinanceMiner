## FinanceMiner
1. https://github.com/minihellboy/factorminer.git
2. https://kalshi.com/
3. https://polymarket.com/
4. https://github.com/TraderAlice/OpenAlice
5. https://github.com/PandaAI-Tech/panda_factor

### Video Generation Pipeline

```bash
cd pipeline/vlog
export NODE_TLS_REJECT_UNAUTHORIZED=0
python make_video.py --script script.txt --out output.mp4
```

### Broker Agent

Ingests YouTube channel transcripts into TeleMem semantic memory for queryable retrieval. Enumerates a channel's videos, downloads captions, and stores them for LLM-grounded semantic search with optional fact extraction via LLM inference.
The broker downloads captions as VTT/SRT format, parses them into plain text, then stores them in TeleMem—which chunks the text, embeds it with an embedder model, and saves the vectors in FAISS for semantic search.

