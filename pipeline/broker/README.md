## broker
py311 is required

### ingest news
```
export WHISPER_MODEL_PATH=~/Workspace/whisper-medium/
export EMBEDDING_MODEL=qgenie_embedd
HF_HUB_OFFLINE=1 python pipeline/broker/ingest_channel.py --channel https://www.youtube.com/@bellafinance  --limit 20 --user-id bellafinance --no-infer --whisper-model medium
python pipeline/broker/query_memory.py --user-id bellafinance --query "inflation outlook" --as-context
python pipeline/broker/holding_recommendation.py  --output holding_rec.md
```

### sc
´´´
sc broker overview --json
sc broker holdings --json
sc broker chart --isin DE0007100000 --timeframe 1y --json
sc broker quote --isin DE0007100000 --json
´´´