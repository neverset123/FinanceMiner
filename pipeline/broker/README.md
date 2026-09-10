## broker
py311 is required

### ingest news
```
export WHISPER_MODEL_PATH=~/Workspace/whisper-medium/
export EMBEDDING_MODEL=qgenie_embedd
HF_HUB_OFFLINE=1 python pipeline/broker/ingest_channel.py --channel https://www.youtube.com/@bellafinance  --limit 2 --user-id bellafinance --no-infer --whisper-model medium
python pipeline/broker/query_memory.py --user-id bellafinance --query "inflation outlook" --as-context
python pipeline/broker/holding_recommendation.py  --output holding_rec.md
python pipeline/broker/broker_blog.py -n 2 -o report.md  --raw
```

### sc
´´´
sc broker overview --json
sc broker holdings --json >> data/sc/json/holdings.json
sc broker chart --isin US67066G1040 --timeframe max --json >> data/sc/json/nv_chart.json
sc broker quote --isin DE0007100000 --json
sc broker security-news --isin US67066G1040 --json >> data/sc/json/nv_news.json
´´´