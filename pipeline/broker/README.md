## broker
py311 is required

### ingest youtube channel/playlist
```
export WHISPER_MODEL_PATH=~/Workspace/whisper-medium/
export EMBEDDING_MODEL=qgenie_embedd
HF_HUB_OFFLINE=1 python pipeline/broker/ingest_channel.py --channel https://www.youtube.com/@bellafinance  --limit 6 --user-id bellafinance --no-infer --whisper-model medium
HF_HUB_OFFLINE=1 python pipeline/broker/ingest_channel.py --playlist https://www.youtube.com/playlist?list=PLangBM27OtEA --limit 10 --user-id standfordRSI --no-infer --whisper-model medium
python pipeline/broker/query_memory.py --user-id bellafinance --query "inflation outlook" --as-context
python pipeline/broker/holding_recommendation.py  --output holding_rec.md
python pipeline/broker/broker_blog.py --user-id bellafinance -n 4 -o report.md  --raw
python pipeline/broker/broker_blog.py --user-id standfordRSI -n 200 -o report_rsi.md  --raw
```
### ingest podcast
´´´
export WHISPER_MODEL_PATH=~/Workspace/whisper-medium/
export EMBEDDING_MODEL=qgenie_embedd
HF_HUB_OFFLINE=1 python pipeline/broker/ingest_podcast.py --user-id crossing --no-infer
python pipeline/broker/broker_blog.py --faiss-json db/faiss_db_podcast/xiaoyuzhou.json -n 1 -o report_cross.md
python pipeline/broker/delete_podcast_episode.py --episode-id "6ae6e230c96c8942ec156ed6"
´´´

### sc
´´´
sc broker overview --json
sc broker holdings --json >> data/sc/json/holdings.json
sc broker chart --isin US67066G1040 --timeframe max --json >> data/sc/json/nv_chart.json
sc broker quote --isin DE0007100000 --json
sc broker security-news --isin US67066G1040 --json >> data/sc/json/nv_news.json
´´´

### t212
´´´
python pipeline/broker/t212.py >> data/t212/json/orders.json
´´´