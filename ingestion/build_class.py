from ingestion.build_index import main

main(db='vectordb_class', chunks_out='data/chunks_class.jsonl', shareable_only=True)