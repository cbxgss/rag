export CUDA_VISIBLE_DEVICES=0
uvicorn nli_small:app --host localhost --port 8004 --reload
