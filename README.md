## Dataset Information
### Text8
Downloaded at http://mattmahoney.net/dc/text8.zip on 15/03/2026 10:58 PM, though it will not probably change

### Wikipedia Text Dump
https://dumps.wikimedia.org/other/mediawiki_content_current/enwiki/2026-03-01/xml/bzip2/enwiki-2026-03-01-p10p1147431.xml.bz2 on 15/03/2026 11:04 PM, though it will not probably change

```bash
python scripts/train.py --preproc_dir_name=cs0-mf5 --epoch=5 --batch_size=256 --initial_lr=0.025 --final_lr=1e-4 --num_neg_samples=5 --window_size=5 --embed_dim=100
```

```bash
python scripts/train.py --preproc_dir_name=cs500000-mf0 --epoch=50 --batch_size=512 --initial_lr=0.025 --final_lr=1e-4 --num_neg_samples=5 --window_size=5 --embed_dim=50
```

```bash
python scripts/evaluate.py --model_path="models/2026-03-21-00-57-29/model.npz" --word_map_path="models/2026-03-21-00-57-29/word_id_map.pkl"
```