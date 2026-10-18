# Garage

Local site. Flask on `127.0.0.1:5000` owns Railway Postgres and Tigris.
Next.js on `127.0.0.1:3000` talks to Flask and does not open the database.

```bash
cp .env.example .env
cd api && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python app.py
```

```bash
cd web && npm install && npm run dev
```

Open `http://127.0.0.1:3000`. Bind stays on localhost. The tree is the
main view. It is built from the store TSV, later scored arms, and a poll
of wandb `f1tenth-genesis` plus local trainer dirs. Videos live on the
run page.

Race is the gigaflow viewer inside Garage. Pick more than one checkpoint
and they share one track, one car each. CNN-GRU, action chunk, TinyLidarNet,
LiDAR Dreamer, and the Gigaflow GRU / feedforward / frame-stack actors all
step in that same sim. Build the viewer once, and point the API at a track
cache (`gigaflow prepare-tracks`):

```bash
cd gigaflow/viewer && npm install && npm run build
```

Tigris keys belong in `dashboard/.env` only. A checkpoint that is only on disk
is copied into the bucket under `dashboard/` and then downloaded.
