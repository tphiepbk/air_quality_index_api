import numpy as np

def attach_station_embedding(df, lookup, station_col="station_id", prefix="station_emb"):
    df = df.copy()
    # Mean embedding used for unknown station
    mean_emb = np.mean(np.stack(list(lookup.values())), axis=0)

    emb_cols = [f"{prefix}_{i}" for i in range(len(mean_emb))]
    # Create embedding matrix for each row
    embs = []
    for sid in df[station_col].astype(int).values:
        embs.append(lookup.get(sid, mean_emb))
    embs = np.vstack(embs)
    for j, col in enumerate(emb_cols):
        df[col] = embs[:, j]
    return df, emb_cols

