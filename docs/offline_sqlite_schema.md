# Expo / React Native offline SQLite schema (KrishiSaathi companion)

Recommended local database when the farmer is offline. Sync from:
`GET /api/v1/sync/bundle` (gzip JSON) plus last conversation turns mirrored from Redis when online.

Tables follow the product plan (messages, cached_tools, farmer_profile).

## DDL

```sql
CREATE TABLE IF NOT EXISTS farmer_profile (
    user_id TEXT PRIMARY KEY,
    profile_json TEXT NOT NULL,
    last_synced INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    role TEXT NOT NULL,
    text TEXT NOT NULL,
    timestamp INTEGER NOT NULL,
    conversation_id TEXT
);

CREATE TABLE IF NOT EXISTS cached_tools (
    key TEXT PRIMARY KEY,
    json TEXT NOT NULL,
    last_updated INTEGER NOT NULL
);
```

### `farmer_profile`

- **`user_id`**: Stable Supabase farmer UUID (`farmer_id` from APIs).
- **`profile_json`**: JSON string mirroring `/api/v1/farmer/{id}/twin` payload (`FarmerTwin`).

### `messages`

Rolling window (e.g. last 40 rows per `conversation_id`); trim on insert after sync.

- **`role`**: `user` | `assistant`.
- **`text`**: User query or truncated assistant reply.
- **`timestamp`**: Unix seconds.

### `cached_tools`

Key/value cache for stale-friendly tool JSON.

Suggested keys:

| key | Contents |
| --- | -------- |
| `weather_today` | Last Open-Meteo widget JSON for home card |
| `mandi:{district}:{crop}` | Mandi aggregator row / tool output |
| `schemes:index` | Subset from bundle `scheme_index` ids |
| `twin_snapshot` | Copy of farmer_profile row for Gemma prompts |

Bundle hydration: after unpacking `bundle.data`, populate `cached_tools` for `mandi_prices`,
`crop_calendar`, `weather_history`, and optionally write scheme titles into `messages`/`cached_tools`.

## Connectivity

Use `@react-native-community/netinfo` (or Expo equivalent). When `isConnected`:
1. Refresh token if needed (`/api/v1/auth/...`).
2. `GET /api/v1/sync/bundle?state=...&district=...&farmer_id=...&conversation_id=...` — updates Redis-backed caches server-side when configured and returns offline bundle payload.
3. Merge into SQLite transactions (profile + caches + optionally messages).
