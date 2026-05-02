-- KrishiSaathi: apply in Supabase SQL Editor or link this repo CI to your project.
-- Enable extension: Dashboard → Database → Extensions → "vector".

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS public.farmer_twin (
  farmer_id UUID PRIMARY KEY REFERENCES auth.users (id) ON DELETE CASCADE,
  payload JSONB NOT NULL,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS public.conversation_metadata (
  conversation_id TEXT PRIMARY KEY,
  farmer_id TEXT NOT NULL,
  title TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS conv_meta_farmer_idx
  ON public.conversation_metadata (farmer_id, created_at DESC);

CREATE TABLE IF NOT EXISTS public.query_history (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  conversation_id TEXT REFERENCES public.conversation_metadata (conversation_id) ON DELETE SET NULL,
  query_text TEXT,
  intent TEXT,
  response TEXT,
  "timestamp" TIMESTAMPTZ NOT NULL DEFAULT now(),
  data_source TEXT
);

CREATE INDEX IF NOT EXISTS query_history_conv_ts
  ON public.query_history (conversation_id, "timestamp" DESC);

CREATE TABLE IF NOT EXISTS public.scheme_vectors (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  scheme_id TEXT NOT NULL UNIQUE,
  content TEXT NOT NULL,
  scheme_json JSONB NOT NULL,
  embedding vector(384) NOT NULL
);

CREATE INDEX IF NOT EXISTS scheme_vectors_embedding_idx ON public.scheme_vectors
  USING hnsw (embedding vector_cosine_ops);

CREATE OR REPLACE FUNCTION public.match_scheme_vectors(
  query_embedding vector(384),
  match_count int DEFAULT 5
)
RETURNS TABLE (
  id uuid,
  scheme_id text,
  scheme_json jsonb,
  similarity float
)
LANGUAGE sql
STABLE
AS $$
  SELECT
    s.id,
    s.scheme_id,
    s.scheme_json,
    (1 - (s.embedding <=> query_embedding))::float AS similarity
  FROM public.scheme_vectors s
  ORDER BY s.embedding <=> query_embedding
  LIMIT match_count;
$$;

COMMENT ON FUNCTION public.match_scheme_vectors IS 'Cosine similarity search for scheme embeddings (aligned with Chroma DefaultEmbeddingFunction).';

-- Migrating older projects: manually ALTER query_history DROP farmer_id etc. if upgrading from legacy schema.
