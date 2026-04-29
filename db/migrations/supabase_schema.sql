-- KrishiSaathi: apply in Supabase SQL Editor or link this repo CI to your project.
-- Enable extension: Dashboard → Database → Extensions → "vector".

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS public.farmer_twin (
  farmer_id UUID PRIMARY KEY REFERENCES auth.users (id) ON DELETE CASCADE,
  payload JSONB NOT NULL,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS public.query_history (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  farmer_id UUID NOT NULL REFERENCES auth.users (id) ON DELETE CASCADE,
  query_text TEXT,
  intent TEXT,
  response TEXT,
  "timestamp" TIMESTAMPTZ NOT NULL DEFAULT now(),
  data_source TEXT
);

CREATE INDEX IF NOT EXISTS query_history_farmer_ts ON public.query_history (farmer_id, "timestamp" DESC);

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

-- Prefer server-side reads/writes using SUPABASE_SERVICE_ROLE_KEY (RLS bypassed for service_role).
ALTER TABLE public.farmer_twin ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.query_history ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.scheme_vectors ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users manage own twin" ON public.farmer_twin
  FOR ALL TO authenticated USING (auth.uid() = farmer_id) WITH CHECK (auth.uid() = farmer_id);

CREATE POLICY "Users manage own queries" ON public.query_history
  FOR ALL TO authenticated USING (auth.uid() = farmer_id) WITH CHECK (auth.uid() = farmer_id);

CREATE POLICY "Public read scheme vectors" ON public.scheme_vectors
  FOR SELECT TO authenticated, anon USING (true);
