-- Revocable JWT sessions for the custom API auth layer.
CREATE TABLE IF NOT EXISTS public.revoked_tokens (
    jti TEXT PRIMARY KEY,
    user_id BIGINT NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
    expires_at TIMESTAMPTZ NOT NULL,
    revoked_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc', now())
);

CREATE INDEX IF NOT EXISTS idx_revoked_tokens_user_id ON public.revoked_tokens(user_id);
CREATE INDEX IF NOT EXISTS idx_revoked_tokens_expires_at ON public.revoked_tokens(expires_at);

ALTER TABLE public.revoked_tokens ENABLE ROW LEVEL SECURITY;
CREATE POLICY revoked_tokens_backend_only ON public.revoked_tokens
    FOR ALL TO service_role USING (true) WITH CHECK (true);
