-- Keep public access denied while allowing trusted server-side Supabase roles.
-- The application still enforces account ownership in its API layer.

CREATE OR REPLACE FUNCTION public.set_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = public
AS $$
BEGIN
    NEW.updated_at = timezone('utc', now());
    RETURN NEW;
END;
$$;

CREATE POLICY users_backend_only ON public.users
    FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY projects_backend_only ON public.projects
    FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY documents_backend_only ON public.documents
    FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY ingest_jobs_backend_only ON public.ingest_jobs
    FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY conversations_backend_only ON public.conversations
    FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY messages_backend_only ON public.messages
    FOR ALL TO service_role USING (true) WITH CHECK (true);
