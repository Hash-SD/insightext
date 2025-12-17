# Sentinel Security Journal

## 2024-12-16 - Hardcoded Secrets in .env File

**Vulnerability:** Real API keys and secrets were present in the `.env` file:
- Supabase API key
- GitHub personal access token  
- Admin password

**Learning:** Even though `.env` is in `.gitignore`, the file was committed with real credentials. This is a common mistake when developers forget to use `.env.example` as a template.

**Prevention:** 
- Always use `.env.example` with placeholder values for version control
- Never commit real secrets, even temporarily
- Use secret scanning tools in CI/CD pipeline
- Rotate any exposed credentials immediately
