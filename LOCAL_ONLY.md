# Local-Only Configuration

This document explains how the app is configured to work **completely offline** without any external network connections.

## What Was Disabled

### 1. **Streamlit Telemetry**
- **File**: `.streamlit/config.toml`
- **Setting**: `gatherUsageStats = false`
- **Purpose**: Prevents Streamlit from sending usage statistics to their servers

### 2. **LangChain/LangSmith Tracing**
- **Files**: `agent.py`, `.env`
- **Settings**: 
  - `LANGCHAIN_TRACING_V2=false`
  - `LANGCHAIN_ENDPOINT=""`
- **Purpose**: Disables LangSmith cloud tracing and monitoring

### 3. **Analytics**
- **Files**: `app.py`, `.env`
- **Setting**: `STREAMLIT_BROWSER_GATHER_USAGE_STATS=false`
- **Purpose**: Disables browser-based analytics

## Network Connections

The app now **ONLY** connects to:
- ✅ **Ollama** at `http://localhost:11434` (local LLM)
- ✅ **Streamlit** at `http://localhost:8501` (local web UI)

**No external internet connections are made.**

## macOS Permissions

### Why You Saw Permission Prompts

1. **Apple Music/Media Library**: 
   - `psutil` library tries to enumerate all processes
   - Some processes may be related to media services
   - **Solution**: You can deny this - the app doesn't need media access

2. **Network Access**:
   - Before our changes, LangChain tried to connect to LangSmith
   - Streamlit tried to send telemetry
   - **Solution**: Now disabled - you shouldn't see this again

### What to Allow/Deny

- ❌ **Deny**: Apple Music, Media Library access
- ❌ **Deny**: Network access (if prompted again)
- ✅ **Allow**: File system access (only for directories you configure in safe zones)

## Restart Instructions

**Stop the current app** (Ctrl+C) and restart:

```bash
streamlit run app.py
```

You should **NOT** see any permission prompts for network or media access anymore.

## Verification

To verify the app is truly local-only:

1. **Disconnect from internet** (turn off WiFi)
2. **Run the app**: `streamlit run app.py`
3. **Test a prompt**: "List all Python files in my home directory"

If it works without internet, you're 100% local! ✅

## Files Created

- `.streamlit/config.toml` - Streamlit configuration
- `.env` - Environment variables (already in `.gitignore`)
- Both added to `.gitignore` for security
