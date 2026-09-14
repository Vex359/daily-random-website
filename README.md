# Daily Random Website

A Python pipeline that curates interesting content from across the internet and generates a fresh static website every day.

## Content Sources

- **Hacker News** — trending tech stories
- **GitHub** — trending repositories
- **RSS Feeds** — curated content from top feeds
- **Wikipedia** — random interesting articles
- **Awesome Lists** — curated resource collections

## Project Structure

```
src/
├── collectors/    # Fetch content from various sources
├── filters/       # Domain, safety, quality, and dedup filters
├── scoring/       # Interestingness scoring algorithm
├── ai/            # AI-powered content enhancement
├── screenshot/    # Playwright-based website screenshots
├── storage/       # Local JSON persistence
├── site/          # Static site generator
├── pipeline.py    # Pipeline orchestrator
└── main.py        # Entry point
```

## Setup

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install Playwright browsers
playwright install chromium

# Run the pipeline
python -m src.main
```

## Deployment

Deployed to Netlify as a static site from the `dist/` directory.
