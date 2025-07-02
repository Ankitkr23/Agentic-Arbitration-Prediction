import asyncio
import os
import time
import html
import json
from urllib.parse import urlparse
from pathlib import Path

# Import necessary libraries for web scraping
from crawlee.crawlers import PlaywrightCrawler, PlaywrightCrawlingContext
from googlesearch import search # Using googlesearch-python for Google search results

# Apply nest_asyncio to allow asyncio.run() in environments like Jupyter notebooks
# where an event loop might already be running.
import nest_asyncio
nest_asyncio.apply()

# --- Configuration ---
# The phrase to search for on Google to find relevant arbitration dispute documents.
PHRASE = "AES Solar v. Kingdom of Spain"
# The number of top search results to retrieve.
TOP_N = 10
# The base directory where scraped content (HTML and JSON) will be saved.
# The directory name is derived from the PHRASE.
output_directory = f"scraped_content/{PHRASE.replace(' ', '_')[:10]}"

# --- Helper Functions ---

def get_top_urls(query, num_results):
    """
    Performs a Google search for the given query and returns a list of top URLs.
    Includes a small delay between requests to avoid being blocked.
    """
    urls = []
    print(f"Searching Google for: {query}")
    for url in search(query, num_results=num_results):
        urls.append(url)
        print(f"Found URL: {url}")
        time.sleep(0.1) # Small delay to be polite to Google
    return urls

def sanitize_filename(url):
    """
    Sanitizes a URL to create a valid and concise filename.
    Removes 'www.' from the domain and replaces '/' with '_' in the path.
    Limits the filename length to 100 characters to prevent issues.
    """
    parsed = urlparse(url)
    netloc = parsed.netloc
    if netloc.startswith('www.'):
        netloc = netloc[4:]
    # Combine netloc and path, replacing non-alphanumeric characters with '_'
    filename = netloc + parsed.path.replace('/', '_')
    filename = ''.join(c if c.isalnum() or c in ['_', '-', '.'] else '_' for c in filename)
    return filename[:100] # Limit filename length

def create_label_studio_xml(html_content, url, title):
    """
    Generates XML and JSON content suitable for importing into Label Studio.
    The XML defines the annotation interface, and the JSON contains the raw data.
    Note: The provided code only uses the JSON part for saving.
    """
    # Create a JSON structure with the HTML content
    json_content = json.dumps([{"html_content": html_content}])
    # Create an XML configuration for Label Studio to view HTML and add labels
    xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<View>
  <HyperText name="text" value="$html_content" valueType="text" />
  <Labels name="labels" toName="text">
    <Label value="Important" />
    <Label value="Review" />
  </Labels>
  <Meta>
    <Info name="url" value="{html.escape(url)}"/>
    <Info name="title" value="{html.escape(title)}"/>
  </Meta>
</View>"""
    return xml_content, json_content

# --- Main Asynchronous Web Scraping Logic ---

async def main() -> None:
    """
    Main function to orchestrate the web scraping process.
    It performs Google searches, initializes the Playwright crawler,
    and saves the scraped HTML and JSON content.
    """
    # Define output paths and create directories if they don't exist
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    html_output_path = output_path / "html"
    html_output_path.mkdir(exist_ok=True)

    json_output_path = output_path / "json"
    json_output_path.mkdir(exist_ok=True)

    # Get top URLs from Google search
    urls = get_top_urls(PHRASE, TOP_N)

    # Initialize PlaywrightCrawler
    # headless=True means the browser will run in the background without a UI
    crawler = PlaywrightCrawler(
        headless=True,
    )

    # Define the request handler for the crawler
    # This function will be called for each URL to be scraped
    @crawler.router.default_handler
    async def request_handler(context: PlaywrightCrawlingContext) -> None:
        url = context.request.url
        context.log.info(f'Processing {url} ...')

        # Wait for the page to load all network requests
        await context.page.wait_for_load_state("networkidle", timeout=30000)
        # Get the full HTML content of the page
        html_content = await context.page.content()
        # Get the title of the page
        title = await context.page.title()

        # Prepare content for Label Studio (though only JSON is saved in this script)
        _, json_content_for_ls = create_label_studio_xml(html_content, url, title)

        # Sanitize the URL to create a safe filename
        filename = sanitize_filename(url)

        # Save JSON content
        json_filename = filename + '.json'
        json_file_path = json_output_path / json_filename
        # Save the HTML content wrapped in a JSON array for Label Studio compatibility
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump([{"html_content": html_content}], f, ensure_ascii=False, indent=4)
        context.log.info(f'Saved JSON content to {json_file_path}')

        # Save raw HTML content
        html_filename = filename + '.html'
        html_file_path = html_output_path / html_filename
        with open(html_file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        context.log.info(f'Saved HTML content to {html_file_path}')

        # Push data about the scraped URL to the crawler's dataset
        data = {
            'url': url,
            'title': title,
            'saved_json': str(json_file_path),
            'saved_html': str(html_file_path),
        }
        await context.push_data(data)

    # Run the crawler with the list of URLs
    await crawler.run(urls)

    print(f"Scraping completed. Files saved to:")
    print(f"- JSON: {output_directory}/json/")
    print(f"- HTML: {output_directory}/html/")
    print(f"You can now import the JSON files into Label Studio for annotation.")

# --- Entry Point ---
if __name__ == '__main__':
    # Run the main asynchronous function
    asyncio.run(main())
