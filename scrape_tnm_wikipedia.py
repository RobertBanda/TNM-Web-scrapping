"""
Script to scrape TNM (Telekom Networks Malawi) information from Wikipedia
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
import json

def scrape_tnm_wikipedia():
    """
    Scrape TNM information from Wikipedia
    """
    # Wikipedia URL for TNM
    url = "https://en.wikipedia.org/wiki/Telekom_Networks_Malawi"
    
    try:
        # Send request to Wikipedia
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # Parse the HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract main content
        content = soup.find('div', {'class': 'mw-parser-output'})
        
        # Extract paragraphs
        paragraphs = []
        for p in content.find_all('p'):
            text = p.get_text().strip()
            if text and len(text) > 50:  # Filter out short/empty paragraphs
                paragraphs.append(text)
        
        # Extract infobox data if available
        infobox = soup.find('table', {'class': 'infobox'})
        infobox_data = {}
        if infobox:
            rows = infobox.find_all('tr')
            for row in rows:
                header = row.find('th')
                data = row.find('td')
                if header and data:
                    infobox_data[header.get_text().strip()] = data.get_text().strip()
        
        # Save scraped data
        tnm_data = {
            'url': url,
            'paragraphs': paragraphs[:10],  # First 10 paragraphs
            'infobox': infobox_data
        }
        
        # Save to JSON
        with open('tnm_wikipedia_data.json', 'w', encoding='utf-8') as f:
            json.dump(tnm_data, f, indent=2, ensure_ascii=False)
        
        print("[OK] Successfully scraped TNM data from Wikipedia")
        print(f"[OK] Found {len(paragraphs)} paragraphs")
        print(f"[OK] Infobox entries: {len(infobox_data)}")
        
        return tnm_data
        
    except requests.exceptions.RequestException as e:
        print(f"Error scraping Wikipedia: {e}")
        print("Creating placeholder data structure...")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    print("Scraping TNM information from Wikipedia...")
    data = scrape_tnm_wikipedia()
    if data:
        print("\nSample of scraped content:")
        if data['paragraphs']:
            print(data['paragraphs'][0][:200] + "...")

