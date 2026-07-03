import os
import urllib.request
import json
import urllib.parse
import time

# Dictionary of dish tags mapping to targeted search queries
DISH_QUERIES = {
    "banh-mi": '"Bánh mì" food',
    "pho": '"Phở bò" OR "Phở gà"',
    "bun-bo-hue": '"Bún bò Huế"',
    "com-tam": '"Cơm tấm" food',
    "banh-xeo": '"Bánh xèo"',
    "goi-cuon": '"Gỏi cuốn"',
    "bun-cha": '"Bún chả"',
    "mi-quang": '"Mì Quảng"',
    "banh-bao": '"Bánh bao" food',
    "banh-bot-loc": '"Bánh bột lọc"',
    "banh-chung": '"Bánh chưng"',
    "banh-khot": '"Bánh khọt"',
    "bo-kho": '"Bò kho" food',
    "bun-rieu": '"Bún riêu"',
    "ca-kho-to": '"Cá kho tộ, cá hú"',
    "canh-chua": '"Canh chua"',
    "chao-long": '"Cháo lòng"',
    "com-ga-xoi-mo": '"Cơm gà"',
    "thit-kho-tau": '"Thịt kho tàu"',
    "xoi-xeo": '"Xôi xéo"'
}

def search_wikimedia_file(query, headers):
    try:
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srnamespace": 6,  # Namespace 6 is for Files
            "format": "json",
            "srlimit": 10      # Fetch top 10 results to filter for images
        }
        query_string = urllib.parse.urlencode(params)
        url = f"https://commons.wikimedia.org/w/api.php?{query_string}"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode('utf-8'))
            results = data.get("query", {}).get("search", [])
            for r in results:
                title = r.get("title", "")
                title_lower = title.lower()
                # Filter strictly for standard image extensions
                if title_lower.endswith(('.jpg', '.jpeg', '.png')):
                    return title
    except Exception as e:
        print(f"Error searching for {query}: {e}")
    return None

def get_wikimedia_image_url(file_title, headers):
    try:
        params = {
            "action": "query",
            "titles": file_title,
            "prop": "imageinfo",
            "iiprop": "url",
            "format": "json"
        }
        query_string = urllib.parse.urlencode(params)
        url = f"https://commons.wikimedia.org/w/api.php?{query_string}"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode('utf-8'))
            pages = data.get("query", {}).get("pages", {})
            for page_id, page_info in pages.items():
                imageinfo = page_info.get("imageinfo", [])
                if imageinfo:
                    return imageinfo[0].get("url")
    except Exception as e:
        print(f"Error resolving URL: {e}")
    return None

def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(root_dir, "test-image")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
        
    headers = {
        'User-Agent': 'MunchinAppTestBot/1.0 (contact@munchin.app) Python/urllib'
    }
    
    print("Starting download of 20 Vietnamese food test images via Wikimedia Search...")
    
    for tag, query in DISH_QUERIES.items():
        dest_path = os.path.join(output_dir, f"{tag}.jpg")
        print(f"\nSearching for {tag}...")
        file_title = search_wikimedia_file(query, headers)
        
        if not file_title:
            print(f"Could not find any file for {tag}")
            continue
            
        url = get_wikimedia_image_url(file_title, headers)
        if not url:
            print(f"Could not resolve URL for {tag}")
            continue
            
        print(f"Downloading {tag} from {url}...")
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req) as response:
                with open(dest_path, 'wb') as out_file:
                    out_file.write(response.read())
            print(f"Successfully saved to {dest_path}")
        except Exception as e:
            print(f"Failed to download {tag} from {url}: {e}")
            
        time.sleep(1.5) # Compliance delay to respect rate limits

    print("\nDownload process completed.")

if __name__ == "__main__":
    main()
