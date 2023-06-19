# https://github.com/RiddlerQ/simple_image_download

from cmath import e
from msilib.schema import Error
from typing import final
import simple_image_download.simple_image_download as simp


queries = ['Mars', 'Jupiter', 'Venus', 'Saturn', 'Neptun', 'Pluto-Globe']

response = simp.Downloader()

res = []

for query in queries:
    print(f"Downloading for {query} Query...")
    
    # print(response.cached_urls)
    try:
        urls = response.search_urls(keywords=query, limit=10, timer=None)

    except Exception as e:
        print(e)
        print(response.get_urls())
        continue

    finally:
        print(response.get_urls)
        response.download(keywords=query, limit=10, verbose=False, download_cache=True)

    response.flush_cache()


