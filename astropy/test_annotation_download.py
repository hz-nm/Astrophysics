# This file will serve as a testbed for downloading files from Google
# We are using the Library -> google_images_download
# https://github.com/hardikvasa/google-images-download
# Tutorial -> https://www.geeksforgeeks.org/how-to-download-google-images-using-python/

from google_images_download import google_images_download

# create an object
response = google_images_download.googleimagesdownload()

# search_queries = [
#     'Mars'
# ]

# def download_images(query):
#     # keywords is the search query
#     # format -> Image file format
#     # limit -> Number of images to download
#     # print urls to print the image file
#     # size is the image size which can be specified manually ("large", "medium", "icon")
#     # aspect ratio denotes the height width ratio of the images to be downloaded ("tall", "square", "wide", "panoramic")
#     arguments ={
#         "keywords": query,
#         "format": "jpg",
#         "limit": 4,
#         "print_urls": True,
#         "size": "medium",
#         "aspect_ratio": "square"
#     }

#     try:
#         response.download(arguments)

#     except FileNotFoundError:
#         print("File was not found")

#         try:
#             response.download(arguments)
#         except:
#             pass
    

# for query in search_queries:
#     print(f"Downloading images for {query} query")
#     download_images(query)
#     print("Images Downloaded")
#     print()
arguments = {
        "keywords": "Mars",
        "limit": 10,
        "print_urls": True
}

paths = response.download(arguments)
print(paths)

