from multiprocessing import Pool
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
from glob import glob
import os, json, csv, html
from typing import List
from tqdm import tqdm


def get_unix_style_paths(dir_path:str) -> List[str]:
    files = glob(dir_path)
    return list(map(lambda s: s.replace('\\','/'), files))


def file_to_str(filepath:str) -> str:
    with open(filepath, encoding='utf-8') as f:
        return f.read()

def file_to_list(filepath:str) -> List[str]:
    with open(filepath, encoding='utf-8') as f:
        return f.readlines()

def open_file(filename:str, mode:str='w'):
    csvfile = open(filename, mode=mode, newline='', encoding='utf-8')
    return  csv.writer(csvfile, delimiter='|', quotechar='"')

def write_to_csv_file(data:List[str], csv_writer)-> None:
        # csv_writer.writerows(data)
        csv_writer.writerow(data)
        return


def build_imdb_dataset(movie_urls, review_file):
    # print(review_file)
    review_text = file_to_str(review_file)
    idx, rank = os.path.splitext(os.path.basename(review_file))[0].split('_') 
    r = Request(movie_urls[int(idx)], headers={"User-Agent": 'Chrome/110.0.0.0'})
    webpage_content = BeautifulSoup(urlopen(r).read(), 'html.parser')
    movie_content = json.loads(webpage_content.find('script', {'type':'application/ld+json'}).string)
    row = idx.zfill(5), html.unescape(movie_content['name']), review_text, int(rank)
    return row


def map_movie_review_to_url(review_file):
    idx, rank = os.path.splitext(os.path.basename(review_file))[0].split('_')
    return int(idx), int(rank)


def process_data_file(path_file:str)->None:

    base_dir, url_file = os.path.split(path_file)
    category = '/pos' if url_file.find('pos') > 0 else '/neg'
    reviews_dir = base_dir + category + '/*'
    reviews_file_paths = get_unix_style_paths(reviews_dir)
    final_data_filename = base_dir + f'{category}_imdb_data.csv'
    
    movie_name_urls = file_to_list(path_file)
    movie_name_urls = list(map(lambda s: s.replace('usercomments\n',''), movie_name_urls))
    
    csv_writer = open_file(final_data_filename)
    with tqdm(total=len(reviews_file_paths)) as subbar:    
        for p in reviews_file_paths:
            data = build_imdb_dataset(movie_name_urls, p)
            write_to_csv_file(data, csv_writer)
            subbar.update()


if __name__ == '__main__':

    imdb_path= './Project_UQE/Datasets/IMDB/**/*.txt'
    url_files = get_unix_style_paths(imdb_path)
    print(url_files)
    url_files = ['./Project_UQE/Datasets/IMDB/test/urls_neg.txt', './Project_UQE/Datasets/IMDB/test/urls_pos.txt']

    with Pool(processes=2) as pool:
        max_files = len(url_files)
        with tqdm(total=max_files) as bar:
            for _ in pool.map(process_data_file, url_files):
                bar.update()