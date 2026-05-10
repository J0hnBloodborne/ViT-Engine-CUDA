#!/usr/bin/env python3
"""Download and extract the fast.ai `imagenet-mini` archive.
Saves dataset under `imagenet-mini/` by default and writes a `dataset_path.txt` file.
"""
import argparse
import os
import sys
import tarfile
import urllib.request


def download_and_extract(url, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    tgz = os.path.join(out_dir, 'imagenet-mini.tgz')
    if not os.path.exists(tgz):
        print('Downloading', url)
        urllib.request.urlretrieve(url, tgz)
    else:
        print('Using existing', tgz)
    print('Extracting...')
    with tarfile.open(tgz, 'r:gz') as tar:
        tar.extractall(out_dir)

    # Find a directory that looks like an ImageFolder layout (many class subdirs with images)
    def looks_like_imagefolder(p):
        try:
            if not os.path.isdir(p):
                return False
            subs = [d for d in os.listdir(p) if os.path.isdir(os.path.join(p, d))]
            if len(subs) < 2:
                return False
            # check a few subdirs for image files
            checked = 0
            for d in subs[:8]:
                checked += 1
                dd = os.path.join(p, d)
                for f in os.listdir(dd):
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        return True
            return False
        except Exception:
            return False

    candidates = []
    for root, dirs, files in os.walk(out_dir):
        if looks_like_imagefolder(root):
            candidates.append(root)
    if candidates:
        chosen = candidates[0]
    else:
        chosen = out_dir

    dataset_path_file = os.path.join(out_dir, 'dataset_path.txt')
    with open(dataset_path_file, 'w') as f:
        f.write(chosen)

    print('Dataset path:', chosen)
    print('Wrote', dataset_path_file)
    return chosen


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='imagenet-mini', help='Output folder')
    p.add_argument('--url', default=None, help='Optional URL to download (if omitted tries known mirrors)')
    args = p.parse_args()

    default_urls = [
        'https://s3.amazonaws.com/fast-ai-imageclas/imagenet-mini.tgz',
        'http://files.fast.ai/data/imagenet-mini.tgz',
        'https://raw.githubusercontent.com/fastai/imagenet-fast/master/imagenet-mini.tgz'
    ]

    if args.url:
        urls = [args.url]
    else:
        urls = default_urls

    for u in urls:
        try:
            download_and_extract(u, args.out)
            return
        except Exception as e:
            print('Failed to download from', u, '-', e)

    print('All download attempts failed. Please provide a valid URL with --url')


if __name__ == '__main__':
    main()
