#!/usr/bin/env python3
"""Extract and download WeChat article images.

Supports two modes:
1) history mode: parse a public account history page and download list cover images.
2) article mode: parse a single article and download in-article images plus cover candidates.

Notes:
- WeChat pages often require login/cookies. Use --cookie when needed.
- This tool is for lawful use on content you are authorized to access.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


@dataclass
class ArticleItem:
    title: str
    content_url: str
    cover_url: str


class WechatImageExtractor:
    def __init__(self, timeout: int = 20, sleep_sec: float = 0.2):
        self.timeout = timeout
        self.sleep_sec = sleep_sec

    def _session(self, cookie: str = "") -> requests.Session:
        s = requests.Session()
        s.headers.update({"User-Agent": DEFAULT_UA})
        if cookie.strip():
            s.headers["Cookie"] = cookie.strip()
        return s

    def fetch_text(self, url: str, cookie: str = "") -> str:
        s = self._session(cookie=cookie)
        r = s.get(url, timeout=self.timeout)
        r.raise_for_status()
        # Most WeChat pages are UTF-8, but requests may need hint.
        r.encoding = r.encoding or "utf-8"
        return r.text

    def parse_history_articles(self, html_text: str) -> List[ArticleItem]:
        """Parse msgList from WeChat history page HTML.

        Expected JavaScript snippets include one of:
        - var msgList = '...json string...';
        - msgList: '...json string...'
        """
        raw_msg_list = self._extract_msg_list_raw(html_text)
        if not raw_msg_list:
            return []

        decoded = html.unescape(raw_msg_list)
        # Common history pages escape slashes and quotes in JS strings.
        decoded = decoded.encode("utf-8").decode("unicode_escape")

        data = json.loads(decoded)
        items: List[ArticleItem] = []

        for message in data.get("list", []):
            # Primary message cover.
            comm = message.get("comm_msg_info", {})
            default_cover = str(comm.get("cover", "")).strip()

            # Main article.
            ext = message.get("app_msg_ext_info", {})
            main_title = str(ext.get("title", "")).strip()
            main_content_url = self._normalize_mp_url(str(ext.get("content_url", "")).strip())
            main_cover = str(ext.get("cover", "")).strip() or default_cover
            if main_content_url or main_cover:
                items.append(ArticleItem(main_title, main_content_url, main_cover))

            # Multi-article cards.
            for sub in ext.get("multi_app_msg_item_list", []) or []:
                title = str(sub.get("title", "")).strip()
                content_url = self._normalize_mp_url(str(sub.get("content_url", "")).strip())
                cover = str(sub.get("cover", "")).strip() or default_cover
                if content_url or cover:
                    items.append(ArticleItem(title, content_url, cover))

        return items

    def _extract_msg_list_raw(self, html_text: str) -> str:
        patterns = [
            r"var\s+msgList\s*=\s*'(?P<v>.*?)';",
            r"msgList\s*:\s*'(?P<v>.*?)'",
            r"window\.__INITIAL_STATE__\s*=\s*(?P<v>\{.*?\})\s*;",
        ]

        for p in patterns:
            m = re.search(p, html_text, flags=re.S)
            if not m:
                continue
            v = m.group("v")
            # If the page exposes __INITIAL_STATE__, try to locate msgList inside it.
            if p.startswith("window"):
                try:
                    state = json.loads(v)
                    msg_list = state.get("msgList")
                    if isinstance(msg_list, str) and msg_list.strip():
                        return msg_list
                except Exception:
                    pass
                continue
            return v

        return ""

    def parse_article_images(self, html_text: str, base_url: str) -> Dict[str, List[str]]:
        """Extract article image URLs and cover candidates from an article page."""
        soup = BeautifulSoup(html_text, "html.parser")

        body_images: Set[str] = set()
        cover_images: Set[str] = set()

        for meta_key in [
            ("property", "og:image"),
            ("name", "twitter:image"),
            ("property", "twitter:image"),
        ]:
            tag = soup.find("meta", attrs={meta_key[0]: meta_key[1]})
            if tag and tag.get("content"):
                cover_images.add(self._normalize_url(tag.get("content", ""), base_url))

        for img in soup.select("img"):
            candidate = (
                img.get("data-src")
                or img.get("data-original")
                or img.get("src")
                or ""
            ).strip()
            if not candidate:
                continue
            full = self._normalize_url(candidate, base_url)
            if full:
                body_images.add(full)

        return {
            "cover_images": sorted(x for x in cover_images if x),
            "body_images": sorted(x for x in body_images if x),
        }

    def download_images(
        self,
        urls: Iterable[str],
        out_dir: str,
        cookie: str = "",
        prefix: str = "img",
    ) -> List[str]:
        os.makedirs(out_dir, exist_ok=True)
        s = self._session(cookie=cookie)
        saved: List[str] = []

        for idx, url in enumerate(urls, 1):
            if not url:
                continue
            ext = self._guess_ext(url)
            file_name = f"{prefix}_{idx:04d}{ext}"
            path = os.path.join(out_dir, file_name)

            try:
                r = s.get(url, timeout=self.timeout)
                r.raise_for_status()
                with open(path, "wb") as f:
                    f.write(r.content)
                saved.append(path)
            except Exception as exc:
                print(f"[warn] download failed: {url} -> {exc}")

            if self.sleep_sec > 0:
                time.sleep(self.sleep_sec)

        return saved

    @staticmethod
    def _normalize_mp_url(url: str) -> str:
        if not url:
            return ""
        if url.startswith("//"):
            return "https:" + url
        if url.startswith("/"):
            return "https://mp.weixin.qq.com" + url
        if url.startswith("http://") or url.startswith("https://"):
            return url
        return "https://mp.weixin.qq.com/" + url.lstrip("/")

    @staticmethod
    def _normalize_url(url: str, base_url: str) -> str:
        if not url:
            return ""
        if url.startswith("//"):
            return "https:" + url
        return urljoin(base_url, url)

    @staticmethod
    def _guess_ext(url: str) -> str:
        path = urlparse(url).path.lower()
        for ext in [".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"]:
            if path.endswith(ext):
                return ext
        return ".jpg"


def write_history_csv(rows: List[ArticleItem], csv_path: str) -> None:
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["title", "content_url", "cover_url"])
        for r in rows:
            w.writerow([r.title, r.content_url, r.cover_url])


def cmd_history(args: argparse.Namespace) -> int:
    extractor = WechatImageExtractor(timeout=args.timeout, sleep_sec=args.sleep)
    page = extractor.fetch_text(args.url, cookie=args.cookie)
    items = extractor.parse_history_articles(page)

    if not items:
        print("No article list found. Try passing browser cookie via --cookie.")
        return 2

    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, "history_articles.csv")
    write_history_csv(items, csv_path)

    covers = [it.cover_url for it in items if it.cover_url]
    # Keep order and remove duplicates.
    covers = list(dict.fromkeys(covers))
    cover_dir = os.path.join(args.out, "covers")
    saved = extractor.download_images(covers, cover_dir, cookie=args.cookie, prefix="cover")

    print(f"Articles parsed: {len(items)}")
    print(f"Unique covers: {len(covers)}")
    print(f"Covers downloaded: {len(saved)}")
    print(f"CSV saved: {csv_path}")
    print(f"Cover dir: {cover_dir}")
    return 0


def cmd_article(args: argparse.Namespace) -> int:
    extractor = WechatImageExtractor(timeout=args.timeout, sleep_sec=args.sleep)
    page = extractor.fetch_text(args.url, cookie=args.cookie)
    found = extractor.parse_article_images(page, base_url=args.url)

    os.makedirs(args.out, exist_ok=True)
    json_path = os.path.join(args.out, "article_images.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(found, f, ensure_ascii=False, indent=2)

    cover_dir = os.path.join(args.out, "cover_candidates")
    body_dir = os.path.join(args.out, "body_images")
    saved_cover = extractor.download_images(found["cover_images"], cover_dir, cookie=args.cookie, prefix="cover")
    saved_body = extractor.download_images(found["body_images"], body_dir, cookie=args.cookie, prefix="body")

    print(f"Cover candidates found: {len(found['cover_images'])}")
    print(f"Body images found: {len(found['body_images'])}")
    print(f"Cover images downloaded: {len(saved_cover)}")
    print(f"Body images downloaded: {len(saved_body)}")
    print(f"Metadata json: {json_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extract WeChat article images (history/article).")
    p.add_argument("--timeout", type=int, default=20, help="HTTP timeout seconds")
    p.add_argument("--sleep", type=float, default=0.2, help="Delay between downloads")

    sp = p.add_subparsers(dest="mode", required=True)

    p_history = sp.add_parser("history", help="Parse account history page and download list covers")
    p_history.add_argument("--url", required=True, help="WeChat history page URL")
    p_history.add_argument("--cookie", default="", help="Browser cookie string if required")
    p_history.add_argument("--out", default="wechat_history_output", help="Output directory")
    p_history.set_defaults(func=cmd_history)

    p_article = sp.add_parser("article", help="Parse a single article and download images")
    p_article.add_argument("--url", required=True, help="Single article URL")
    p_article.add_argument("--cookie", default="", help="Browser cookie string if required")
    p_article.add_argument("--out", default="wechat_article_output", help="Output directory")
    p_article.set_defaults(func=cmd_article)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
