#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import random
import sqlite3
import string
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import quote

import aiohttp
import dns.asyncresolver
import dns.exception
import dns.resolver


IANA_RDAP_BOOTSTRAP_URL = "https://data.iana.org/rdap/dns.json"
TERMINAL_STATUSES = {"available", "registered", "dns_exists"}


@dataclass(slots=True)
class ScanResult:
    domain: str
    status: str
    source: str
    detail: str = ""


class Database:
    def __init__(self, path: Path) -> None:
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS domains (
                domain TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                source TEXT NOT NULL,
                checked_at INTEGER NOT NULL,
                detail TEXT NOT NULL DEFAULT ''
            )
            """
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_domains_status ON domains(status)"
        )
        self.connection.commit()

    def load_terminal_domains(self) -> set[str]:
        placeholders = ",".join("?" for _ in TERMINAL_STATUSES)
        rows = self.connection.execute(
            f"SELECT domain FROM domains WHERE status IN ({placeholders})",
            tuple(TERMINAL_STATUSES),
        )
        return {row[0] for row in rows}

    def save_results(self, results: Iterable[ScanResult]) -> None:
        now = int(time.time())
        self.connection.executemany(
            """
            INSERT INTO domains(domain, status, source, checked_at, detail)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(domain) DO UPDATE SET
                status = excluded.status,
                source = excluded.source,
                checked_at = excluded.checked_at,
                detail = excluded.detail
            """,
            (
                (result.domain, result.status, result.source, now, result.detail)
                for result in results
            ),
        )
        self.connection.commit()

    def export_available(self, output_path: Path) -> int:
        rows = self.connection.execute(
            """
            SELECT domain
            FROM domains
            WHERE status = 'available'
            ORDER BY domain
            """
        )

        count = 0
        with output_path.open("w", encoding="utf-8") as output:
            for (domain,) in rows:
                output.write(f"{domain}\n")
                count += 1

        return count

    def status_counts(self) -> dict[str, int]:
        rows = self.connection.execute(
            """
            SELECT status, COUNT(*)
            FROM domains
            GROUP BY status
            """
        )
        return dict(rows)

    def close(self) -> None:
        self.connection.close()


class RateLimiter:
    def __init__(self, requests_per_second: float) -> None:
        if requests_per_second <= 0:
            raise ValueError("requests_per_second must be positive")

        self.interval = 1.0 / requests_per_second
        self.lock = asyncio.Lock()
        self.next_request_time = 0.0

    async def wait(self) -> None:
        async with self.lock:
            now = asyncio.get_running_loop().time()
            delay = self.next_request_time - now

            if delay > 0:
                await asyncio.sleep(delay)
                now = asyncio.get_running_loop().time()

            self.next_request_time = max(now, self.next_request_time) + self.interval


def generate_domains(
    tld: str,
    length: int,
    alphabet: str,
) -> Iterable[str]:
    suffix = f".{tld}"
    for characters in itertools.product(alphabet, repeat=length):
        yield "".join(characters) + suffix


def batched(iterable: Iterable[str], batch_size: int) -> Iterable[list[str]]:
    iterator = iter(iterable)

    while batch := list(itertools.islice(iterator, batch_size)):
        yield batch


async def discover_rdap_base(
    session: aiohttp.ClientSession,
    tld: str,
) -> str:
    async with session.get(IANA_RDAP_BOOTSTRAP_URL) as response:
        response.raise_for_status()
        bootstrap = await response.json(content_type=None)

    for tlds, service_urls in bootstrap["services"]:
        if tld.lower() in {entry.lower() for entry in tlds}:
            if not service_urls:
                break

            return service_urls[0].rstrip("/") + "/"

    raise RuntimeError(f"No RDAP service found for .{tld}")


async def dns_name_exists(
    resolver: dns.asyncresolver.Resolver,
    domain: str,
    semaphore: asyncio.Semaphore,
) -> ScanResult:
    async with semaphore:
        try:
            await resolver.resolve(
                domain + ".",
                "SOA",
                search=False,
                lifetime=resolver.lifetime,
            )
            return ScanResult(domain, "dns_exists", "dns")

        except dns.resolver.NXDOMAIN:
            return ScanResult(domain, "rdap_required", "dns")

        except dns.resolver.NoAnswer:
            return ScanResult(domain, "dns_exists", "dns", "NOERROR/NODATA")

        except (
            dns.resolver.NoNameservers,
            dns.exception.Timeout,
            dns.resolver.LifetimeTimeout,
        ) as error:
            return ScanResult(domain, "unknown", "dns", type(error).__name__)

        except Exception as error:
            return ScanResult(
                domain,
                "unknown",
                "dns",
                f"{type(error).__name__}: {error}",
            )


async def tld_uses_dns_wildcards(
    resolver: dns.asyncresolver.Resolver,
    tld: str,
) -> bool:
    random_label = "".join(
        random.SystemRandom().choices(
            string.ascii_lowercase + string.digits,
            k=40,
        )
    )
    domain = f"{random_label}.{tld}"

    try:
        await resolver.resolve(
            domain + ".",
            "SOA",
            search=False,
            lifetime=resolver.lifetime,
        )
        return True
    except dns.resolver.NXDOMAIN:
        return False
    except dns.resolver.NoAnswer:
        return True
    except Exception:
        return True


async def query_rdap(
    session: aiohttp.ClientSession,
    rdap_base: str,
    domain: str,
    semaphore: asyncio.Semaphore,
    rate_limiter: RateLimiter,
    retries: int,
) -> ScanResult:
    url = f"{rdap_base}domain/{quote(domain, safe='.')}"

    async with semaphore:
        for attempt in range(retries + 1):
            await rate_limiter.wait()

            try:
                async with session.get(
                    url,
                    headers={
                        "Accept": "application/rdap+json, application/json",
                        "User-Agent": "four-character-domain-scanner/1.0",
                    },
                ) as response:
                    if response.status == 200:
                        return ScanResult(domain, "registered", "rdap")

                    if response.status == 404:
                        return ScanResult(domain, "available", "rdap")

                    if response.status == 429:
                        retry_after = response.headers.get("Retry-After")
                        delay = (
                            float(retry_after)
                            if retry_after and retry_after.isdigit()
                            else min(2**attempt, 60)
                        )
                        await asyncio.sleep(delay)
                        continue

                    if 500 <= response.status < 600:
                        await asyncio.sleep(min(2**attempt, 60))
                        continue

                    body = await response.text()
                    return ScanResult(
                        domain,
                        "unknown",
                        "rdap",
                        f"HTTP {response.status}: {body[:200]}",
                    )

            except (
                aiohttp.ClientError,
                asyncio.TimeoutError,
            ) as error:
                if attempt < retries:
                    await asyncio.sleep(min(2**attempt, 60))
                    continue

                return ScanResult(
                    domain,
                    "unknown",
                    "rdap",
                    f"{type(error).__name__}: {error}",
                )

    return ScanResult(domain, "unknown", "rdap", "Retries exhausted")


async def scan(args: argparse.Namespace) -> None:
    database = Database(args.database)
    terminal_domains = database.load_terminal_domains()

    timeout = aiohttp.ClientTimeout(total=args.http_timeout)
    connector = aiohttp.TCPConnector(limit=args.rdap_concurrency)

    resolver = dns.asyncresolver.Resolver(configure=not bool(args.nameserver))
    resolver.timeout = args.dns_timeout
    resolver.lifetime = args.dns_timeout

    if args.nameserver:
        resolver.nameservers = args.nameserver

    dns_semaphore = asyncio.Semaphore(args.dns_concurrency)
    rdap_semaphore = asyncio.Semaphore(args.rdap_concurrency)
    rate_limiter = RateLimiter(args.rdap_rps)

    processed = 0
    skipped = 0

    try:
        async with aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
        ) as session:
            rdap_base = args.rdap_base or await discover_rdap_base(
                session,
                args.tld,
            )

            use_dns_prefilter = not args.rdap_all

            if use_dns_prefilter:
                wildcarded = await tld_uses_dns_wildcards(
                    resolver,
                    args.tld,
                )

                if wildcarded:
                    print(
                        f".{args.tld} appears to use DNS wildcards; "
                        "disabling DNS prefilter."
                    )
                    use_dns_prefilter = False

            domains = generate_domains(
                tld=args.tld,
                length=args.length,
                alphabet=args.alphabet,
            )

            for batch_number, batch in enumerate(
                batched(domains, args.batch_size),
                start=1,
            ):
                pending = []

                for domain in batch:
                    if domain in terminal_domains:
                        skipped += 1
                    else:
                        pending.append(domain)

                if not pending:
                    continue

                if use_dns_prefilter:
                    dns_results = await asyncio.gather(
                        *(
                            dns_name_exists(
                                resolver,
                                domain,
                                dns_semaphore,
                            )
                            for domain in pending
                        )
                    )

                    database.save_results(
                        result
                        for result in dns_results
                        if result.status != "rdap_required"
                    )

                    rdap_domains = [
                        result.domain
                        for result in dns_results
                        if result.status == "rdap_required"
                    ]
                else:
                    rdap_domains = pending

                if rdap_domains:
                    rdap_results = await asyncio.gather(
                        *(
                            query_rdap(
                                session=session,
                                rdap_base=rdap_base,
                                domain=domain,
                                semaphore=rdap_semaphore,
                                rate_limiter=rate_limiter,
                                retries=args.retries,
                            )
                            for domain in rdap_domains
                        )
                    )
                    database.save_results(rdap_results)

                processed += len(pending)

                counts = database.status_counts()
                print(
                    json.dumps(
                        {
                            "batch": batch_number,
                            "processed_this_run": processed,
                            "skipped_from_checkpoint": skipped,
                            "available": counts.get("available", 0),
                            "registered": counts.get("registered", 0),
                            "dns_exists": counts.get("dns_exists", 0),
                            "unknown": counts.get("unknown", 0),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

        available_count = database.export_available(args.output)

        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "available_domains": available_count,
                    "database": str(args.database),
                },
                sort_keys=True,
            )
        )

    finally:
        database.close()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Enumerate fixed-length domain names and use DNS plus RDAP "
            "to identify names that are not currently registered."
        )
    )

    parser.add_argument(
        "--tld",
        default="com",
        help="Top-level domain without the leading dot.",
    )
    parser.add_argument(
        "--length",
        type=int,
        default=4,
        help="Length of the domain label.",
    )
    parser.add_argument(
        "--alphabet",
        default=string.ascii_lowercase,
        help="Characters permitted in generated domain labels.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("available_domains.txt"),
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("domain_scan.sqlite3"),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--dns-concurrency",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--dns-timeout",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--nameserver",
        action="append",
        default=[],
        help="DNS resolver IP address. May be specified multiple times.",
    )
    parser.add_argument(
        "--rdap-concurrency",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--rdap-rps",
        type=float,
        default=2.0,
        help="Maximum RDAP request start rate.",
    )
    parser.add_argument(
        "--rdap-base",
        help="Override the RDAP base URL discovered from IANA.",
    )
    parser.add_argument(
        "--rdap-all",
        action="store_true",
        help="Query RDAP for every generated domain instead of using DNS first.",
    )
    parser.add_argument(
        "--http-timeout",
        type=float,
        default=30.0,
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=5,
    )

    args = parser.parse_args()
    args.tld = args.tld.lower().lstrip(".")

    if args.length < 1:
        parser.error("--length must be at least 1")

    if not args.alphabet:
        parser.error("--alphabet cannot be empty")

    if len(set(args.alphabet)) != len(args.alphabet):
        parser.error("--alphabet must not contain duplicate characters")

    return args


def main() -> None:
    args = parse_arguments()
    asyncio.run(scan(args))


if __name__ == "__main__":
    main()