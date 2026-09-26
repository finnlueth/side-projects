import Foundation
import Combine

struct CoinEntry: Codable, Hashable {
    let id: Int
    let symbol: String
    let name: String
    let rank: Int
}

/// Offline coin search backed by CoinMarketCap's id map (~8k active listings, two free calls), cached on
/// disk and refreshed weekly.
@MainActor
final class CoinDirectory: ObservableObject {
    @Published private(set) var coins: [CoinEntry] = []
    /// Called after a fresh load so tickers added by symbol before the map arrived can be resolved.
    var onLoad: (() -> Void)?

    private static let maxCacheAge: TimeInterval = 7 * 24 * 3600
    private var isLoading = false

    private let cacheURL: URL = {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        return base.appendingPathComponent("StockTicker/coins.json")
    }()

    func loadIfNeeded(apiKey: String) {
        guard !isLoading else { return }
        if coins.isEmpty, let cached = try? Data(contentsOf: cacheURL),
           let decoded = try? JSONDecoder().decode([CoinEntry].self, from: cached) {
            coins = decoded
            onLoad?()
        }
        let cacheAge = (try? cacheURL.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate)
            .map { Date().timeIntervalSince($0) } ?? .infinity
        guard coins.isEmpty || cacheAge > Self.maxCacheAge else { return }
        guard !apiKey.isEmpty else { return }

        isLoading = true
        Task {
            defer { isLoading = false }
            guard let fetched = try? await Self.fetch(apiKey: apiKey), !fetched.isEmpty else { return }
            coins = fetched
            try? FileManager.default.createDirectory(at: cacheURL.deletingLastPathComponent(), withIntermediateDirectories: true)
            try? JSONEncoder().encode(fetched).write(to: cacheURL, options: .atomic)
            onLoad?()
        }
    }

    /// The highest-ranked coin with exactly this symbol.
    func bestMatch(symbol: String) -> CoinEntry? {
        let wanted = symbol.uppercased()
        return coins.lazy.filter { $0.symbol == wanted }.min { $0.rank < $1.rank }
    }

    /// Symbol matches (exact or prefix) first, then name matches; within a tier, market-cap rank decides.
    /// Exact and prefix share a tier on purpose: thousands of junk listings reuse popular symbols, and
    /// rank separates Ethereum from "Bridged ETH" far better than exactness does.
    func suggestions(for query: String, excluding: Set<Int>, limit: Int = 6) -> [CoinEntry] {
        let q = query.trimmingCharacters(in: .whitespaces).uppercased()
        guard !q.isEmpty else { return [] }
        let needle = q.lowercased()

        func score(_ coin: CoinEntry) -> Int? {
            if coin.symbol.hasPrefix(q) { return 0 }
            let name = coin.name.lowercased()
            if name.hasPrefix(needle) { return 1 }
            if q.count >= 3, name.contains(needle) { return 2 }
            return nil
        }

        var scored: [(coin: CoinEntry, score: Int)] = []
        for coin in coins where !excluding.contains(coin.id) {
            if let value = score(coin) { scored.append((coin, value)) }
        }
        scored.sort { a, b in
            if a.score != b.score { return a.score < b.score }
            return a.coin.rank < b.coin.rank
        }
        return scored.prefix(limit).map(\.coin)
    }

    private static func fetch(apiKey: String) async throws -> [CoinEntry] {
        var all: [CoinEntry] = []
        var start = 1
        // The map endpoint pages 5,000 at a time and is free of credits.
        while start < 20_000 {
            let page = try await fetchPage(start: start, apiKey: apiKey)
            all += page
            if page.count < 5000 { break }
            start += 5000
        }
        return all
    }

    private static func fetchPage(start: Int, apiKey: String) async throws -> [CoinEntry] {
        var components = URLComponents(string: "https://pro-api.coinmarketcap.com/v1/cryptocurrency/map")!
        components.queryItems = [
            URLQueryItem(name: "listing_status", value: "active"),
            URLQueryItem(name: "sort", value: "cmc_rank"),
            URLQueryItem(name: "limit", value: "5000"),
            URLQueryItem(name: "start", value: String(start)),
        ]
        var request = URLRequest(url: components.url!)
        request.setValue(apiKey, forHTTPHeaderField: "X-CMC_PRO_API_KEY")
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode),
              let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let raw = json["data"] as? [[String: Any]] else { return [] }
        return raw.compactMap { coin in
            guard let id = coin["id"] as? Int, let symbol = coin["symbol"] as? String,
                  let name = coin["name"] as? String else { return nil }
            return CoinEntry(id: id, symbol: symbol, name: name, rank: coin["rank"] as? Int ?? Int.max)
        }
    }
}
