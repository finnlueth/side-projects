import Foundation
import Combine

struct SymbolEntry: Codable, Hashable {
    let symbol: String
    let name: String
    let exchange: String
}

/// Offline stock symbol search backed by Alpaca's asset list (~14k active US equities), cached on disk
/// and refreshed weekly. Alpha Vantage's SYMBOL_SEARCH would cost a quota request per keystroke, so the
/// Alpaca list is used for suggestions regardless of the selected quote source.
@MainActor
final class SymbolDirectory: ObservableObject {
    @Published private(set) var entries: [SymbolEntry] = []

    private static let maxCacheAge: TimeInterval = 7 * 24 * 3600
    private var isLoading = false

    private let cacheURL: URL = {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        return base.appendingPathComponent("StockTicker/symbols.json")
    }()

    func loadIfNeeded(keyID: String, secretKey: String) {
        guard !isLoading else { return }
        if entries.isEmpty, let cached = try? Data(contentsOf: cacheURL),
           let decoded = try? JSONDecoder().decode([SymbolEntry].self, from: cached) {
            entries = decoded
        }
        let cacheAge = (try? cacheURL.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate)
            .map { Date().timeIntervalSince($0) } ?? .infinity
        guard entries.isEmpty || cacheAge > Self.maxCacheAge else { return }
        guard !keyID.isEmpty, !secretKey.isEmpty else { return }

        isLoading = true
        Task {
            defer { isLoading = false }
            guard let fetched = try? await Self.fetch(keyID: keyID, secretKey: secretKey), !fetched.isEmpty else { return }
            entries = fetched
            try? FileManager.default.createDirectory(at: cacheURL.deletingLastPathComponent(), withIntermediateDirectories: true)
            try? JSONEncoder().encode(fetched).write(to: cacheURL, options: .atomic)
        }
    }

    /// Symbol prefix matches first (exact match on top), then company-name matches; listed exchanges
    /// before OTC; shorter symbols first.
    func suggestions(for query: String, excluding: Set<String>, limit: Int = 6) -> [SymbolEntry] {
        let q = query.trimmingCharacters(in: .whitespaces).uppercased()
        guard !q.isEmpty, q.allSatisfy({ $0.isLetter || $0.isNumber || ".- ".contains($0) }) else { return [] }
        let needle = q.lowercased()

        func score(_ entry: SymbolEntry) -> Int? {
            if entry.symbol == q { return 0 }
            if entry.symbol.hasPrefix(q) { return 1 }
            let name = entry.name.lowercased()
            if name.hasPrefix(needle) { return 2 }
            if q.count >= 2, name.contains(needle) { return 3 }
            return nil
        }

        return entries.lazy
            .filter { !excluding.contains($0.symbol) }
            .compactMap { entry in score(entry).map { (entry, $0) } }
            .sorted { a, b in
                if a.1 != b.1 { return a.1 < b.1 }
                let aOTC = a.0.exchange == "OTC", bOTC = b.0.exchange == "OTC"
                if aOTC != bOTC { return !aOTC }
                if a.0.symbol.count != b.0.symbol.count { return a.0.symbol.count < b.0.symbol.count }
                return a.0.symbol < b.0.symbol
            }
            .prefix(limit)
            .map(\.0)
    }

    private static func fetch(keyID: String, secretKey: String) async throws -> [SymbolEntry] {
        // Paper keys ("PK…") only work against the paper trading host; the asset list is the same.
        let host = keyID.hasPrefix("PK") ? "paper-api.alpaca.markets" : "api.alpaca.markets"
        var request = URLRequest(url: URL(string: "https://\(host)/v2/assets?status=active&asset_class=us_equity")!)
        request.setValue(keyID, forHTTPHeaderField: "APCA-API-KEY-ID")
        request.setValue(secretKey, forHTTPHeaderField: "APCA-API-SECRET-KEY")
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode),
              let raw = try JSONSerialization.jsonObject(with: data) as? [[String: Any]] else { return [] }
        return raw.compactMap { asset in
            guard asset["tradable"] as? Bool == true,
                  let symbol = asset["symbol"] as? String,
                  let name = asset["name"] as? String else { return nil }
            return SymbolEntry(symbol: symbol, name: name, exchange: asset["exchange"] as? String ?? "")
        }
    }
}
