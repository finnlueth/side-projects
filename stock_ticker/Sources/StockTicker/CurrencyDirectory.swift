import Foundation
import Combine

/// The ~30 currencies the ECB publishes, for validating and suggesting pairs. Ships with a built-in
/// list and refreshes it from Frankfurter weekly.
@MainActor
final class CurrencyDirectory: ObservableObject {
    @Published private(set) var currencies: [String: String] = CurrencyDirectory.builtIn

    /// The quote currencies offered first when only a base has been typed.
    static let popularQuotes = ["USD", "EUR", "GBP", "JPY", "CHF"]

    private static let maxCacheAge: TimeInterval = 7 * 24 * 3600
    private var isLoading = false

    private let cacheURL: URL = {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        return base.appendingPathComponent("StockTicker/currencies.json")
    }()

    func loadIfNeeded() {
        guard !isLoading else { return }
        if let cached = try? Data(contentsOf: cacheURL),
           let decoded = try? JSONDecoder().decode([String: String].self, from: cached), !decoded.isEmpty {
            currencies = decoded
        }
        let cacheAge = (try? cacheURL.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate)
            .map { Date().timeIntervalSince($0) } ?? .infinity
        guard cacheAge > Self.maxCacheAge else { return }

        isLoading = true
        Task {
            defer { isLoading = false }
            guard let (data, response) = try? await URLSession.shared.data(from: URL(string: "https://api.frankfurter.dev/v1/currencies")!),
                  (response as? HTTPURLResponse).map({ (200..<300).contains($0.statusCode) }) ?? false,
                  let fetched = try? JSONDecoder().decode([String: String].self, from: data), !fetched.isEmpty else { return }
            currencies = fetched
            try? FileManager.default.createDirectory(at: cacheURL.deletingLastPathComponent(), withIntermediateDirectories: true)
            try? data.write(to: cacheURL, options: .atomic)
        }
    }

    func name(for pair: CurrencyPair) -> String? {
        guard let base = currencies[pair.base], let quote = currencies[pair.quote] else { return nil }
        return "\(base) → \(quote)"
    }

    func isKnown(_ pair: CurrencyPair) -> Bool {
        currencies[pair.base] != nil && currencies[pair.quote] != nil && pair.base != pair.quote
    }

    /// "EUR/USD", "EURUSD", "eur usd", "eur-usd" → EUR/USD. Several pairs may be separated by commas.
    static func parsePairs(_ input: String) -> [CurrencyPair] {
        input.uppercased().split(separator: ",").compactMap { token -> CurrencyPair? in
            let letters = token.filter(\.isLetter)
            guard letters.count == 6 else { return nil }
            return CurrencyPair(base: String(letters.prefix(3)), quote: String(letters.suffix(3)))
        }
    }

    /// Pairs matching a partial "EUR", "EUR/G", "euro" or "eurusd" query.
    func suggestions(for query: String, excluding: Set<String>, limit: Int = 6) -> [CurrencyPair] {
        let q = query.trimmingCharacters(in: .whitespaces).uppercased()
        guard !q.isEmpty else { return [] }
        let separators = CharacterSet(charactersIn: "/ -")
        let parts = q.components(separatedBy: separators).filter { !$0.isEmpty }
        let (basePart, quotePart): (String, String)
        if parts.count >= 2 {
            (basePart, quotePart) = (parts[0], parts[1])
        } else if q.count > 3, q.allSatisfy(\.isLetter), currencies[String(q.prefix(3))] != nil {
            (basePart, quotePart) = (String(q.prefix(3)), String(q.dropFirst(3)))
        } else {
            (basePart, quotePart) = (q, "")
        }

        let codes = currencies.keys.sorted()
        let bases = codes.filter { $0.hasPrefix(basePart) || (currencies[$0] ?? "").uppercased().hasPrefix(basePart) }
        let quotes = quotePart.isEmpty ? Self.popularQuotes : codes.filter { $0.hasPrefix(quotePart) }

        var result: [CurrencyPair] = []
        for base in bases {
            for quote in quotes where quote != base {
                let pair = CurrencyPair(base: base, quote: quote)
                if !excluding.contains(pair.symbol) { result.append(pair) }
                if result.count == limit { return result }
            }
        }
        return result
    }

    private static let builtIn: [String: String] = [
        "AUD": "Australian Dollar", "BGN": "Bulgarian Lev", "BRL": "Brazilian Real", "CAD": "Canadian Dollar",
        "CHF": "Swiss Franc", "CNY": "Chinese Renminbi Yuan", "CZK": "Czech Koruna", "DKK": "Danish Krone",
        "EUR": "Euro", "GBP": "British Pound", "HKD": "Hong Kong Dollar", "HUF": "Hungarian Forint",
        "IDR": "Indonesian Rupiah", "ILS": "Israeli New Sheqel", "INR": "Indian Rupee", "ISK": "Icelandic Króna",
        "JPY": "Japanese Yen", "KRW": "South Korean Won", "MXN": "Mexican Peso", "MYR": "Malaysian Ringgit",
        "NOK": "Norwegian Krone", "NZD": "New Zealand Dollar", "PHP": "Philippine Peso", "PLN": "Polish Złoty",
        "RON": "Romanian Leu", "SEK": "Swedish Krona", "SGD": "Singapore Dollar", "THB": "Thai Baht",
        "TRY": "Turkish Lira", "USD": "United States Dollar", "ZAR": "South African Rand",
    ]
}
